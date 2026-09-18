/*
 * M2 ref-image visual conditioning -- vision tower device kernels.
 *
 * Device kernels for the Qwen3-VL vision tower (see src/model/vision.c for
 * the host orchestration and docs/REF_IMAGE_NATIVE_IMPLEMENTATION.md).
 * Compiled with nvcc; exposed through a C ABI consumed by vision.c.
 */

#include "cuda_internal.h"
#include "vision_kernels.h"

#include <stdio.h>

/* patch_embed: y[n, 1152] = x[n, 1536] x W[1152, 1536]^T + b[1152].
 * The Conv3d over the patchified input is a plain matmul. */
__global__ void hd_vision_patch_kernel(const uint16_t *__restrict__ x,
                                       const uint16_t *__restrict__ w,
                                       const uint16_t *__restrict__ bias,
                                       uint16_t *__restrict__ y,
                                       int n, int in_dim, int out_dim) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= n || col >= out_dim) return;
    const uint16_t *xr = x + (size_t)row * in_dim;
    const uint16_t *wr = w + (size_t)col * in_dim;
    float acc = 0.0f;
    for (int k = 0; k < in_dim; k++)
        acc += hd_dev_bf16_to_f32(xr[k]) * hd_dev_bf16_to_f32(wr[k]);
    if (bias) acc += hd_dev_bf16_to_f32(bias[col]);
    y[(size_t)row * out_dim + col] = hd_dev_f32_to_bf16(acc);
}

void hd_vision_patch(const void *x, const void *w, const void *bias,
                     void *y, int n, int in_dim, int out_dim) {
    if (!x || !w || !y || n <= 0 || in_dim <= 0 || out_dim <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_patch: bad args");
        return;
    }
    /* row = blockIdx.x (one block per input row), col = blockIdx.y*blockDim.x
     * + threadIdx.x. Grid must cover ALL n rows. */
    int threads = 256;
    dim3 blk(threads);
    dim3 grd(n, (out_dim + threads - 1) / threads);
    hd_vision_patch_kernel<<<grd, blk>>>(
        (const uint16_t *)x, (const uint16_t *)w, (const uint16_t *)bias,
        (uint16_t *)y, n, in_dim, out_dim);
}

/* LayerNorm over the last dim (eps 1e-6), bf16 in/out, fp32 internal. */
__global__ void hd_vision_layernorm_kernel(const uint16_t *__restrict__ x,
                                           const uint16_t *__restrict__ w,
                                           const uint16_t *__restrict__ b,
                                           uint16_t *__restrict__ y,
                                           int rows, int cols, float eps) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const uint16_t *xr = x + (size_t)row * cols;
    uint16_t *yr = y + (size_t)row * cols;
    int t = threadIdx.x;
    int nt = blockDim.x;
    float mean = 0.0f, var = 0.0f;
    for (int i = t; i < cols; i += nt) {
        float v = hd_dev_bf16_to_f32(xr[i]);
        mean += v;
        var += v * v;
    }
    __shared__ float s_m[256], s_v[256];
    s_m[t] = mean; s_v[t] = var;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
        if (t < s) { s_m[t] += s_m[t + s]; s_v[t] += s_v[t + s]; }
        __syncthreads();
    }
    if (t == 0) {
        float inv = 1.0f / (float)cols;
        s_m[0] *= inv;
        s_v[0] = s_v[0] * inv - s_m[0] * s_m[0];
    }
    __syncthreads();
    float inv_std = rsqrtf(s_v[0] + eps);
    for (int i = t; i < cols; i += nt) {
        float v = (hd_dev_bf16_to_f32(xr[i]) - s_m[0]) * inv_std;
        float wv = hd_dev_bf16_to_f32(w[i]);
        float bv = b ? hd_dev_bf16_to_f32(b[i]) : 0.0f;
        yr[i] = hd_dev_f32_to_bf16(v * wv + bv);
    }
}

void hd_vision_layernorm(const void *x, const void *w, const void *b,
                         void *y, int rows, int cols, float eps) {
    if (!x || !w || !y || rows <= 0 || cols <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_layernorm: bad args");
        return;
    }
    hd_vision_layernorm_kernel<<<rows, 256>>>(
        (const uint16_t *)x, (const uint16_t *)w, (const uint16_t *)b,
        (uint16_t *)y, rows, cols, eps);
}

/* GELU pytorch_tanh: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))). */
__global__ void hd_vision_gelu_kernel(const uint16_t *__restrict__ x,
                                      uint16_t *__restrict__ y, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = hd_dev_bf16_to_f32(x[i]);
    float c = 0.7978845608028654f;   /* sqrt(2/pi) */
    float g = c * (v + 0.044715f * v * v * v);
    float t = tanhf(g);
    y[i] = hd_dev_f32_to_bf16(0.5f * v * (1.0f + t));
}

void hd_vision_gelu(const void *x, void *y, size_t n) {
    if (!x || !y || n == 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_gelu: bad args");
        return;
    }
    size_t threads = 256;
    size_t blocks = (n + threads - 1) / threads;
    hd_vision_gelu_kernel<<<blocks, threads>>>(
        (const uint16_t *)x, (uint16_t *)y, n);
}

/* pos_embed 4-corner bilinear interpolation (fast_pos_embed_interpolate).
 * Each output row is a weighted sum of 4 pos_embed rows. */
__global__ void hd_vision_pos_interp_kernel(
        const uint16_t *__restrict__ pos_embed,   /* [2304, 1152] */
        const int *__restrict__ idx,              /* [4, n] */
        const float *__restrict__ wgt,            /* [4, n] */
        uint16_t *__restrict__ y, int n, int hidden) {
    int row = blockIdx.x;
    if (row >= n) return;
    const uint16_t *pe = pos_embed;
    uint16_t *yr = y + (size_t)row * hidden;
    int t = threadIdx.x;
    int nt = blockDim.x;
    for (int c = t; c < hidden; c += nt) {
        float acc = 0.0f;
        for (int k = 0; k < 4; k++) {
            int i = idx[(size_t)k * n + row];
            float w = wgt[(size_t)k * n + row];
            acc += w * hd_dev_bf16_to_f32(pe[(size_t)i * hidden + c]);
        }
        yr[c] = hd_dev_f32_to_bf16(acc);
    }
}

void hd_vision_pos_interp(const void *pos_embed, const int *idx,
                          const float *wgt, void *y, int n, int hidden) {
    if (!pos_embed || !idx || !wgt || !y || n <= 0 || hidden <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_pos_interp: bad args");
        return;
    }
    hd_vision_pos_interp_kernel<<<n, 256>>>(
        (const uint16_t *)pos_embed, idx, wgt, (uint16_t *)y, n, hidden);
}

/* rot_pos_emb: build the rotary freq table [n, 36] from (row,col) coords.
 * freq_table[coord, k] = inv_freq[k] * coord. Output row = cat([row_freq,
 * col_freq]) each of length 18 -> [n, 36]. */
__global__ void hd_vision_rot_kernel(
        const float *__restrict__ inv_freq,   /* [18] */
        const int *__restrict__ coords,       /* [2, n] (row, col) */
        uint16_t *__restrict__ y, int n, int half) {
    int row = blockIdx.x;
    if (row >= n) return;
    int t = threadIdx.x;
    int nt = blockDim.x;
    int r = coords[row], c = coords[n + row];
    uint16_t *yr = y + (size_t)row * (2 * half);
    for (int k = t; k < half; k += nt) {
        float f = inv_freq[k];
        yr[k] = hd_dev_f32_to_bf16(f * (float)r);
        yr[half + k] = hd_dev_f32_to_bf16(f * (float)c);
    }
}

void hd_vision_rot(const float *inv_freq, const int *coords, void *y,
                   int n, int half) {
    if (!inv_freq || !coords || !y || n <= 0 || half <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_rot: bad args");
        return;
    }
    hd_vision_rot_kernel<<<n, 256>>>(inv_freq, coords, (uint16_t *)y, n, half);
}

/* cos/sin from the bf16 rot table [n, 36]: emb = cat([rot, rot], -1) ->
 * [n, 72]; cos = cos(emb), sin = sin(emb) fp32. */
__global__ void hd_vision_rot_cos_sin_kernel(
        const uint16_t *__restrict__ rot,   /* [n, half] */
        float *__restrict__ cosd,           /* [n, 2*half] */
        float *__restrict__ sind,           /* [n, 2*half] */
        int n, int half) {
    int row = blockIdx.x;
    if (row >= n) return;
    int t = threadIdx.x;
    int nt = blockDim.x;
    const uint16_t *rr = rot + (size_t)row * half;
    float *cr = cosd + (size_t)row * (2 * half);
    float *sr = sind + (size_t)row * (2 * half);
    for (int k = t; k < half; k += nt) {
        float v = hd_dev_bf16_to_f32(rr[k]);
        float c = cosf(v), s = sinf(v);
        cr[k] = c; cr[half + k] = c;
        sr[k] = s; sr[half + k] = s;
    }
}

void hd_vision_rot_cos_sin(const void *rot, float *cosd, float *sind,
                           int n, int half) {
    if (!rot || !cosd || !sind || n <= 0 || half <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_rot_cos_sin: bad args");
        return;
    }
    hd_vision_rot_cos_sin_kernel<<<n, 256>>>(
        (const uint16_t *)rot, cosd, sind, n, half);
}

/* spatial merge (2x2 unshuffle): in [gh*gw, 1152] -> out [gh/2*gw/2, 4608].
 * out[oh, ow, :] = cat of in[2*oh, 2*ow], in[2*oh, 2*ow+1],
 *                       in[2*oh+1, 2*ow], in[2*oh+1, 2*ow+1]  (each 1152) */
__global__ void hd_vision_spatial_merge_kernel(
        const uint16_t *__restrict__ in, uint16_t *__restrict__ out,
        int gh, int gw, int hidden) {
    int out_row = blockIdx.x;
    int mw = gw / 2;
    int oh = out_row / mw;
    int ow = out_row % mw;
    int t = threadIdx.x;
    int nt = blockDim.x;
    int m = 2;
    uint16_t *orow = out + (size_t)out_row * (hidden * m * m);
    for (int c = t; c < hidden; c += nt) {
        for (int dh = 0; dh < m; dh++) {
            for (int dw = 0; dw < m; dw++) {
                int in_row = (2 * oh + dh) * gw + (2 * ow + dw);
                uint16_t v = in[(size_t)in_row * hidden + c];
                int slot = (dh * m + dw) * hidden + c;
                orow[slot] = v;
            }
        }
    }
}

void hd_vision_spatial_merge(const void *in, void *out, int gh, int gw,
                             int hidden) {
    if (!in || !out || gh <= 0 || gw <= 0 || hidden <= 0 || gh % 2 || gw % 2) {
        snprintf(hd_cuda_errbuf(), 512, "vision_spatial_merge: bad args");
        return;
    }
    /* one block per merged output row: out_n = (gh/2)*(gw/2) */
    int out_n = (gh / 2) * (gw / 2);
    hd_vision_spatial_merge_kernel<<<out_n, 256>>>(
        (const uint16_t *)in, (uint16_t *)out, gh, gw, hidden);
}

/* qkv split: qkv [n, 3*Hd*D] -> q/k/v [Hd, n, D] head-major.
 * qkv layout per torch: reshape(n, 3, Hd, D).permute(1,0,2,3). */
__global__ void hd_vision_qkv_split_kernel(
        const uint16_t *__restrict__ qkv,   /* [n, 3*Hd*D] */
        uint16_t *__restrict__ q,           /* [Hd, n, D] */
        uint16_t *__restrict__ k,
        uint16_t *__restrict__ v,
        int n, int heads, int d) {
    long total = (long)n * heads * d;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dd = idx % d;
    long sh = idx / d;
    int s = (int)(sh % n);
    int h = (int)(sh / n);
    long src = ((long)s * 3 + 0) * heads * d + (long)h * d + dd;
    q[(long)h * n * d + (long)s * d + dd] = qkv[src];
    src = ((long)s * 3 + 1) * heads * d + (long)h * d + dd;
    k[(long)h * n * d + (long)s * d + dd] = qkv[src];
    src = ((long)s * 3 + 2) * heads * d + (long)h * d + dd;
    v[(long)h * n * d + (long)s * d + dd] = qkv[src];
}

void hd_vision_qkv_split(const void *qkv, void *q, void *k, void *v,
                         int n, int heads, int d) {
    if (!qkv || !q || !k || !v || n <= 0 || heads <= 0 || d <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_qkv_split: bad args");
        return;
    }
    long total = (long)n * heads * d;
    hd_vision_qkv_split_kernel<<<(total + 255) / 256, 256>>>(
        (const uint16_t *)qkv, (uint16_t *)q, (uint16_t *)k, (uint16_t *)v,
        n, heads, d);
}

/* attention output [Hd, n, D] head-major -> seq-major [n, Hd*D]. */
__global__ void hd_vision_attn_merge_kernel(
        const uint16_t *__restrict__ hm,   /* [Hd, n, D] */
        uint16_t *__restrict__ sm,         /* [n, Hd*D] */
        int heads, int n, int d) {
    long total = (long)heads * n * d;
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int dd = idx % d;
    long sh = idx / d;
    int s = (int)(sh % n);
    int h = (int)(sh / n);
    sm[(long)s * heads * d + (long)h * d + dd] =
        hm[(long)h * n * d + (long)s * d + dd];
}

void hd_vision_attn_merge(const void *hm, void *sm, int heads, int n, int d) {
    if (!hm || !sm || heads <= 0 || n <= 0 || d <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_attn_merge: bad args");
        return;
    }
    long total = (long)heads * n * d;
    hd_vision_attn_merge_kernel<<<(total + 255) / 256, 256>>>(
        (const uint16_t *)hm, (uint16_t *)sm, heads, n, d);
}

/* masked_scatter: for each row where mask[row]!=0, copy the next row of
 * `src` into out[row]; else copy dst[row]. */
__global__ void hd_vision_masked_scatter_kernel(
        const uint16_t *__restrict__ dst, const uint8_t *__restrict__ mask,
        const uint16_t *__restrict__ src, uint16_t *__restrict__ out,
        int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int t = threadIdx.x;
    int nt = blockDim.x;
    if (mask[row]) {
        int src_row = 0;
        for (int r = 0; r < row; r++) src_row += mask[r];
        const uint16_t *sr = src + (size_t)src_row * cols;
        uint16_t *dr = out + (size_t)row * cols;
        for (int c = t; c < cols; c += nt) dr[c] = sr[c];
    } else {
        const uint16_t *dd = dst + (size_t)row * cols;
        uint16_t *dr = out + (size_t)row * cols;
        for (int c = t; c < cols; c += nt) dr[c] = dd[c];
    }
}

void hd_vision_masked_scatter(const void *dst, const uint8_t *mask,
                              const void *src, void *out, int rows, int cols) {
    if (!dst || !mask || !src || !out || rows <= 0 || cols <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_masked_scatter: bad args");
        return;
    }
    hd_vision_masked_scatter_kernel<<<rows, 256>>>(
        (const uint16_t *)dst, mask, (const uint16_t *)src, (uint16_t *)out,
        rows, cols);
}

/* deepstack injection: out[row] = in[row] + emb[src_row] for masked rows. */
__global__ void hd_vision_deepstack_inject_kernel(
        const uint16_t *__restrict__ in, const uint8_t *__restrict__ mask,
        const uint16_t *__restrict__ emb, uint16_t *__restrict__ out,
        int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int t = threadIdx.x;
    int nt = blockDim.x;
    if (mask[row]) {
        int src_row = 0;
        for (int r = 0; r < row; r++) src_row += mask[r];
        const uint16_t *er = emb + (size_t)src_row * cols;
        const uint16_t *ir = in + (size_t)row * cols;
        uint16_t *orow = out + (size_t)row * cols;
        for (int c = t; c < cols; c += nt) {
            float a = hd_dev_bf16_to_f32(ir[c]);
            float b = hd_dev_bf16_to_f32(er[c]);
            orow[c] = hd_dev_f32_to_bf16(a + b);
        }
    } else {
        const uint16_t *ir = in + (size_t)row * cols;
        uint16_t *orow = out + (size_t)row * cols;
        for (int c = t; c < cols; c += nt) orow[c] = ir[c];
    }
}

void hd_vision_deepstack_inject(const void *in, const uint8_t *mask,
                                const void *emb, void *out, int rows,
                                int cols) {
    if (!in || !mask || !emb || !out || rows <= 0 || cols <= 0) {
        snprintf(hd_cuda_errbuf(), 512, "vision_deepstack_inject: bad args");
        return;
    }
    hd_vision_deepstack_inject_kernel<<<rows, 256>>>(
        (const uint16_t *)in, mask, (const uint16_t *)emb, (uint16_t *)out,
        rows, cols);
}