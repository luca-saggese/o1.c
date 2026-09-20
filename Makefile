# HiDream GB10 inference engine — M1 native build

CC      ?= gcc
NVCC    ?= nvcc
CFLAGS  ?= -O2 -g -std=c11 -Wall -Wextra
CPPFLAGS += -Iinclude -Isrc/io -Isrc/model -Isrc/cuda -Isrc/runtime -Isrc/image

CUDA_HOME  ?= /usr/local/cuda
CUDA_CPPFLAGS := -I$(CUDA_HOME)/include
CUDA_LDFLAGS  := -L$(CUDA_HOME)/lib64 -lcudart -lnvrtc
CUBLAS_LDFLAGS := -lcublas -lcublasLt

# cuDNN SDPA backend (M2 pre-baseline). The C++ Frontend is vendored under
# third_party/cudnn-frontend (v1.22.1, recommended for cuDNN 9.20). The
# runtime libcudnn.so comes from the nvidia-cudnn pip package (no Python in
# the runtime; only the .so is linked). Override CUDNN_HOME to point at a
# native cuDNN install if available.
CUDNN_HOME ?= /home/lvx/.local/lib/python3.12/site-packages/nvidia/cudnn
CUDNN_CPPFLAGS := -Ithird_party/cudnn-frontend/include -I$(CUDNN_HOME)/include
CUDNN_LDFLAGS  := -L$(CUDNN_HOME)/lib -lcudnn

CORE_SRCS := src/model/model.c src/io/json.c src/io/sha256.c src/io/safetensors.c src/io/gguf.c

# Model used by the resident-engine preload test (override on the command line).
ENGINE_MODEL ?= artifacts/models/hidream-o1-dev-bf16.gguf

BIN        := build/hidream
SERVER_BIN := build/hidream-server
TEST_BIN   := build/test_model_loader
TEST_W_BIN := build/test_weights
CUBIN      := build/obj/cuda
CUDA_OBJS  := $(CUBIN)/support.o $(CUBIN)/norm.o $(CUBIN)/act.o \
              $(CUBIN)/gemm.o $(CUBIN)/rope.o $(CUBIN)/attn.o \
              $(CUBIN)/residual.o $(CUBIN)/embed.o $(CUBIN)/sched.o \
              $(CUBIN)/hd_cudnn_sdpa.o $(CUBIN)/lora_merge.o \
              $(CUBIN)/vision_kernels.o
TEST_P_BIN := build/test_primitives
TEST_P_SRCS := tests/unit/test_primitives.c $(CORE_SRCS)
TEST_P_OBJS := $(TEST_P_SRCS:.c=.o)
TEST_TOK_BIN := build/test_tokenizer
TEST_TOK_SRCS := tests/unit/test_tokenizer.c src/model/tokenizer.c src/model/model.c src/io/json.c
TEST_TOK_OBJS := $(TEST_TOK_SRCS:.c=.o)

TEST_BLOCK_BIN := build/test_block
TEST_BLOCK_SRCS := tests/unit/test_block.c src/model/block.c $(CORE_SRCS) src/model/weights.c
TEST_BLOCK_OBJS := $(TEST_BLOCK_SRCS:.c=.o)

TEST_SDPA_BIN := build/test_sdpa_block
TEST_SDPA_SRCS := tests/unit/test_sdpa_block.c src/model/block.c $(CORE_SRCS) src/model/weights.c
TEST_SDPA_OBJS := $(TEST_SDPA_SRCS:.c=.o)

TEST_SDPA_FWD_BIN := build/test_sdpa_forward
TEST_SDPA_FWD_SRCS := tests/unit/test_sdpa_forward.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_SDPA_FWD_OBJS := $(TEST_SDPA_FWD_SRCS:.c=.o)

BENCH_SDPA2_BIN := build/bench_sdpa_twopass
BENCH_SDPA2_SRCS := tests/unit/bench_sdpa_twopass.cu src/runtime/sequence.c $(CORE_SRCS)
BENCH_SDPA2_OBJS := tests/unit/bench_sdpa_twopass.o
BENCH_BLOCK_BIN := build/bench_block
BENCH_BLOCK_SRCS := tests/unit/bench_block.c src/model/block.c $(CORE_SRCS) src/model/weights.c
BENCH_BLOCK_OBJS := $(BENCH_BLOCK_SRCS:.c=.o)

BENCH_GEMM_BIN := build/bench_gemm
BENCH_GEMM_SRCS := tests/unit/bench_gemm.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
BENCH_GEMM_OBJS := $(BENCH_GEMM_SRCS:.c=.o)

TEST_FF_BIN := build/test_full_forward
TEST_FF_SRCS := tests/unit/test_full_forward.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_FF_OBJS := $(TEST_FF_SRCS:.c=.o)

TEST_VISION_BIN := build/test_vision
TEST_VISION_SRCS := tests/unit/test_vision.c src/model/vision.c $(CORE_SRCS) src/model/weights.c
TEST_VISION_OBJS := $(TEST_VISION_SRCS:.c=.o)

TEST_GEMM_BIN := build/test_gemm_smoke
TEST_GEMM_SRCS := tests/unit/test_gemm_smoke.c
TEST_GEMM_OBJS := $(TEST_GEMM_SRCS:.c=.o)

TEST_GGUF_BIN := build/test_gguf
TEST_GGUF_SRCS := tests/unit/test_gguf.c $(CORE_SRCS)
TEST_GGUF_OBJS := $(TEST_GGUF_SRCS:.c=.o)

TEST_REPLAY_BIN := build/test_layer_replay
TEST_REPLAY_SRCS := tests/unit/test_layer_replay.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_REPLAY_OBJS := $(TEST_REPLAY_SRCS:.c=.o)

TEST_FH_BIN := build/test_final_head
TEST_FH_SRCS := tests/unit/test_final_head.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_FH_OBJS := $(TEST_FH_SRCS:.c=.o)

TEST_TAIL_BIN := build/test_block_tail
TEST_TAIL_SRCS := tests/unit/test_block_tail.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_TAIL_OBJS := $(TEST_TAIL_SRCS:.c=.o)

TEST_M15_BIN := build/test_m1_5_scheduler
TEST_M15_SRCS := tests/unit/test_m1_5_scheduler.c src/model/block.c src/model/forward.c src/model/scheduler.c $(CORE_SRCS) src/model/weights.c
TEST_M15_OBJS := $(TEST_M15_SRCS:.c=.o)

SRCS      := src/main.c $(CORE_SRCS) src/model/weights.c src/model/block.c \
             src/model/forward.c src/model/scheduler.c src/model/tokenizer.c \
             src/model/lora.c src/model/vision.c \
             src/runtime/sequence.c src/runtime/request.c src/runtime/decode.c \
             src/runtime/ref_alias.c src/runtime/torch_rng.c \
             src/runtime/generate.c src/runtime/engine.c src/io/png_wrap.c \
             src/image/hd_image.c src/image/layout.c src/runtime/o1_timing.c
OBJS      := $(SRCS:.c=.o)

TEST_SRCS   := tests/unit/test_model_loader.c src/model/model.c src/io/json.c
TEST_OBJS   := $(TEST_SRCS:.c=.o)
TEST_W_SRCS := tests/unit/test_weights.c $(CORE_SRCS) src/model/weights.c
TEST_W_OBJS := $(TEST_W_SRCS:.c=.o)

all: $(BIN)

test: $(TEST_BIN) $(TEST_W_BIN) $(TEST_P_BIN) $(TEST_TOK_BIN) $(TEST_BLOCK_BIN) $(TEST_FF_BIN) $(TEST_TAIL_BIN)
	./$(TEST_BIN)
	./$(TEST_W_BIN)
	./$(TEST_P_BIN)
	./$(TEST_TOK_BIN)
	./$(TEST_BLOCK_BIN)
	./$(TEST_FF_BIN)

test-primitives: $(TEST_P_BIN)

test-block: $(TEST_BLOCK_BIN)

test-full-forward: $(TEST_FF_BIN)
	./$(TEST_FF_BIN)

test-gemm-smoke: $(TEST_GEMM_BIN)

test-gguf: $(TEST_GGUF_BIN)
	./$(TEST_GGUF_BIN) $(GGUF_PATH) $(GGUF_ST_DIR)

test-layer-replay: $(TEST_REPLAY_BIN)

test-final-head: $(TEST_FH_BIN)
	./$(TEST_FH_BIN)

test-m15: $(TEST_M15_BIN)
	./$(TEST_M15_BIN)

TEST_M17_BIN := build/test_m1_7_dev
TEST_M17_SRCS := tests/unit/test_m1_7_dev.c src/model/block.c src/model/forward.c src/model/scheduler.c $(CORE_SRCS) src/model/weights.c
TEST_M17_OBJS := $(TEST_M17_SRCS:.c=.o)
$(TEST_M17_BIN): $(TEST_M17_OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $(TEST_M17_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

test-m17: $(TEST_M17_BIN)
	./$(TEST_M17_BIN)

TEST_RNG_BIN := build/test_torch_rng
TEST_RNG_SRCS := tests/unit/test_torch_rng.c src/runtime/torch_rng.c
TEST_RNG_OBJS := $(TEST_RNG_SRCS:.c=.o)
$(TEST_RNG_BIN): $(TEST_RNG_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_RNG_OBJS) -lm

test-rng: $(TEST_RNG_BIN)
	./$(TEST_RNG_BIN)

TEST_PNG_BIN := build/test_png_roundtrip
TEST_PNG_SRCS := tests/unit/test_png_roundtrip.c src/io/png_wrap.c
TEST_PNG_OBJS := $(TEST_PNG_SRCS:.c=.o)
$(TEST_PNG_BIN): $(TEST_PNG_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_PNG_OBJS) -lm

test-png: $(TEST_PNG_BIN)
	./$(TEST_PNG_BIN)

TEST_IMG_BIN := build/test_image
TEST_IMG_SRCS := tests/unit/test_image.c src/image/hd_image.c src/io/png_wrap.c src/model/model.c src/io/json.c
TEST_IMG_OBJS := $(TEST_IMG_SRCS:.c=.o)
$(TEST_IMG_BIN): $(TEST_IMG_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_IMG_OBJS) -lm -l:libjpeg.so.8

test-image: $(TEST_IMG_BIN)
	./$(TEST_IMG_BIN)

TEST_LAYOUT_BIN := build/test_layout_pipe
TEST_LAYOUT_SRCS := tests/unit/layout_pipe.c src/image/layout.c src/image/hd_image.c src/io/json.c src/model/model.c src/io/png_wrap.c
TEST_LAYOUT_OBJS := $(TEST_LAYOUT_SRCS:.c=.o)
$(TEST_LAYOUT_BIN): $(TEST_LAYOUT_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_LAYOUT_OBJS) -lm -l:libjpeg.so.8

test-layout: $(TEST_LAYOUT_BIN)
	./$(TEST_LAYOUT_BIN)

TEST_SEQ_BIN := build/test_sequence
TEST_SEQ_SRCS := tests/unit/test_sequence.c src/runtime/sequence.c src/runtime/request.c src/model/tokenizer.c $(CORE_SRCS)
TEST_SEQ_OBJS := $(TEST_SEQ_SRCS:.c=.o)
$(TEST_SEQ_BIN): $(TEST_SEQ_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_SEQ_OBJS) -lm

test-seq: $(TEST_SEQ_BIN)
	./$(TEST_SEQ_BIN)

TEST_SEQREF_BIN := build/test_seq_ref
TEST_SEQREF_SRCS := tests/unit/test_seq_ref.c src/runtime/sequence.c src/runtime/request.c src/model/tokenizer.c $(CORE_SRCS)
TEST_SEQREF_OBJS := $(TEST_SEQREF_SRCS:.c=.o)
$(TEST_SEQREF_BIN): $(TEST_SEQREF_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_SEQREF_OBJS) -lm

test-seq-ref: $(TEST_SEQREF_BIN)
	./$(TEST_SEQREF_BIN)

TEST_REFALIAS_BIN := build/test_ref_alias
TEST_REFALIAS_SRCS := tests/unit/test_ref_alias.c src/runtime/ref_alias.c $(CORE_SRCS)
TEST_REFALIAS_OBJS := $(TEST_REFALIAS_SRCS:.c=.o)
$(TEST_REFALIAS_BIN): $(TEST_REFALIAS_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_REFALIAS_OBJS) -lm

test-ref-alias: $(TEST_REFALIAS_BIN)
	./$(TEST_REFALIAS_BIN)

TEST_SEQDIAG_BIN := build/test_seq_diag
TEST_SEQDIAG_SRCS := tests/unit/test_seq_diag.c src/runtime/sequence.c src/runtime/request.c src/model/tokenizer.c $(CORE_SRCS)
TEST_SEQDIAG_OBJS := $(TEST_SEQDIAG_SRCS:.c=.o)
$(TEST_SEQDIAG_BIN): $(TEST_SEQDIAG_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_SEQDIAG_OBJS) -lm

test-seq-diag: $(TEST_SEQDIAG_BIN)
	./$(TEST_SEQDIAG_BIN)

TEST_SEQPROF_BIN := build/seq_profiles
TEST_SEQPROF_SRCS := tests/unit/seq_profiles.c src/runtime/sequence.c src/runtime/request.c src/model/tokenizer.c $(CORE_SRCS)
TEST_SEQPROF_OBJS := $(TEST_SEQPROF_SRCS:.c=.o)
$(TEST_SEQPROF_BIN): $(TEST_SEQPROF_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_SEQPROF_OBJS) -lm

test-seq-profiles: $(TEST_SEQPROF_BIN)
	./$(TEST_SEQPROF_BIN)

TEST_DECODE_BIN := build/test_decode
TEST_DECODE_SRCS := tests/unit/test_decode.c src/runtime/decode.c
TEST_DECODE_OBJS := $(TEST_DECODE_SRCS:.c=.o)
$(TEST_DECODE_BIN): $(TEST_DECODE_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_DECODE_OBJS) -lm

test-decode: $(TEST_DECODE_BIN)
	./$(TEST_DECODE_BIN)

TEST_REFINER_BIN := build/test_refiner
TEST_REFINER_SRCS := tests/unit/test_refiner.c src/runtime/refiner.c src/io/json.c
TEST_REFINER_OBJS := $(TEST_REFINER_SRCS:.c=.o)
$(TEST_REFINER_BIN): $(TEST_REFINER_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_REFINER_OBJS) -lm

test-refiner: $(TEST_REFINER_BIN)
	./$(TEST_REFINER_BIN)

TEST_PROGRESS_BIN := build/test_progress
TEST_PROGRESS_SRCS := tests/unit/test_progress.c
TEST_PROGRESS_OBJS := $(TEST_PROGRESS_SRCS:.c=.o)
$(TEST_PROGRESS_BIN): $(TEST_PROGRESS_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_PROGRESS_OBJS) -lm

test-progress: $(TEST_PROGRESS_BIN)
	./$(TEST_PROGRESS_BIN)

TEST_PREVIEW_BIN := build/test_preview
TEST_PREVIEW_SRCS := tests/unit/test_preview.c src/runtime/preview.c
TEST_PREVIEW_OBJS := $(TEST_PREVIEW_SRCS:.c=.o)
$(TEST_PREVIEW_BIN): $(TEST_PREVIEW_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_PREVIEW_OBJS) -lm -l:libjpeg.so.8

test-preview: $(TEST_PREVIEW_BIN)
	./$(TEST_PREVIEW_BIN)

TEST_ENGINE_BIN := build/test_engine_preload
TEST_ENGINE_SRCS := tests/unit/test_engine_preload.c src/runtime/engine.c \
                    src/runtime/generate.c src/runtime/request.c \
                    src/runtime/sequence.c src/runtime/ref_alias.c \
                    src/runtime/decode.c src/runtime/torch_rng.c \
                    src/runtime/o1_timing.c src/model/block.c \
                    src/model/forward.c src/model/scheduler.c \
                    src/model/tokenizer.c src/model/lora.c src/model/vision.c \
                    src/image/hd_image.c src/image/layout.c src/io/png_wrap.c \
                    $(CORE_SRCS) src/model/weights.c
TEST_ENGINE_OBJS := $(TEST_ENGINE_SRCS:.c=.o)
$(TEST_ENGINE_BIN): $(TEST_ENGINE_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_ENGINE_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++ -l:libjpeg.so.8

test-engine: $(TEST_ENGINE_BIN)
	./$(TEST_ENGINE_BIN) $(ENGINE_MODEL)

TEST_M17B_BIN := build/test_m1_7_base
TEST_M17B_SRCS := tests/unit/test_m1_7_base.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_M17B_OBJS := $(TEST_M17B_SRCS:.c=.o)

TEST_SANITY_BIN := build/sanity_gen
TEST_SANITY_SRCS := tests/unit/sanity_gen.c src/model/block.c src/model/forward.c \
                    src/model/scheduler.c src/runtime/decode.c src/runtime/torch_rng.c \
                    src/io/png_wrap.c $(CORE_SRCS) src/model/weights.c
TEST_SANITY_OBJS := $(TEST_SANITY_SRCS:.c=.o)
$(TEST_SANITY_BIN): $(TEST_SANITY_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_SANITY_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

test-sanity: $(TEST_SANITY_BIN)
$(TEST_M17B_BIN): $(TEST_M17B_OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $(TEST_M17B_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

test-m17-base: $(TEST_M17B_BIN)
	./$(TEST_M17B_BIN)

TEST_M17BL_BIN := build/test_m1_7_base_local
TEST_M17BL_SRCS := tests/unit/test_m1_7_base_local.c src/model/block.c src/model/forward.c $(CORE_SRCS) src/model/weights.c
TEST_M17BL_OBJS := $(TEST_M17BL_SRCS:.c=.o)
$(TEST_M17BL_BIN): $(TEST_M17BL_OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) -o $@ $(TEST_M17BL_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

test-m17-base-local: $(TEST_M17BL_BIN)
	./$(TEST_M17BL_BIN)

test-tokenizer: $(TEST_TOK_BIN)
	./$(TEST_TOK_BIN)

$(TEST_BIN): $(TEST_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_OBJS) -lm

$(TEST_W_BIN): $(TEST_W_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_W_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_P_BIN): $(TEST_P_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_P_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_TOK_BIN): $(TEST_TOK_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_TOK_OBJS) -lm

$(TEST_BLOCK_BIN): $(TEST_BLOCK_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_BLOCK_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_SDPA_BIN): $(TEST_SDPA_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_SDPA_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

test-sdpa-block: $(TEST_SDPA_BIN)
	LD_LIBRARY_PATH="$(CUDNN_HOME)/lib:$$LD_LIBRARY_PATH" ./$(TEST_SDPA_BIN)

$(TEST_SDPA_FWD_BIN): $(TEST_SDPA_FWD_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_SDPA_FWD_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

test-sdpa-forward: $(TEST_SDPA_FWD_BIN)
	LD_LIBRARY_PATH="$(CUDNN_HOME)/lib:$$LD_LIBRARY_PATH" ./$(TEST_SDPA_FWD_BIN)

$(BENCH_BLOCK_BIN): $(BENCH_BLOCK_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(BENCH_BLOCK_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

bench-block: $(BENCH_BLOCK_BIN)
	LD_LIBRARY_PATH="$(CUDNN_HOME)/lib:$$LD_LIBRARY_PATH" ./$(BENCH_BLOCK_BIN)

$(BENCH_SDPA2_BIN): $(BENCH_SDPA2_OBJS) $(CUDA_OBJS) src/runtime/sequence.o src/model/tokenizer.o $(CORE_SRCS:.c=.o)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(BENCH_SDPA2_OBJS) src/runtime/sequence.o src/model/tokenizer.o $(CORE_SRCS:.c=.o) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

bench-sdpa-twopass: $(BENCH_SDPA2_BIN)
	LD_LIBRARY_PATH="$(CUDNN_HOME)/lib:$$LD_LIBRARY_PATH" ./$(BENCH_SDPA2_BIN)

$(BENCH_GEMM_BIN): $(BENCH_GEMM_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(BENCH_GEMM_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

bench-gemm: $(BENCH_GEMM_BIN)
	LD_LIBRARY_PATH="$(CUDNN_HOME)/lib:$$LD_LIBRARY_PATH" ./$(BENCH_GEMM_BIN)

$(TEST_FF_BIN): $(TEST_FF_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_FF_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_VISION_BIN): $(TEST_VISION_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_VISION_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

test-vision: $(TEST_VISION_BIN)
	./$(TEST_VISION_BIN)

$(TEST_GEMM_BIN): $(TEST_GEMM_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_GEMM_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_GGUF_BIN): $(TEST_GGUF_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_GGUF_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_REPLAY_BIN): $(TEST_REPLAY_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_REPLAY_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_FH_BIN): $(TEST_FH_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_FH_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_TAIL_BIN): $(TEST_TAIL_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_TAIL_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(TEST_M15_BIN): $(TEST_M15_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_M15_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

$(BIN): $(OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++ -l:libjpeg.so.8

# M2 pre-baseline: same engine with compile-time timing instrumentation.
# Production builds (default) contain zero timing code; only this target
# compiles with -DO1_DEBUG_TIMING. Timing objects live in a separate dir so
# they never collide with the production objects.
TIMING_BIN := build/hidream_timing
TIMING_OBJDIR := build/obj/timing
TIMING_OBJS := $(patsubst %.c,$(TIMING_OBJDIR)/%.o,$(SRCS))
$(TIMING_OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DO1_DEBUG_TIMING $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<
$(TIMING_BIN): $(TIMING_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DO1_DEBUG_TIMING -o $@ $(TIMING_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++ -l:libjpeg.so.8

timing: $(TIMING_BIN)

# M2 block-level timing: same as `timing` plus per-block sub-stage CUDA
# events (-DO1_DEBUG_BLOCK_TIMING). Used for the B1-B4 microbenchmarks.
TIMING_BLOCK_BIN := build/hidream_timing_block
TIMING_BLOCK_OBJDIR := build/obj/timing_block
TIMING_BLOCK_OBJS := $(patsubst %.c,$(TIMING_BLOCK_OBJDIR)/%.o,$(SRCS))
$(TIMING_BLOCK_OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DO1_DEBUG_TIMING -DO1_DEBUG_BLOCK_TIMING $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<
$(TIMING_BLOCK_BIN): $(TIMING_BLOCK_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -DO1_DEBUG_TIMING -DO1_DEBUG_BLOCK_TIMING -o $@ $(TIMING_BLOCK_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++ -l:libjpeg.so.8

timing-block: $(TIMING_BLOCK_BIN)

# OpenAI-compatible Images API server. It links the same runtime/CUDA objects
# as the CLI: no inference logic is duplicated, only the HTTP layer is added.
SERVER_SRCS := src/server/o1_server.c $(filter-out src/main.c,$(SRCS))
SERVER_OBJS := $(SERVER_SRCS:.c=.o)
$(SERVER_BIN): $(SERVER_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(SERVER_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++ -l:libjpeg.so.8 -lpthread

server: $(SERVER_BIN)

# Model-free server unit tests: the server TU is included directly, so only
# the runtime objects it references are needed (no CUDA, no weights).
TEST_SERVER_BIN := build/test_server
TEST_SERVER_SRCS := tests/unit/test_server.c src/runtime/request.c \
                    src/runtime/sequence.c src/model/tokenizer.c \
                    src/image/layout.c src/image/hd_image.c \
                    src/io/png_wrap.c $(CORE_SRCS)
TEST_SERVER_OBJS := $(TEST_SERVER_SRCS:.c=.o)
# The server TU is included whole, so its socket/worker entry points are
# legitimately unused in this test binary.
tests/unit/test_server.o: tests/unit/test_server.c src/server/o1_server.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -Wno-unused-function $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<
$(TEST_SERVER_BIN): $(TEST_SERVER_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_SERVER_OBJS) -lm -l:libjpeg.so.8 -lpthread

test-server: $(TEST_SERVER_BIN)
	./$(TEST_SERVER_BIN)



%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<
$(CUBIN)/%.o: src/cuda/%.cu
	@mkdir -p $(CUBIN)
	$(NVCC) -arch=sm_121 -O2 -std=c++17 $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<
# LoRA GPU merge lives under src/model/ (not src/cuda/).
$(CUBIN)/lora_merge.o: src/model/lora_merge.cu
	@mkdir -p $(CUBIN)
	$(NVCC) -arch=sm_121 -O2 -std=c++17 $(CPPFLAGS) $(CUDA_CPPFLAGS) -c -o $@ $<
# cuDNN SDPA wrapper: the vendored C++ Frontend is header-only and heavy to
# compile, so this object is built with -O0 (the wrapper is a thin shim; the
# hot path is inside libcudnn.so). Requires the CUDA driver header (cuda.h)
# to precede cudnn_frontend.h — the project's own header is hd_cuda.h, so
# <cuda.h> always resolves to the NVIDIA driver API.
$(CUBIN)/hd_cudnn_sdpa.o: src/cuda/hd_cudnn_sdpa.cu
	@mkdir -p $(CUBIN)
	$(NVCC) -arch=sm_121 -O0 -std=c++17 $(CPPFLAGS) $(CUDA_CPPFLAGS) $(CUDNN_CPPFLAGS) -c -o $@ $<
clean:
	rm -rf build
	find src tests -name '*.o' -delete

.PHONY: all test test-primitives test-block test-full-forward test-tokenizer test-m15 test-m17 test-m17-base test-rng test-png test-image test-layout test-seq test-seq-ref test-seq-diag test-seq-profiles test-decode test-refiner test-progress test-preview test-sanity test-server test-engine server clean timing timing-block bench-gemm bench-sdpa-twopass
# Base FlowUniPC CUDA kernel validation (M-post base default path).
TEST_BUNIPC_BIN := build/test_base_unipc_cuda
TEST_BUNIPC_SRCS := tests/unit/test_base_unipc_cuda.c src/model/scheduler.c $(CORE_SRCS)
TEST_BUNIPC_OBJS := $(TEST_BUNIPC_SRCS:.c=.o)
test-base-unipc: $(TEST_BUNIPC_BIN)
	./$(TEST_BUNIPC_BIN)
$(TEST_BUNIPC_BIN): $(TEST_BUNIPC_OBJS) $(CUDA_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $(TEST_BUNIPC_OBJS) $(CUDA_OBJS) $(CUDA_LDFLAGS) $(CUBLAS_LDFLAGS) $(CUDNN_LDFLAGS) -lm -lstdc++

# C4 two-pass SDPA A/B harness (CUDA TU: needs the driver header and the
# vendored cuDNN frontend include path).
tests/unit/bench_sdpa_twopass.o: tests/unit/bench_sdpa_twopass.cu
	@mkdir -p $(dir $@)
	$(NVCC) -arch=sm_121 -O2 -std=c++17 $(CPPFLAGS) $(CUDA_CPPFLAGS) $(CUDNN_CPPFLAGS) -c -o $@ $<
