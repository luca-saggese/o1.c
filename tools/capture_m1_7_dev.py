#!/usr/bin/env python3
"""M1.7 full Dev generation golden capture (V5, 28 steps).

Captures the COMPLETE 28-step denoising trajectory of the frozen HiDream Dev
pipeline using the frozen Python oracle (1 fresh full-model oracle load, 1
oracle run -- the single allowed V5 consumption for the M1.7 gate). The
forward path is exactly the production `_forward_generation` (called with
`use_flash_attn=False` to match the eager 4D attention mask used by the
native path), and the scheduler is the manifest flash Euler scheduler.

This is the closure of the M1 chain: every denoising step's z_next is
captured so the native test can replay the full generation with frozen
per-step post-clamp noise (docs/M1_NUMERICAL_CONTRACT.md §7) and verify the
final decoded 64x64 image.

Per-step captures (28 steps):
  scheduler_timestep      [1]    f32   step_t from manifest schedule
  sigma                   [1]    f32   clamp_min(t_eps) of step_t/1000
  model_timestep          [1]    f32   t_pixeldit = 1 - step_t/1000
  timestep_embedder_input [1]    f32   t_pixeldit * 1000
  raw_sinusoidal          [1,256] f32  timestep_embedding(embedder_input, 256)
  projected_t_emb         [1,4096] bf16 t_embedder1(model_timestep)
  x_pred_full             [23,3072] bf16 full model output (text+img rows)
  x_pred_masked           [4,3072]  bf16 image rows only (used by scheduler)
  v_cond                  [4,3072]  f32  (x_pred_masked - z)/sigma
  model_output            [4,3072]  f32  -v_guided (guidance_scale=1.0)
  z_prev                  [4,3072]  bf16 scheduler input for this step
  noise_pre_clamp         [4,3072]  f32  raw randn tensor (CUDA global RNG)
  noise_std               [1]    f32   noise.std().item()
  clip_val                [1]    f32   noise_clip_std * noise_std
  noise_post_clamp        [4,3072]  f32  noise.clamp(-clip_val, +clip_val)
  z_next                  [4,3072]  bf16 scheduler output (next step input)

Final payloads (image decode, oracle formula):
  final_z                 [4,3072]  bf16 step-27 z_next (input to decode)
  final_image             [64,64,3] uint8 (z+1)/2 rearranged -> 64x64 RGB

Usage:
    .venv/bin/python tools/capture_m1_7_dev.py
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_ROOT = ROOT / "artifacts" / "m1" / "golden" / "M1_7_DEV_FULLGEN"
ORACLE_SHA = "3237a638a5c2c7be106b0175958f4c0db8c2dfbf"
DEV_REVISION = "b6acc2fe452b3120430620dc4354fa442ee081ea"
PROMPT = "a red fox sits under a cherry blossom tree"
SEED = 42
RES = 64            # FAST_VALIDATION_RES
PATCH_SIZE = 32
TMS_TOKEN_ID = 151673
NOISE_SCALE = 8.0
NOISE_CLIP_STD = 8.0
T_EPS = 0.001
NUM_STEPS = 28      # full 28-step trajectory (999, 987, ..., 8)
MANIFEST = ROOT / "config" / "startup_manifest_dev.json"
FIXTURE_ID = "M1_7_DEV_FULLGEN"


def run_guard():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "m1_guard.py"), "check-env"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        print("m1_guard check-env FAILED:\n", r.stdout, r.stderr)
        sys.exit(1)
    print("m1_guard check-env OK")


def bf16_bytes(t):
    t = t.to(torch.bfloat16).cpu().contiguous()
    return t.view(torch.int16).numpy().astype("<u2").tobytes()


def f32_bytes(t):
    return t.float().cpu().contiguous().numpy().tobytes()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    run_guard()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("FA_VERSION", "0")
    os.environ.setdefault("USE_BF16_ROPE", "0")
    sys.path.insert(0, str(ROOT / "python"))

    import torch
    from transformers import AutoProcessor
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration, \
        TimestepEmbedder
    from models.pipeline import build_t2i_text_sample, build_scheduler

    with open(MANIFEST) as f:
        manifest_cfg = json.load(f)
    sched_cfg = manifest_cfg["scheduler"]
    timesteps_list = sched_cfg["timesteps"][:NUM_STEPS]
    assert len(timesteps_list) == NUM_STEPS, (len(timesteps_list), NUM_STEPS)
    assert timesteps_list[0] == 999, timesteps_list
    assert timesteps_list[-1] == 8, timesteps_list

    GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)

    local_path = str(ROOT / "models" / "dev")
    print(f"Loading frozen Dev model (fixture {FIXTURE_ID}, {NUM_STEPS} steps)...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_path, torch_dtype=torch.bfloat16, device_map="cuda",
        local_files_only=True, _attn_implementation="eager",
    ).eval()
    print(f"  model loaded in {time.time()-t0:.1f}s")

    device = model.device
    dtype = torch.bfloat16
    cfg = model.config
    text_cfg = cfg.text_config
    H = text_cfg.hidden_size            # 4096
    NLAYERS = text_cfg.num_hidden_layers  # 36
    OUT_DIM = PATCH_SIZE * PATCH_SIZE * 3  # 3072
    sched = build_scheduler(
        sched_cfg["num_inference_steps"], sched_cfg["timesteps"],
        sched_cfg["shift"], device, scheduler_name=sched_cfg["scheduler_name"])
    s_noise_schedule = sched_cfg.get("noise_scale_start", NOISE_SCALE)
    s_noise_end = sched_cfg.get("noise_scale_end", s_noise_schedule)
    n_steps = len(sched.timesteps)
    noise_scale_schedule = [
        s_noise_schedule + (s_noise_end - s_noise_schedule) * i / (n_steps - 1)
        for i in range(n_steps)
    ]
    noise_clip_std = float(sched_cfg.get("noise_clip_std", NOISE_CLIP_STD))

    sample = build_t2i_text_sample(
        PROMPT, RES, RES, processor.tokenizer, processor, cfg)
    input_ids = sample["input_ids"].to(device)              # [1,19]
    position_ids = sample["position_ids"].to(device)        # [3,1,23]
    token_types = sample["token_types"].to(device)          # [1,23]
    vinput_mask = sample["vinput_mask"].to(device)          # [1,23]
    TS = input_ids.shape[-1]                                # 19 (text seq)
    S = position_ids.shape[-1]                              # 23 (total seq)
    image_tokens = S - TS                                    # 4

    # deterministic explicit noise -> vinputs [1, img_tokens, 3072] bf16
    torch.manual_seed(SEED + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED + 1)
    noise = NOISE_SCALE * torch.randn(
        (1, 3, RES, RES),
        generator=torch.Generator("cpu").manual_seed(SEED + 1),
    ).to(device, dtype)
    # fast-validation reduction of the pipeline patchify at RES=64:
    # pixel_unshuffle(noise, PATCH_SIZE) -> [1, 3072, 2, 2]
    z = torch.nn.functional.pixel_unshuffle(noise, PATCH_SIZE)
    z = z.flatten(2).transpose(1, 2)                         # [1, 4, 3072]

    # ---- noise interception (freeze CUDA-global-RNG noise per step) ----
    import models.flash_scheduler as fs
    captured = {}
    fs_orig_hack = fs.hack_randn_tensor

    def noise_hook(*args, **kwargs):
        raw = fs_orig_hack(*args, **kwargs)
        captured["noise_pre_clamp"] = raw.detach()
        return raw

    fs.hack_randn_tensor = noise_hook

    def run_step(step_idx, step_t, z_in):
        """One denoising step faithful to pipeline.py generate_image."""
        step_t_f32 = step_t.to(dtype=torch.float32)
        t_pixeldit = 1.0 - step_t_f32 / 1000.0              # model domain
        sigma = (step_t_f32 / 1000.0).to(dtype=torch.float32).clamp_min(T_EPS)
        embedder_input = t_pixeldit * 1000.0

        # raw sinusoidal + projected timestep embedding (exact oracle txns)
        raw_sin = TimestepEmbedder.timestep_embedding(
            embedder_input.reshape(-1).to(torch.float32), 256)
        t_emb = model.model.t_embedder1(t_pixeldit.reshape(-1).to(device))

        with torch.no_grad():
            with torch.autocast(device.type, dtype=dtype, cache_enabled=False):
                outputs = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    vinputs=z_in,
                    timestep=t_pixeldit.reshape(-1).to(device),
                    token_types=token_types,
                    use_flash_attn=False,
                )
        x_pred_full = outputs.x_pred                                # [1,23,3072]
        x_pred_masked = x_pred_full[0, vinput_mask[0]].unsqueeze(0)  # [1,4,3072]

        z_f32 = z_in.to(dtype=torch.float32)
        v_cond = (x_pred_masked.to(dtype=torch.float32) - z_f32) / sigma
        v_guided = v_cond                      # guidance_scale=1.0, 1 sample
        model_output = -v_guided               # [1,4,3072] f32

        with torch.no_grad():
            z_next = sched.step(
                model_output.float(), step_t.to(dtype=torch.float32),
                z_f32, s_noise=noise_scale_schedule[step_idx],
                noise_clip_std=noise_clip_std,
                return_dict=False)[0]

        # post-clamp noise: recompute exactly as sched.step did (same device
        # fp32 std -> bit-identical clip bounds)
        raw = captured["noise_pre_clamp"]
        ns = raw.std().item()
        cv = noise_clip_std * ns
        post = raw.clamp(min=-cv, max=cv)

        return {
            "scheduler_timestep": step_t_f32.reshape(-1),
            "sigma": sigma.reshape(-1),
            "model_timestep": t_pixeldit.reshape(-1),
            "timestep_embedder_input": embedder_input.reshape(-1),
            "raw_sinusoidal": raw_sin,
            "projected_t_emb": t_emb,
            "x_pred_full": x_pred_full[0].detach(),
            "x_pred_masked": x_pred_masked[0].detach(),
            "v_cond": v_cond[0].detach(),
            "model_output": model_output[0].detach(),
            "z_prev": z_in[0].detach(),
            "noise_pre_clamp": raw.detach(),
            "noise_std": torch.tensor([ns], dtype=torch.float32),
            "clip_val": torch.tensor([cv], dtype=torch.float32),
            "noise_post_clamp": post.detach(),
            "z_next": z_next[0].detach().to(dtype),
        }

    steps = []
    t_loop = time.time()
    for step_idx, step_t in enumerate(torch.tensor(timesteps_list, device=device)):
        print(f"  step {step_idx}/{NUM_STEPS}: t={step_t.item()}", flush=True)
        if step_idx == 0:
            z_in = z.clone()
        else:
            z_in = steps[-1]["z_next"].unsqueeze(0).to(dtype=dtype)
        rec = run_step(step_idx, step_t, z_in)
        steps.append(rec)
    print(f"  28-step trajectory captured in {time.time()-t_loop:.1f}s")

    # ---- final decode (oracle formula, pipeline.py) ----
    z_final = steps[-1]["z_next"]                               # [4,3072] bf16
    img = (z_final + 1) / 2
    img = img.float().cpu().reshape(1, image_tokens, 3, PATCH_SIZE, PATCH_SIZE)
    # token t -> spatial block (t/2, t%2), channel c stride PATCH_SIZE^2:
    img = img.permute(0, 2, 3, 1, 4)          # B C p1 tok p2
    arr = np.zeros((RES, RES, 3), dtype=np.uint8)
    for tok in range(image_tokens):
        for c in range(3):
            blk = img[0, c, :, tok, :].numpy()   # [32,32]
            arr[tok // 2 * PATCH_SIZE:(tok // 2 + 1) * PATCH_SIZE,
                tok % 2 * PATCH_SIZE:(tok % 2 + 1) * PATCH_SIZE, c] = \
                np.round(np.clip(blk * 255, 0, 255)).astype(np.uint8)

    # ---- write golden (single .bin + per-fixture .json + manifest) ----
    payloads = []
    for si, rec in enumerate(steps):
        pfx = f"step{si:02d}_"
        payloads.extend([
            (f"{pfx}scheduler_timestep", "float32", [1], f32_bytes(rec["scheduler_timestep"]), "in"),
            (f"{pfx}sigma", "float32", [1], f32_bytes(rec["sigma"]), "in"),
            (f"{pfx}model_timestep", "float32", [1], f32_bytes(rec["model_timestep"]), "in"),
            (f"{pfx}timestep_embedder_input", "float32", [1], f32_bytes(rec["timestep_embedder_input"]), "in"),
            (f"{pfx}raw_sinusoidal", "float32", [1, 256], f32_bytes(rec["raw_sinusoidal"]), "ckpt"),
            (f"{pfx}projected_t_emb", "bfloat16", [1, H], bf16_bytes(rec["projected_t_emb"]), "ckpt"),
            (f"{pfx}x_pred_full", "bfloat16", [S, OUT_DIM], bf16_bytes(rec["x_pred_full"]), "ckpt"),
            (f"{pfx}x_pred_masked", "bfloat16", [image_tokens, OUT_DIM], bf16_bytes(rec["x_pred_masked"]), "ckpt"),
            (f"{pfx}v_cond", "float32", [image_tokens, OUT_DIM], f32_bytes(rec["v_cond"]), "ckpt"),
            (f"{pfx}model_output", "float32", [image_tokens, OUT_DIM], f32_bytes(rec["model_output"]), "ckpt"),
            (f"{pfx}z_prev", "bfloat16", [image_tokens, OUT_DIM], bf16_bytes(rec["z_prev"]), "ckpt"),
            (f"{pfx}noise_pre_clamp", "float32", [image_tokens, OUT_DIM], f32_bytes(rec["noise_pre_clamp"]), "ckpt"),
            (f"{pfx}noise_std", "float32", [1], f32_bytes(rec["noise_std"]), "in"),
            (f"{pfx}clip_val", "float32", [1], f32_bytes(rec["clip_val"]), "in"),
            (f"{pfx}noise_post_clamp", "float32", [image_tokens, OUT_DIM], f32_bytes(rec["noise_post_clamp"]), "ckpt"),
            (f"{pfx}z_next", "bfloat16", [image_tokens, OUT_DIM], bf16_bytes(rec["z_next"]), "out"),
        ])
    payloads.append(("final_z", "bfloat16", [image_tokens, OUT_DIM],
                     bf16_bytes(z_final), "out"))
    payloads.append(("final_image", "uint8", [RES, RES, 3],
                     arr.tobytes(), "out"))

    bin_path = GOLDEN_ROOT / f"{FIXTURE_ID}.bin"
    offset = 0
    rows = []
    for name, dt, shape, data, role in payloads:
        rows.append({"name": name, "dtype": dt, "shape": list(shape),
                     "offset": offset, "nbytes": len(data), "role": role})
        offset += len(data)
    bin_path.write_bytes(b"".join(p[3] for p in payloads))
    bin_sha = sha256_file(bin_path)

    step_meta = []
    for si, (step_t_val, rec) in enumerate(zip(timesteps_list, steps)):
        step_meta.append({
            "step_index": si,
            "scheduler_timestep": round(rec["scheduler_timestep"].item(), 6),
            "sigma": round(rec["sigma"].item(), 6),
            "model_timestep": round(rec["model_timestep"].item(), 8),
            "timestep_embedder_input": round(rec["timestep_embedder_input"].item(), 6),
            "s_noise": noise_scale_schedule[si],
            "noise_clip_std": noise_clip_std,
            "noise_std": round(rec["noise_std"].item(), 8),
            "clip_val": round(rec["clip_val"].item(), 6),
        })

    meta = {
        "fixture_id": FIXTURE_ID,
        "schema_version": 1,
        "milestone": "M1.7",
        "validation_level": "V5",
        "oracle_sha": ORACLE_SHA,
        "model_profile": "dev",
        "model_revision": DEV_REVISION,
        "resolution": [RES, RES],
        "batch": 1,
        "compute_dtype": "bfloat16",
        "stored_dtype": "float32_bf16pairs",
        "seed": SEED,
        "prompt": PROMPT,
        "num_steps": NUM_STEPS,
        "timesteps_captured": timesteps_list,
        "noise_scale": NOISE_SCALE,
        "noise_clip_std": noise_clip_std,
        "t_eps": T_EPS,
        "scheduler_name": sched_cfg["scheduler_name"],
        "use_flash_attn": False,
        "forward_path": "the frozen _forward_generation with eager 4D attention mask",
        "timestep_policy": "model domain: t_pixeldit = 1 - step_t/1000; "
                           "embedder receives t_pixeldit (x1000 internal)",
        "steps": step_meta,
        "text_seq_len": TS,
        "image_tokens": image_tokens,
        "seq_len": S,
        "hidden": H,
        "num_layers": NLAYERS,
        "out_dim": OUT_DIM,
        "mrope_section": text_cfg.rope_scaling["mrope_section"],
        "mrope_interleaved": text_cfg.rope_scaling.get("mrope_interleaved", True),
        "rope_theta": text_cfg.rope_theta,
        "rms_eps": text_cfg.rms_norm_eps,
        "tensors": rows,
        "bin": {"file": f"{FIXTURE_ID}.bin", "nbytes": offset, "sha256": bin_sha},
        "pipeline_forward_wall_s": round(float(time.time() - t0), 3),
        "capture_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (GOLDEN_ROOT / f"{FIXTURE_ID}.json").write_text(
        json.dumps(meta, indent=2) + "\n")

    manifest = {"fixture": FIXTURE_ID, "files": {
        f"{FIXTURE_ID}.json": sha256_file(GOLDEN_ROOT / f"{FIXTURE_ID}.json"),
        f"{FIXTURE_ID}.bin": bin_sha,
    }, "tensor_count": len(rows)}
    (GOLDEN_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    # ---- sanity ----
    for si, rec in enumerate(steps):
        for name, tt in rec.items():
            assert torch.isfinite(tt.float()).all(), \
                f"step {si} {name} contains NaN/Inf"
    for si, sm in enumerate(step_meta):
        expect_in = timesteps_list[si] / 1000.0
        assert abs(sm["model_timestep"] - (1.0 - expect_in)) < 1e-3, \
            f"step {si} model_timestep domain wrong: {sm['model_timestep']}"
        assert abs(sm["timestep_embedder_input"] - (1.0 - expect_in) * 1000) < 1.0, \
            f"step {si} embedder input wrong: {sm['timestep_embedder_input']}"

    print("\n=== captured trajectory (frozen, first/last shown) ===")
    for si in (0, 13, 27):
        sm = step_meta[si]
        print(f"  step{si:02d}: scheduler_t={sm['scheduler_timestep']:>7.1f}  "
              f"sigma={sm['sigma']:>7.4f}  model_t={sm['model_timestep']:>9.6f}  "
              f"embed_in={sm['timestep_embedder_input']:>9.4f}  "
              f"noise_std={sm['noise_std']:.6f} clip={sm['clip_val']:.6f}")
    print(f"\nseq_len={S} text={TS} img={image_tokens} steps={NUM_STEPS}")
    print(f"final_image uint8[64,64,3] px range "
          f"[{arr.min()}, {arr.max()}] mean={arr.mean():.2f}")
    print(f"golden -> {GOLDEN_ROOT}  [{offset} B, sha {bin_sha[:16]}...]")
    print("sanity checks OK (all finite, timestep domain correct)")
    return 0


if __name__ == "__main__":
    import torch  # noqa: E402
    sys.exit(main())
