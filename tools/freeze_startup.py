"""M0.3 startup-only oracle freezer.

Loads the processor and model (V1: startup/load only), and dumps:
  - environment identity
  - oracle identity
  - model identity
  - state-dict tensor manifest (name/shape/dtype/numel)
  - canonical prompt tokenization
  - scheduler structural config (timesteps/sigmas), WITHOUT denoising

NO transformer forward, NO denoising, NO image generation is performed.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

CANONICAL_PROMPT = "a red fox sits under a cherry blossom tree"


def _sh(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"


def _git_oracle(root: str) -> dict:
    py = os.path.join(root, "python")
    return {
        "remote": _sh(["git", "-C", py, "remote", "get-url", "origin"]),
        "sha": _sh(["git", "-C", py, "rev-parse", "HEAD"]),
        "branch": _sh(["git", "-C", py, "branch", "--show-current"]),
        "status_short": _sh(["git", "-C", py, "status", "--short"]),
    }


def _env_identity() -> dict:
    import torch
    import transformers
    import safetensors
    import huggingface_hub

    cuda_ver = "n/a"
    gpu = {}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        gpu = {
            "name": p.name,
            "compute_capability": [p.major, p.minor],
            "total_memory_bytes": p.total_memory,
        }
        cuda_ver = torch.version.cuda

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "architecture": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "kernel": platform.version(),
        "python": platform.python_version(),
        "cuda_torch_compiled": cuda_ver,
        "pytorch": torch.__version__,
        "transformers": transformers.__version__,
        "safetensors": safetensors.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "gpu": gpu,
    }


def _tensor_fingerprint(model) -> str:
    """Deterministic fingerprint over (name, shape, dtype, numel) only."""
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode())
        h.update(b"\x00")
        h.update(json.dumps(list(t.shape)).encode())
        h.update(b"\x00")
        h.update(str(t.dtype).encode())
        h.update(b"\x00")
        h.update(str(int(t.numel())).encode())
        h.update(b"\x00")
    return h.hexdigest()


def _tensor_manifest(model) -> list[dict]:
    manifest = []
    sd = model.state_dict()
    for name, t in sd.items():
        manifest.append({
            "name": name,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": int(t.numel()),
        })
    return manifest


def _tokenizer_freeze(processor, model_config) -> dict:
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Attach special-token shortcuts exactly as the oracle does (metadata only).
    for attr, tok in [
        ("boi_token", "<|boi_token|>"),
        ("bor_token", "<|bor_token|>"),
        ("eor_token", "<|eor_token|>"),
        ("bot_token", "<|bot_token|>"),
        ("tms_token", "<|tms_token|>"),
    ]:
        setattr(tokenizer, attr, tok)

    messages = [{"role": "user", "content": CANONICAL_PROMPT}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(template, add_special_tokens=False)

    special_tokens = {
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "image_token_id": getattr(model_config, "image_token_id", None),
        "video_token_id": getattr(model_config, "video_token_id", None),
        "vision_start_token_id": getattr(model_config, "vision_start_token_id", None),
        "vision_end_token_id": getattr(model_config, "vision_end_token_id", None),
        "boi": "<|boi_token|>",
        "bor": "<|bor_token|>",
        "eor": "<|eor_token|>",
        "bot": "<|bot_token|>",
        "tms": "<|tms_token|>",
    }

    return {
        "prompt": CANONICAL_PROMPT,
        "template": template,
        "input_ids": input_ids,
        "n_tokens": len(input_ids),
        "special_tokens": special_tokens,
        "tokenizer_class": type(tokenizer).__name__,
        "processor_class": type(processor).__name__,
    }


def _scheduler_freeze(model_config, profile_cfg: dict, dtype: str = "float32") -> dict:
    import torch  # noqa: F401
    from models.pipeline import DEFAULT_TIMESTEPS, build_scheduler

    # Scheduler selection mirrors python/inference.py, keyed off the profile's
    # model_type ("full" -> Base 50-step FlowUniPC default; "dev" -> Dev T2I
    # 28-step flash). Structural freeze only — no denoising step is run.
    profile = profile_cfg.get("profile")
    model_type = profile_cfg.get("model_type", "dev")
    if model_type == "full":
        num_steps = int(profile_cfg.get("num_inference_steps", 50))
        shift = 3.0
        scheduler_name = "default"
        timesteps_list = None
    else:
        # Dev T2I (non-editing) uses the "flash" stochastic flow-match
        # scheduler. Editing (single ref image) uses "flow_match". Structural
        # timesteps/sigmas are identical between "flash" and "flow_match" for
        # Dev (same DEFAULT_TIMESTEPS, shift=1.0); only step() semantics
        # differ. Defaults below match python/inference.py for model_type dev.
        num_steps = int(profile_cfg.get("num_inference_steps", 28))
        shift = float(profile_cfg.get("scheduler_shift", 1.0))
        scheduler_name = profile_cfg.get("scheduler_name", "flash")
        timesteps_list = list(DEFAULT_TIMESTEPS)
    device = "cpu"
    sched = build_scheduler(num_steps, timesteps_list, shift, device, scheduler_name)

    return {
        "profile": profile,
        "scheduler_name": scheduler_name,
        "num_inference_steps": num_steps,
        "shift": shift,
        "timesteps": [int(t) for t in sched.timesteps.tolist()],
        "sigmas": [float(s) for s in sched.sigmas.tolist()],
        "patch_size": 32,
        "noise_scale_start": 8.0,
        "noise_scale_end": 8.0,
        "noise_clip_std": 8.0,
        "t_eps": 0.001,
        "condition_image_size": 384,
    }


def freeze_startup(root: str, profile: str, local_path: str, profile_cfg: dict) -> str:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("FA_VERSION", "0")

    import torch
    from transformers import AutoProcessor

    # Oracle module path (imported, never patched).
    sys.path.insert(0, os.path.join(root, "python"))
    from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration  # noqa: E402

    t0 = time.time()

    out = {
        "profile": profile,
        "environment": _env_identity(),
        "oracle": _git_oracle(root),
    }

    # Processor (tokenizer) only.
    processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)

    # Model load — startup only, no forward.
    dtype = profile_cfg.get("dtype", "float32")
    torch_dtype = getattr(torch, dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_path, torch_dtype=torch_dtype, device_map="cuda", local_files_only=True
    ).eval()

    model_config = model.config

    out["model"] = {
        "profile": profile,
        "hf_repo": profile_cfg["hf_repo"],
        "hf_revision": profile_cfg["immutable_revision"],
        "local_path": local_path,
        "dtype": dtype,
        "model_type_hf": getattr(model_config, "model_type", None),
        "architectures": getattr(model_config, "architectures", None),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "total_size_bytes": int(
            sum(t.numel() * t.element_size() for t in model.parameters())
        ),
        "device": str(next(model.parameters()).device),
    }

    out["tensor_manifest"] = _tensor_manifest(model)
    out["tensor_fingerprint"] = _tensor_fingerprint(model)
    out["model_config"] = model_config.to_dict()
    out["tokenizer"] = _tokenizer_freeze(processor, model_config)
    out["scheduler"] = _scheduler_freeze(model_config, profile_cfg)
    out["load_seconds"] = round(time.time() - t0, 2)

    out_dir = Path(root) / "artifacts" / "m0" / "startup"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"freeze_{profile}.json"
    out_path.write_text(json.dumps(out, indent=2))

    # Compact canonical manifest (versioned reproducibility contract).
    compact = {
        "profile": profile,
        "oracle_sha": out["oracle"]["sha"],
        "oracle_clean": out["oracle"]["status_short"] == "",
        "hf_repo": out["model"]["hf_repo"],
        "hf_revision": out["model"]["hf_revision"],
        "model_type_hf": out["model"]["model_type_hf"],
        "dtype": out["model"]["dtype"],
        "parameter_count": out["model"]["parameter_count"],
        "python": out["environment"]["python"],
        "pytorch": out["environment"]["pytorch"],
        "transformers": out["environment"]["transformers"],
        "gpu": out["environment"]["gpu"],
        "n_tensors": len(out["tensor_manifest"]),
        "tensor_fingerprint": out["tensor_fingerprint"],
        "model_config": out["model_config"],
        "canonical_prompt": out["tokenizer"]["prompt"],
        "canonical_input_ids": out["tokenizer"]["input_ids"],
        "scheduler": out["scheduler"],
    }
    manifest_path = Path(root) / "config" / f"startup_manifest_{profile}.json"
    manifest_path.write_text(json.dumps(compact, indent=2))

    print(json.dumps({
        "profile": profile,
        "parameter_count": out["model"]["parameter_count"],
        "n_tensors": len(out["tensor_manifest"]),
        "load_seconds": out["load_seconds"],
        "device": out["model"]["device"],
        "gpu": out["environment"]["gpu"],
        "output": str(out_path),
        "manifest": str(manifest_path),
    }, indent=2))
    return str(out_path)


if __name__ == "__main__":
    root, profile = sys.argv[1], sys.argv[2]
    cfg = json.load(open(os.path.join(root, "config", profile + ".json")))
    freeze_startup(root, profile, os.path.join(root, cfg["local_path"]), cfg)
