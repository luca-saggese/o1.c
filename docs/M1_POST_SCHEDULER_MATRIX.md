# M1-post Scheduler Matrix

Frozen at M1-post.0 from the official upstream audit.
Source: `python/models/flash_scheduler.py`, `pipeline.py`, `inference.py`.

## Scheduler selection (pipeline.py:80-97)

| Name | Class | Used by |
|------|-------|---------|
| `default` | FlowUniPCMultistepScheduler | Full (50 steps) |
| `flash` | flash scheduler | Dev (28 steps) |
| `flow_match` | flow-match scheduler | Dev editing (default) |

## Recipes

| Profile | steps | guidance | shift | scheduler | timesteps |
|---------|-------|----------|-------|-----------|-----------|
| Full | 50 | 5.0 | 3.0 | default | scheduler-derived |
| Dev | 28 | 0.0 | 1.0 | flash | DEFAULT_TIMESTEPS |
| Dev edit | 28 | 0.0 | 1.0 | flow_match / flash | DEFAULT_TIMESTEPS |

## Timestep / sigma semantics

- `DEFAULT_TIMESTEPS = [999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764, 737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8]` (pipeline.py:26-28).
- Explicit `timesteps_list` → sigmas = t/1000, plus terminal 0.0.
- Flash schedule: linspace sigma_max→sigma_min over steps (flash_scheduler.py:197-245).
- `use_dynamic_shifting=False`: sigma' = `shift*sigma / (1 + (shift-1)*sigma)`.
- Transformer timestep = `1 - step_t/1000` (model_timestep); embedder input = model_timestep * 1000.

## Noise

- Initial noise: `noise_scale_start * torch.randn((1,3,H,W), generator=torch.Generator('cpu').manual_seed(seed+1))` (pipeline.py:313-317). NOISE_SCALE = 8.0.
- Per-step flash noise: `hack_randn_tensor` → diffusers `randn_tensor` (flash_scheduler.py:34-42, 337-351). No explicit generator → CUDA default (Philox). Native stand-in: CPU MT19937 port (documented).
- Noise scaled by `s_noise`; optional clip `noise_clip_std * noise.std()` (flash_scheduler.py:337-351).

## CFG

- `v_uncond + guidance_scale * (v_cond - v_uncond)` (pipeline.py:376-416).
- Unconditional text = single-space prompt `" "`.

## Resolution

- `PREDEFINED_RESOLUTIONS` (utils.py:11-24): 2048², 2304×1728, 1728×2304, 2560×1440, 1440×2560, 2496×1664, 1664×2496, 3104×1312, 1312×3104, 2304×1792, 1792×2304.
- `find_closest_resolution(w,h)` picks the bucket with closest aspect ratio (utils.py:190-200).
- PATCH_SIZE = 32; `keep_original_aspect` (1 ref) → resize ref to max_size 2048, derive dims.