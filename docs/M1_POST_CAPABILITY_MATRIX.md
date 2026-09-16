# M1-post Capability Matrix

Frozen at M1-post.0 from the official upstream audit (see `artifacts/m1post/audit/A–G`).
Source of truth: frozen oracle at `/python` (commit `3237a638…c2dfbf`).

## Capability × profile

| Capability | Full / Base | Dev | Dev-2604 | Notes |
|------------|-------------|-----|----------|-------|
| Text-to-image | ✅ | ✅ | ✅ (primary) | T2I via `build_t2i_text_sample` |
| Long-text rendering | ✅ | ✅ | ✅ | tokenizer-level, no truncation |
| Multilingual text | ✅ | ✅ | ✅ | Unicode exact at tokenizer |
| Instruction editing | ✅ | ✅ | ✅ | single-ref edit |
| Single-ref aspect-preserving edit | ✅ | ✅ | ✅ | `keep_original_aspect`, max_size 2048 |
| Multi-ref personalization | ✅ | ✅ | ✅ | >= 2 refs, ordered concat |
| Layout conditioning | ✅ | ✅ | ✅ | bbox canvas appended as image |
| Skeleton / pose conditioning | ⚠️ | ⚠️ | ⚠️ | **NOT in audited source** — see blockers |
| Storyboard | ⚠️ | ⚠️ | ⚠️ | **NOT in audited source** — see blockers |
| High-res (2048) | ✅ | ✅ | ✅ | PREDEFINED_RESOLUTIONS incl. 2048² |
| Aspect ratios | ✅ | ✅ | ✅ | `find_closest_resolution` snapping |
| CFG / guidance | ✅ (5.0) | ✅ (0.0) | ✅ | blank " " uncond prompt |
| Distilled sampling | — | ✅ (28 steps, flash) | ✅ | DEFAULT_TIMESTEPS |
| Editing scheduler | — | flow_match default / flash | — | Dev only |
| Noise controls | ✅ | ✅ | ✅ | noise_scale_start/end, noise_clip_std |
| Deterministic seed | ✅ | ✅ | ✅ | seed+1 CPU initial noise |
| Prompt refinement | ✅ | ✅ | ✅ | prompt_agent (legacy) / v2 (Dev-2604) |
| Progress/preview | ✅ | ✅ | ✅ | callback per step; previews 1/4,1/2,3/4 |
| Native image I/O | ✅ | ✅ | ✅ | PIL decode, RGB, no EXIF transpose |
| Output decode | ✅ | ✅ | ✅ | unpatchify, NO VAE |

## Profile recipes (frozen)

| Profile | steps | guidance | shift | scheduler | timesteps |
|---------|-------|----------|-------|-----------|-----------|
| Full | 50 | 5.0 | 3.0 | default (FlowUniPC) | scheduler-derived |
| Dev | 28 | 0.0 | 1.0 | flash | DEFAULT_TIMESTEPS [999…8] |
| Dev edit | 28 | 0.0 | 1.0 | flow_match (default) / flash | DEFAULT_TIMESTEPS |

## Blockers / unknown semantics (must not be assumed)

1. **Skeleton conditioning**: no OpenPose/keypoint parser exists in the audited Python sources. The advertised "skeleton conditioning" is not executable from the frozen oracle → mark UNSUPPORTED unless upstream source elsewhere defines it.
2. **Storyboard**: no multi-panel decomposition exists in the audited sources. Generation accepts one prompt + optional refs → mark UNSUPPORTED unless upstream defines it.
3. **Per-step noise RNG**: per-step noise uses diffusers `randn_tensor` with no explicit generator → CUDA default RNG (Philox). Native stand-in: CPU MT19937 port (documented, deterministic).
4. **Reference role tags**: no subject/style/type role labels; only ordered refs + optional bbox text.