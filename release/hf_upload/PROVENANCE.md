# Provenance

This file records, for every artifact in this repository, exactly which
upstream revision it was converted from and how.

## Summary

| Artifact | Source repository | Source revision (immutable commit SHA) | Upstream license |
|----------|-------------------|----------------------------------------|------------------|
| `hidream-o1-dev-2604-bf16.gguf` | `HiDream-ai/HiDream-O1-Image-Dev-2604` | `b6acc2fe452b3120430620dc4354fa442ee081ea` | MIT |
| `hidream-o1-dev-bf16.gguf`      | `HiDream-ai/HiDream-O1-Image-Dev`      | `c0bada0e15c54a9f96a6d1ecc35575b32bc21544` | MIT |
| `hidream-o1-base-bf16.gguf`     | `HiDream-ai/HiDream-O1-Image`          | `0b0901d99f200389e138c61946af1185f5f49a13` | MIT |

## What the conversion does

The conversion is performed by `tools/hidream_convert.py` from the o1.c
engine repository, driven by the reproducible definitions in
`release/models.def.json` and the script `scripts/build_release_models.sh`.

It performs a **layout-preserving re-serialization**:

* reads the upstream `safetensors` shards at the pinned revision;
* keeps tensor values and dtype unchanged (BF16 in, BF16 out);
* writes them into a GGUF container with 256-byte alignment;
* adds the o1.c metadata keys (`general.architecture`, `general.name`,
  `hidream.profile`, `hidream.variant`, `hidream.revision`,
  `hidream.dtype`, `hidream.quantization`, `hidream.num_layers`,
  `hidream.layout_version`).

No weights are modified, fine-tuned, merged, pruned or quantized.

## Identifying the correct upstream revision

All three upstream repositories publish the **same**
`model.safetensors.index.json` (sha256
`33865f9d84679a85...`), so the index file alone does **not** identify a
checkpoint. The artifacts were pinned by comparing shard LFS hashes and the
repository revision. Example, `model-00001-of-00008.safetensors`:

| Repository | First shard LFS sha256 prefix |
|------------|-------------------------------|
| `HiDream-O1-Image-Dev-2604` | `f34477502c47...` |
| `HiDream-O1-Image-Dev`      | `575a1b54a028...` |
| `HiDream-O1-Image`          | `db5d56d92c14...` |

## Reproducing

From an o1.c checkout:

```bash
git checkout <engine commit recorded in docs/RELEASE_MODELS.md>
./scripts/build_release_models.sh
```

The script downloads the pinned revisions, converts them, and writes
`SHA256SUMS` plus `models.built.json`. Two consecutive runs were verified to
produce byte-identical artifacts.
