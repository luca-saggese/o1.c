"""L1: canonical map vs tensor manifest consistency test.

Verifies that every entry in config/hidream_lora_map_dev.json:
  - has a base_tensor that exists in the frozen tensor manifest
  - has consistent shapes (out_dim/in_dim match the manifest)
  - uses the pinned lora_unet_ naming contract
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from common import lora_key_for_module  # noqa: E402


def load_manifest():
    path = os.path.join(ROOT, "config", "tensor_manifest_dev.json")
    with open(path) as f:
        return json.load(f)


def load_map():
    path = os.path.join(ROOT, "config", "hidream_lora_map_dev.json")
    with open(path) as f:
        return json.load(f)


def main():
    manifest = load_manifest()
    tensors = {t["name"]: t for t in manifest["tensors"]}
    lora_map = load_map()

    assert lora_map["profile"] == "dev", "map profile must be dev"
    entries = lora_map["entries"]
    assert len(entries) == 257, f"expected 257 linear targets, got {len(entries)}"

    for e in entries:
        base = e["base_tensor"]
        assert base in tensors, f"base tensor missing from manifest: {base}"
        t = tensors[base]
        # manifest shape is [out, in] for linear weights
        assert t["shape"][0] == e["shape"][0], f"{base}: out_dim mismatch"
        assert t["shape"][1] == e["shape"][1], f"{base}: in_dim mismatch"
        # naming contract: lora_prefix == lora_unet_<module with _>
        module = base[: -len(".weight")]
        assert e["lora_prefix"] == lora_key_for_module(module), \
            f"{base}: lora_prefix does not follow pinned naming"

    print(f"test_name_mapping PASS ({len(entries)} targets)")


if __name__ == "__main__":
    main()