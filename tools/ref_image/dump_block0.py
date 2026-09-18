#!/usr/bin/env python3
"""Dump block-0 stage snapshots from the oracle model.

Reads artifacts/ref_image_audit/oracle_visual_stages.pt (produced by
/tmp/oracle_full.py) and writes fp32 .bin files under
artifacts/ref_image_audit/oracle_dump/ for the native C test.

Each fixture records its provenance in a sidecar .txt:
  module path / hook type (PRE/POST) / shape / dtype
"""
import os
import sys
import torch

SRC = "artifacts/ref_image_audit/oracle_visual_stages.pt"
OUT = "artifacts/ref_image_audit/oracle_dump"

# (fixture name, module path, hook type, tensor key in captured)
STAGES = [
    ("block0_input",      "model.visual.blocks.0",            "PRE",  "block0_input"),
    ("block0_norm1",      "model.visual.blocks.0.norm1",      "POST", "norm1_out"),
    ("block0_qkv",        "model.visual.blocks.0.attn.qkv",   "POST", "qkv_out"),
    ("block0_q",          "model.visual.blocks.0.attn.q",     "POST", "q_out"),
    ("block0_k",          "model.visual.blocks.0.attn.k",     "POST", "k_out"),
    ("block0_v",          "model.visual.blocks.0.attn.v",     "POST", "v_out"),
    ("block0_q_rot",      "model.visual.blocks.0.attn.q_rot", "POST", "q_rot_out"),
    ("block0_k_rot",      "model.visual.blocks.0.attn.k_rot", "POST", "k_rot_out"),
    ("block0_attn_heads", "model.visual.blocks.0.attn",       "POST", "attn_heads_out"),
    ("block0_attn_merged","model.visual.blocks.0.attn",       "POST", "attn_out"),
    ("block0_proj",       "model.visual.blocks.0.attn.proj",  "POST", "o_proj_out"),
    ("block0_attn_resid", "model.visual.blocks.0",            "POST", "attn_resid_out"),
    ("block0_norm2",      "model.visual.blocks.0.norm2",      "POST", "norm2_out"),
    ("block0_fc1",        "model.visual.blocks.0.mlp.linear_fc1", "POST", "fc1_out"),
    ("block0_fc2",        "model.visual.blocks.0.mlp.linear_fc2", "POST", "fc2_out"),
    ("block0_output",     "model.visual.blocks.0",            "POST", "block0_out"),
]


def main():
    d = torch.load(SRC, weights_only=False)
    c = d["captured"]
    os.makedirs(OUT, exist_ok=True)
    meta = []
    for name, modpath, hook, key in STAGES:
        if key not in c:
            print(f"skip {name}: {key} not captured")
            continue
        t = c[key]
        t = t.detach().float().cpu().contiguous().view(-1)
        t.numpy().tofile(os.path.join(OUT, f"oracle_block0_{name}.bin"))
        meta.append(f"{name}\t{modpath}\t{hook}\t{list(t.shape)}\t{t.dtype}")
        print(f"{name}: {t.numel()} floats")
    with open(os.path.join(OUT, "oracle_block0_manifest.txt"), "w") as f:
        f.write("fixture\tmodule_path\thook_type\tshape\tdtype\n")
        f.write("\n".join(meta) + "\n")
    print("done ->", OUT)


if __name__ == "__main__":
    main()