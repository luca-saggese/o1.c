#!/usr/bin/env python3
"""Dump the oracle visual capture to raw .bin files for the native test.

Reads artifacts/ref_image_audit/oracle_visual_stages.pt (produced by
/tmp/oracle_full.py) and writes fp32 .bin files under
artifacts/ref_image_audit/oracle_dump/ for the native C test.
"""
import os
import sys
import torch

SRC = "artifacts/ref_image_audit/oracle_visual_stages.pt"
OUT = "artifacts/ref_image_audit/oracle_dump"

def main():
    d = torch.load(SRC, weights_only=False)
    os.makedirs(OUT, exist_ok=True)
    c = d["captured"]
    names = {
        "pixel_values": d["pixel_values"],
        "pos_embed": c["pos_embed"],
        "rot_pos": c["rot_pos"],
        "image_embeds": d["image_embeds"],
        "deepstack_0": d["deepstack"][0],
        "deepstack_1": d["deepstack"][1],
        "deepstack_2": d["deepstack"][2],
        "merger_out": c["merger_out"],
        "patch_embed": c["patch_embed"],
        "block0_in": c["block0_in"],
        "block0_out": c["block0_out"],
        "block8_out": c["block8_out"],
        "decoder_input_embeds": c["decoder_input_embeds"],
        "embed_tokens_raw": c["embed_tokens_raw"],
    }
    for name, t in names.items():
        t = t.detach().float().cpu().contiguous().view(-1)
        t.numpy().tofile(os.path.join(OUT, name + ".bin"))
        print(f"{name}: {t.numel()} floats")
    print("done ->", OUT)

if __name__ == "__main__":
    main()
