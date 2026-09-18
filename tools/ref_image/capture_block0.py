#!/usr/bin/env python3
"""Capture block-0 stage snapshots from the oracle Qwen3-VL vision tower.

Uses the canonical custom model class (loads visual weights correctly).
Saves artifacts/ref_image_audit/oracle_block0_stages.pt with PRE/POST
fixtures and a manifest with module path / hook type / shape / dtype.
"""
import math
import sys

import torch
from transformers import AutoProcessor, AutoConfig
from models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration
from PIL import Image

torch.manual_seed(0)
dev = "cuda"
dtype = torch.bfloat16

p = AutoProcessor.from_pretrained("/home/lvx/o1.c/models/dev", trust_remote_code=True)
cfg = AutoConfig.from_pretrained("/home/lvx/o1.c/models/dev", trust_remote_code=True)
m = Qwen3VLForConditionalGeneration.from_pretrained(
    "/home/lvx/o1.c/models/dev", config=cfg,
    torch_dtype=dtype, device_map="cuda", local_files_only=True).eval()

img = Image.open("/home/lvx/o1.c/example_assets/edit/test.jpg").convert("RGB")
W, H = img.size
ratio = W / H
width = math.sqrt(384 * 384 * ratio)
height = width / ratio
cw = int(width / 32) * 32
ch = int(height / 32) * 32
pil_cond = img.resize((cw, ch), Image.LANCZOS)

content = [{"type": "image"}, {"type": "text", "text": "make it a sunset"}]
msgs = [{"role": "user", "content": content}]
tpl = p.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
proc = p(text=[tpl], images=[pil_cond], padding="longest", return_tensors="pt")
pv = proc.pixel_values.to(dev, dtype)
grid = proc.image_grid_thw.to(dev)
print("grid:", grid.tolist(), "pv:", pv.shape, flush=True)

vis = m.visual
blk0 = vis.blocks[0]
captured = {}


def post(name):
    def fn(mod, args, out):
        captured[name] = out.detach().float().cpu()
    return fn


def pre(name):
    def fn(mod, args):
        captured[name] = args[0].detach().float().cpu()
        return None
    return fn


hooks = []
hooks.append(blk0.register_forward_pre_hook(pre("block0_input")))
hooks.append(blk0.norm1.register_forward_hook(post("norm1_out")))
hooks.append(blk0.attn.qkv.register_forward_hook(post("qkv_out")))
hooks.append(blk0.attn.register_forward_hook(post("attn_out")))
hooks.append(blk0.attn.proj.register_forward_hook(post("o_proj_out")))
hooks.append(blk0.norm2.register_forward_hook(post("norm2_out")))
hooks.append(blk0.mlp.linear_fc1.register_forward_hook(post("fc1_out")))
hooks.append(blk0.mlp.linear_fc2.register_forward_hook(post("fc2_out")))
hooks.append(blk0.register_forward_hook(post("block0_out")))

with torch.no_grad():
    image_embeds, deepstack = m.get_image_features(pv, grid)

for h in hooks:
    h.remove()

# attn_resid: block0_out - o_proj_out (residual add) is not directly
# capturable; compute it as block0_input + o_proj_out (the residual path).
captured["attn_resid_out"] = captured["block0_input"] + captured["o_proj_out"]

for k, v in captured.items():
    print("CAP", k, tuple(v.shape), v.dtype, flush=True)

torch.save(captured, "/home/lvx/o1.c/artifacts/ref_image_audit/oracle_block0_stages.pt")
print("saved oracle_block0_stages.pt", flush=True)