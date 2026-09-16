#!/usr/bin/env python3
"""Generate bitwise golden files for the torch-exact CPU RNG port.

Pure oracle capture: draws torch.randn with a CPU MT19937 generator and
writes the raw float32 bit patterns. No transformer forward, no model load.
"""
import os
import struct
import torch

OUT = "artifacts/m1/golden/M1_RNG"
SEED = 123457  # sanity seed 123456 + 1
SIZES = [1, 8, 15, 16, 17, 31, 32, 1024]

os.makedirs(OUT, exist_ok=True)
for n in SIZES:
    g = torch.Generator("cpu").manual_seed(SEED)
    x = torch.randn(n, generator=g, dtype=torch.float32)
    bits = struct.pack(f"<{n}f", *x.tolist())
    with open(f"{OUT}/randn_n{n}_seed{SEED}.txt", "w") as f:
        for i in range(0, len(bits), 4):
            f.write(f"{int.from_bytes(bits[i:i+4], 'little'):08x}\n")
    print(f"wrote randn_n{n}_seed{SEED}.txt ({n} values)")
print("RNG_GOLDEN_OK")
