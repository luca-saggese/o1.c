#!/usr/bin/env python3
"""
Dump the frozen Python scheduler oracle for M1-post scheduler matrix validation.

Runs the actual oracle scheduler classes (flash / flow_match / default) through
build_scheduler semantics and emits:
  - the derived sigma schedules (incl. terminal 0.0)
  - the noise_scale_schedule
  - step outputs for a fixed synthetic example:
      * Euler step (flash & flow_match)
      * 3-step UniPC (default)

IMPORTANT: every RNG draw uses an explicit CPU torch.Generator (MT19937) seeded
once at the top.  This makes all STEP outputs bit-reproducible from a native
MT19937 port (src/runtime/torch_rng.c), so the C unit test can compare step
outputs directly without any CUDA / Philox dependency.

Usage:  python dump_scheduler_matrix_oracle.py <out.txt>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "python"))

import numpy as np
import torch
from models.pipeline import build_scheduler, DEFAULT_TIMESTEPS

# Explicit MT19937 CPU generator so native C can reproduce every draw.
G = torch.Generator(device="cpu").manual_seed(1234)

NOISE_SCALE = 8.0

# recipe: (steps, shift, scheduler_name, timesteps_list_or_None)
RECIPES = {
    "flash_dev28":       (28, 1.0, "flash",      DEFAULT_TIMESTEPS),
    "flash_dev28_gen":   (28, 1.0, "flash",      None),
    "flowmatch_dev28":   (28, 1.0, "flow_match", DEFAULT_TIMESTEPS),
    "flowmatch_dev28_gen": (28, 1.0, "flow_match", None),
    "default_full50":    (50, 3.0, "default",    None),
}

N = 64


def make_example():
    z = torch.randn(1, N, dtype=torch.float32, generator=G)
    mo = torch.randn(1, N, dtype=torch.float32, generator=G) * 0.3
    noise = torch.randn(1, N, dtype=torch.float32, generator=G)
    return z, mo, noise


def noise_scale_schedule(start, end, num_steps):
    if num_steps > 1:
        return [start + (end - start) * i / (num_steps - 1)
                for i in range(num_steps)]
    return [start]


def main():
    out_path = sys.argv[1]

    lines = []
    lines.append("# M1-post scheduler oracle dump (MT19937-seeded)")
    lines.append(f"# default noise_scale_start/end = {NOISE_SCALE}")

    for name, (steps, shift, sched_name, tlist) in RECIPES.items():
        sched = build_scheduler(steps, tlist, shift, "cpu", sched_name)
        sigmas = sched.sigmas
        num_steps = len(sched.timesteps)
        nss = noise_scale_schedule(NOISE_SCALE, NOISE_SCALE, num_steps)

        lines.append("")
        lines.append(f"RECIPE {name} {steps} {shift} {sched_name} {tlist is not None}")
        lines.append("SIGMAS " + " ".join(repr(float(s)) for s in sigmas.tolist()))
        lines.append("NOISESCALE " + " ".join(repr(v) for v in nss))

        z, mo, noise = make_example()

        if sched_name == "flash":
            out = sched.step(
                mo, torch.tensor(float(sched.timesteps[0]), dtype=torch.float32),
                z, s_noise=nss[0], noise_clip_std=NOISE_SCALE,
                generator=G, return_dict=False)[0]
            lines.append("STEP " + " ".join(repr(float(v)) for v in out[0].tolist()))
        elif sched_name == "flow_match":
            out = sched.step(
                mo, torch.tensor(float(sched.timesteps[0]), dtype=torch.float32),
                z, return_dict=False)[0]
            lines.append("STEP " + " ".join(repr(float(v)) for v in out[0].tolist()))
        else:  # default -> UniPC 3 steps
            zz = z.clone()
            for i in range(min(3, num_steps)):
                t = torch.tensor(float(sched.timesteps[i]), dtype=torch.float32)
                zz = sched.step(mo.clone(), t, zz, return_dict=False)[0]
            lines.append("STEP " + " ".join(repr(float(v)) for v in zz[0].tolist()))

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
