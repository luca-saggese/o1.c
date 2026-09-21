#!/usr/bin/env python3
"""OpenAI Python client smoke test for the o1.c image server.

Requires a running `build/hidream-server` and the `openai` package:

    build/hidream-server --port 8000 --model-path artifacts/models/hidream-o1-dev-bf16.gguf &
    python3 tests/server/openai_client_smoke.py --base-url http://127.0.0.1:8000/v1

The script exercises the official client end to end (no custom HTTP code):
generation, single-reference edit, and a repeated-image personalization call.
"""

import argparse
import base64
import os
import sys

from openai import OpenAI

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDIT_REF = os.path.join(REPO_ROOT, "example_assets", "edit", "test.jpg")
IP_REF = os.path.join(REPO_ROOT, "example_assets", "IP_2.jpg")


def check_png(path, label):
    with open(path, "rb") as f:
        head = f.read(8)
    if head != b"\x89PNG\r\n\x1a\n":
        raise SystemExit(f"{label}: {path} is not a PNG")
    print(f"  {label}: {path} ({os.path.getsize(path)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--model", default="hidream-o1-image-dev")
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--skip-edit", action="store_true")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print("models.list")
    models = client.models.list()
    ids = [m.id for m in models.data]
    print(f"  {ids}")
    if args.model not in ids:
        raise SystemExit(f"model {args.model} not advertised by the server")

    print("images.generate")
    r = client.images.generate(
        model=args.model,
        prompt="A cinematic photograph of a red fox in snow",
        size="2048x2048",
        extra_body={"o1_seed": 42, "o1_steps": 28},
    )
    gen_path = os.path.join(args.out_dir, "o1-server.png")
    with open(gen_path, "wb") as f:
        f.write(base64.b64decode(r.data[0].b64_json))
    check_png(gen_path, "generate")

    if args.skip_edit:
        print("SMOKE_OK")
        return

    print("images.edit (single reference)")
    with open(EDIT_REF, "rb") as f:
        r = client.images.edit(
            model=args.model,
            image=f,
            prompt="remove the earphones",
            extra_body={"o1_seed": 42, "o1_keep_original_aspect": True},
        )
    edit_path = os.path.join(args.out_dir, "o1-server-edit.png")
    with open(edit_path, "wb") as f:
        f.write(base64.b64decode(r.data[0].b64_json))
    check_png(edit_path, "edit")

    print("images.edit (repeated references)")
    with open(EDIT_REF, "rb") as a, open(IP_REF, "rb") as b:
        r = client.images.edit(
            model=args.model,
            image=[a, b],
            prompt="a portrait of @subject in the style of @style",
            extra_body={
                "o1_seed": 7,
                "o1_mode": "personalize",
                "o1_reference_aliases": ["subject", "style"],
            },
        )
    multi_path = os.path.join(args.out_dir, "o1-server-multiref.png")
    with open(multi_path, "wb") as f:
        f.write(base64.b64decode(r.data[0].b64_json))
    check_png(multi_path, "multiref")

    print("SMOKE_OK")


if __name__ == "__main__":
    sys.exit(main())
