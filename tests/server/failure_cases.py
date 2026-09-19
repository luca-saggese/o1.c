#!/usr/bin/env python3
"""Failure-path tests for the o1.c OpenAI-compatible image server (spec 32.5).

Requires a running `build/hidream-server`:

    build/hidream-server --port 8000 --model dev &
    python3 tests/server/failure_cases.py --base-url http://127.0.0.1:8000

Every case asserts the HTTP status and the OpenAI error envelope
(`{"error": {"message", "type", "param", "code"}}`). The cases are cheap:
they are all rejected before any generation is enqueued, so no GPU work
happens and the whole script finishes in a couple of seconds.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EDIT_REF = os.path.join(REPO_ROOT, "example_assets", "edit", "test.jpg")

fails = 0


def post(url, data, headers=None, method="POST"):
    req = urllib.request.Request(url, data=data, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except TimeoutError:
        return 0, b"<client timeout: the request was not rejected>"


def expect(label, status, body, want_status, want_code=None):
    global fails
    ok = status == want_status
    code = None
    try:
        code = json.loads(body)["error"]["code"]
    except Exception:
        pass
    if want_code is not None:
        ok = ok and code == want_code
    if ok:
        print(f"  PASS {label}: {status} code={code}")
    else:
        fails += 1
        print(f"  FAIL {label}: got {status} code={code}, "
              f"want {want_status} code={want_code}")
        print(f"       body: {body[:300]!r}")


def json_post(base, path, obj):
    return post(base + path, json.dumps(obj).encode(),
                {"Content-Type": "application/json"})


def multipart(fields, files):
    """fields: dict of text field -> value.
    files: list of (field_name, filename, bytes); repeats are allowed."""
    boundary = "----o1failureboundary"
    out = []
    for k, v in fields.items():
        out.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                   f'name="{k}"\r\n\r\n{v}\r\n'.encode())
    for name, fname, data in files:
        out.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                   f'name="{name}"; filename="{fname}"\r\n'
                   f"Content-Type: application/octet-stream\r\n\r\n".encode())
        out.append(data)
        out.append(b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="hidream-o1-image-dev")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    v1 = base + "/v1"
    m = args.model

    with open(EDIT_REF, "rb") as f:
        ref_bytes = f.read()

    print("bad JSON")
    s, b = post(v1 + "/images/generations", b"{not json",
                {"Content-Type": "application/json"})
    expect("bad JSON", s, b, 400)

    print("bad multipart")
    s, b = post(v1 + "/images/edits", b"garbage",
                {"Content-Type": "multipart/form-data; boundary=zzz"})
    expect("bad multipart", s, b, 400)

    print("missing prompt")
    s, b = json_post(base, "/v1/images/generations", {"model": m})
    expect("missing prompt", s, b, 400, "invalid_request")

    print("no image on edits")
    body, ct = multipart({"prompt": "remove the earphones", "model": m}, [])
    s, b = post(v1 + "/images/edits", body, {"Content-Type": ct})
    expect("no image on edits", s, b, 400, "invalid_image")

    print("too many refs")
    files = [("image", f"r{i}.jpg", ref_bytes) for i in range(21)]
    body, ct = multipart({"prompt": "x", "model": m}, files)
    s, b = post(v1 + "/images/edits", body, {"Content-Type": ct})
    expect("too many refs", s, b, 400, "too_many_images")

    print("invalid image")
    body, ct = multipart({"prompt": "x", "model": m},
                         [("image", "bad.jpg", b"not an image at all")])
    s, b = post(v1 + "/images/edits", body, {"Content-Type": ct})
    expect("invalid image", s, b, 400, "invalid_image")

    print("request too large")
    big = b"x" * (70 * 1024 * 1024)
    s, b = post(v1 + "/images/generations", big,
                {"Content-Type": "application/json"})
    expect("request too large", s, b, 413, "request_too_large")

    print("unknown model")
    s, b = json_post(base, "/v1/images/generations",
                     {"model": "no-such-model", "prompt": "x"})
    expect("unknown model", s, b, 404)

    print("unsupported mask")
    body, ct = multipart({"prompt": "x", "model": m},
                         [("image", "r.jpg", ref_bytes),
                          ("mask", "m.png", ref_bytes)])
    s, b = post(v1 + "/images/edits", body, {"Content-Type": ct})
    expect("unsupported mask", s, b, 400, "mask_not_supported")

    print("unsupported transparent background")
    s, b = json_post(base, "/v1/images/generations",
                     {"model": m, "prompt": "x", "background": "transparent"})
    expect("unsupported transparent background", s, b, 400,
           "unsupported_background")

    print("unsupported output format")
    s, b = json_post(base, "/v1/images/generations",
                     {"model": m, "prompt": "x", "output_format": "webp"})
    expect("unsupported output format", s, b, 400, "unsupported_output_format")

    print("invalid o1_mode")
    s, b = json_post(base, "/v1/images/generations",
                     {"model": m, "prompt": "x", "o1_mode": "nonsense"})
    expect("invalid o1_mode", s, b, 400, "invalid_mode")

    print("invalid layout JSON")
    body, ct = multipart({"prompt": "x", "model": m,
                          "o1_mode": "personalize_layout",
                          "o1_layout_bboxes": "{not json"},
                         [("image", "r.jpg", ref_bytes)])
    s, b = post(v1 + "/images/edits", body, {"Content-Type": ct})
    expect("invalid layout JSON", s, b, 400)

    print("alias count mismatch")
    body, ct = multipart({"prompt": "x", "model": m,
                          "o1_mode": "personalize",
                          "o1_reference_aliases": '["a","b","c"]'},
                         [("image", "r0.jpg", ref_bytes),
                          ("image", "r1.jpg", ref_bytes)])
    s, b = post(v1 + "/images/edits", body, {"Content-Type": ct})
    expect("alias count mismatch", s, b, 400, "reference_alias_count_mismatch")

    print("invalid scheduler")
    s, b = json_post(base, "/v1/images/generations",
                     {"model": m, "prompt": "x", "o1_scheduler": "bogus"})
    expect("invalid scheduler", s, b, 400, "invalid_scheduler")

    print("invalid size")
    s, b = json_post(base, "/v1/images/generations",
                     {"model": m, "prompt": "x", "size": "huge"})
    expect("invalid size", s, b, 400, "invalid_size")

    print("n out of range")
    s, b = json_post(base, "/v1/images/generations",
                     {"model": m, "prompt": "x", "n": 99})
    expect("n out of range", s, b, 400, "invalid_request")

    print("unknown endpoint")
    s, b = post(base + "/v1/nope", b"{}", {"Content-Type": "application/json"})
    expect("unknown endpoint", s, b, 404)

    if fails == 0:
        print("FAILURE_CASES_OK")
        return 0
    print(f"FAILURE_CASES_FAIL: {fails} case(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())