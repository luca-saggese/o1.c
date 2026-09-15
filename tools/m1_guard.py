#!/usr/bin/env python3
"""M1 preflight guardrails.

Subcommands:
  check-env     Fail-closed oracle environment guard (interpreter, packages,
                oracle SHA, oracle clean, offline mode).
  check-locks   Verify M0 lock/manifest consistency.
  ledger        Append a model-run ledger entry (V2+ only).

All oracle captures must pass `check-env` before any V2+ model execution.
This tool is V0/V1-only and performs no model forward.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Frozen M0 environment contract (artifacts/m0/env/packages.json).
FROZEN_TORCH = "2.12.1+cu130"
FROZEN_TRANSFORMERS = "4.57.1"


def _die(msg: str) -> "NoReturn":  # noqa: F821
    print(f"[m1_guard][ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


def _load_json(path: Path) -> dict:
    if not path.is_file():
        _die(f"missing required file: {path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        _die(f"invalid JSON in {path}: {e}")


def _git(path: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        _die(f"git -C {path} {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def check_env() -> dict:
    """Fail closed if the oracle environment does not match the M0 freeze."""
    out: dict = {}

    # 1. Interpreter. Do NOT resolve() the symlink: .venv/bin/python -> python3
    #    -> /usr/bin/python3 (system-site venv), so resolve() would hide the
    #    fact that we ran through the venv. Use sys.prefix instead.
    exe = sys.executable
    out["sys_executable"] = exe
    out["sys_prefix"] = sys.prefix
    if ".venv" not in sys.prefix:
        _die(
            f"wrong interpreter {exe}; sys.prefix={sys.prefix} does not "
            f"point at .venv (frozen transformers {FROZEN_TRANSFORMERS})"
        )

    # 2. Package versions.
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        _die(f"cannot import torch: {e}")
    try:
        import transformers
    except Exception as e:  # noqa: BLE001
        _die(f"cannot import transformers: {e}")

    out["torch"] = torch.__version__
    out["transformers"] = transformers.__version__
    if torch.__version__ != FROZEN_TORCH:
        _die(f"torch {torch.__version__} != frozen {FROZEN_TORCH}")
    if transformers.__version__ != FROZEN_TRANSFORMERS:
        _die(
            f"transformers {transformers.__version__} != frozen "
            f"{FROZEN_TRANSFORMERS}"
        )

    # 3. Oracle SHA + clean tree.
    lock = _load_json(ROOT / "config" / "oracle.lock")
    oracle_sha = lock["upstream_commit_sha"]
    head = _git(ROOT / "python", "rev-parse", "HEAD")
    out["oracle_sha_expected"] = oracle_sha
    out["oracle_sha_actual"] = head
    if head != oracle_sha:
        _die(f"oracle HEAD {head} != locked SHA {oracle_sha}")
    dirty = _git(ROOT / "python", "status", "--short")
    out["oracle_clean"] = dirty == ""
    if dirty:
        _die(f"oracle working tree dirty:\n{dirty}")

    # 4. Offline mode.
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        val = os.environ.get(var)
        out[var] = val
        if val != "1":
            _die(f"{var} must be set to '1' (got {val!r})")

    print(json.dumps({"m1_guard": "check-env", "ok": True, **out}, indent=2))
    return out


def check_locks() -> dict:
    """Verify M0 locks/manifests agree with each other and with disk state."""
    oracle = _load_json(ROOT / "config" / "oracle.lock")
    models = _load_json(ROOT / "config" / "models.lock")
    dev = _load_json(ROOT / "config" / "dev.json")
    base = _load_json(ROOT / "config" / "base.json")
    manifest = _load_json(ROOT / "config" / "startup_manifest_dev.json")

    problems: list[str] = []

    # oracle.lock <-> python HEAD.
    head = _git(ROOT / "python", "rev-parse", "HEAD")
    if head != oracle["upstream_commit_sha"]:
        problems.append(
            f"oracle HEAD {head} != oracle.lock {oracle['upstream_commit_sha']}"
        )

    # models.lock dev <-> dev.json.
    if models["profiles"]["dev"]["immutable_revision"] != dev["immutable_revision"]:
        problems.append("models.lock dev revision != dev.json revision")
    if models["profiles"]["dev"]["hf_repo"] != dev["hf_repo"]:
        problems.append("models.lock dev repo != dev.json repo")
    # models.lock base <-> base.json.
    if models["profiles"]["base"]["immutable_revision"] != base["immutable_revision"]:
        problems.append("models.lock base revision != base.json revision")

    # manifest <-> dev.json + models.lock.
    if manifest["hf_revision"] != dev["immutable_revision"]:
        problems.append("startup manifest hf_revision != dev.json revision")
    if manifest["hf_repo"] != dev["hf_repo"]:
        problems.append("startup manifest hf_repo != dev.json repo")
    if manifest["oracle_sha"] != oracle["upstream_commit_sha"]:
        problems.append("startup manifest oracle_sha != oracle.lock")

    out = {
        "m1_guard": "check-locks",
        "ok": not problems,
        "oracle_sha": oracle["upstream_commit_sha"],
        "dev_revision": dev["immutable_revision"],
        "base_revision": base["immutable_revision"],
        "problems": problems,
    }
    print(json.dumps(out, indent=2))
    if problems:
        _die("M0 lock consistency check failed")
    return out


def ledger(args: argparse.Namespace) -> None:
    """Append one run-ledger entry (V2+ only)."""
    required = (
        "engine_commit",
        "oracle_commit",
        "model_profile",
        "model_revision",
        "implementation",
        "validation_level",
        "purpose",
        "resolution",
        "steps",
        "whole_model_forwards",
        "seed",
        "golden_id",
        "result",
        "notes",
    )
    for k in required:
        if k not in args.__dict__:
            _die(f"ledger: missing field {k}")

    if args.validation_level in ("V2", "V3", "V4", "V5", "V6"):
        # Expensive run: print intended level prominently, then record.
        print(
            f"[m1_ledger] INTENDED VALIDATION LEVEL: {args.validation_level} "
            f"({args.implementation}, {args.model_profile})"
        )
    else:
        _die("ledger only records V2+ model executions")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "engine_commit": args.engine_commit,
        "oracle_commit": args.oracle_commit,
        "model_profile": args.model_profile,
        "model_revision": args.model_revision,
        "implementation": args.implementation,
        "validation_level": args.validation_level,
        "purpose": args.purpose,
        "resolution": args.resolution,
        "steps": int(args.steps),
        "whole_model_forwards": int(args.whole_model_forwards),
        "seed": int(args.seed),
        "golden_id": args.golden_id,
        "result": args.result,
        "notes": args.notes,
    }
    path = ROOT / "artifacts" / "m1" / "run_ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"[m1_ledger] recorded {args.validation_level} entry for "
          f"{args.implementation}/{args.model_profile}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="m1_guard")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-env", help="fail-closed oracle env guard")
    sub.add_parser("check-locks", help="M0 lock/manifest consistency check")

    lp = sub.add_parser("ledger", help="record a V2+ model-run ledger entry")
    for k in (
        "engine_commit", "oracle_commit", "model_profile", "model_revision",
        "implementation", "validation_level", "purpose", "resolution",
        "steps", "whole_model_forwards", "seed", "golden_id", "result", "notes",
    ):
        lp.add_argument(f"--{k}", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "check-env":
        check_env()
    elif args.cmd == "check-locks":
        check_locks()
    elif args.cmd == "ledger":
        ledger(args)


if __name__ == "__main__":
    main()
