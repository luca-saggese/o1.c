# Working Mode — Agent Engineering Discipline

This document is the persistent modus operandi for all agents and sessions
working on this repository. It is authoritative for the M0–M2 milestone plan.
When in doubt between this document and terminal output, this document wins.

## Project invariants

The following rules are non-negotiable:

1. **Python upstream is an oracle only.** Production inference must not
   depend on Python.
2. **Official upstream checkout lives at `/python`** relative to the project
   root.
3. **`/python` must be ignored by the engine repository.**
4. **Never patch files inside `/python`.**
5. **If `/python` becomes dirty, the oracle is invalid until restored and
   re-frozen.**
6. **Dev is the primary development checkpoint** (`HiDream-O1-Image-Dev-2604`).
7. **Dev and Base/Full must remain supported** through one shared
   implementation/configuration path. Do not hardcode Dev-only topology where
   model configuration differs.
8. **Prefer the cheapest validation level capable of proving a change.**
9. **Full model runs are scarce validation resources, not a debugging loop.**
10. **M0 permits V0 and V1 only.**
11. **M0 explicitly forbids transformer forward passes and image generation.**
12. **Revisions, versions, hashes, environment facts, and gate evidence must
    be persisted to files**, not left only in terminal output.

## Validation-cost policy

The validation ladder from `MILESTONES.md` is reproduced here as the
authoritative budget for this repository:

| Level | Model work | Intended use |
|---|---|---|
| V0 | none | parsing, configs, file layout, tensor metadata, scheduler tables |
| V1 | startup/load only | processor/model construction, state dict manifest, memory footprint |
| V2 | one selected layer/block | kernel parity, tensor layout, normalization/RoPE/GEMM checks |
| V3 | one whole model forward | end-to-end transformer numerical parity without denoising loop |
| V4 | 1–3 denoising steps | scheduler/state evolution and cross-step correctness |
| V5 | full Dev generation (28 steps) | milestone/regression closure only |
| V6 | full Base generation (50 steps) | release/major compatibility gate only |

For milestone **M0**:

- **Allowed:** V0, V1
- **Forbidden:** V2, V3, V4, V5, V6

Additional rules:

- Do not escalate validation level while a cheaper gate is failing.
- Never rerun an expensive model validation merely to gather basic diagnostics.
- Future golden tensors are captured once from the frozen oracle and reused.
- A failing V2/V3/V4 gate must be debugged at the same or lower level; do not
  rerun V5/V6 repeatedly.

## Repository ownership boundaries

| Path | Versioned by engine repo? |
|---|---|
| engine repository | yes (versioned) |
| `docs/` | yes (versioned) |
| `scripts/` | yes (versioned) |
| config templates | yes (versioned) |
| `/python/` | **NO** — owned by upstream oracle |
| `/models/` | **NO** — model weights |
| HF caches | **NO** |
| build output | **NO** |
| generated runtime data | **NO** |

The nested `/python/.git` belongs to the upstream oracle and must **never** be
absorbed by the engine repository.

## Evidence policy

Every M0 gate must leave durable evidence. At minimum persist:

```
artifacts/m0/
    env/
    oracle/
    models/
    startup/
```

The whole `artifacts/` tree remains git-ignored (see `.gitignore`).

Version only compact manifests/configuration that are intentionally part of the
reproducibility contract (e.g. `config/oracle.lock`, `config/models.lock`).
Large logs, model shards, caches, and generated binary data must **not** be
committed.

## Commit policy

One logical gate per commit.

Do not mix unrelated cleanup/refactoring into an M0 commit.

Use these milestone commit subjects unless a pre-existing matching commit is
already present:

1. `chore(m0): scaffold repo and define low-run validation policy`
2. `chore(m0): pin official hidream python oracle`
3. `chore(m0): pin dev checkpoint and model acquisition workflow`
4. `test(m0): freeze reproducible python startup oracle`

Before every commit:

```
git status --short
git diff --check
git diff
```

After every commit:

```
git status --short
git log -1 --oneline
```

The engine repository must be clean before proceeding to the next M0 sub-step,
except for intentional ignored data.

## Failure/stop policy

Do not improvise around failures that invalidate reproducibility. Stop the
current sub-step and record evidence if any of these occur:

- upstream revision cannot be resolved immutably;
- `/python` is unexpectedly tracked by engine Git;
- oracle working tree is dirty;
- model revision is moving/unpinned;
- local-only startup still accesses the network;
- Dev model files are incomplete;
- a script attempts a transformer forward during M0;
- CUDA/GPU identity cannot be established on the intended target host;
- required environment/package versions cannot be recorded;
- two startup-only manifests disagree in structural model metadata.

A failure is **not** permission to perform a more expensive model run.

## Session continuity

`docs/M0_STATUS.md` is maintained continuously and must always show:

- current M0 sub-step;
- last completed gate;
- current engine Git commit;
- upstream oracle SHA, once known;
- Dev HF revision, once known;
- Base HF revision/path status;
- commands successfully executed;
- commands failed;
- unresolved issues;
- exact next action.

Update `docs/M0_STATUS.md` before ending a work session or handing work to
another agent.
