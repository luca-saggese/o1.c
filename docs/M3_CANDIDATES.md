# M3 Candidates

Deferred I/O / system-level backlog for M3, kept out of M1.3a scope per
`docs/M1_3A_PRE_FORWARD_ARCHITECTURE_FREEZE.md` §64. None implemented now.

## Loader / I/O

- **Pinned `pread` loader** — double-buffered pinned host staging.
- **`pread` direct-to-pinned** — minimize copies; replace M1's load-once
  mechanism without changing transformer semantics (forward never reads files).
- **Large device arenas** — single big device allocation pool.
- **Soft / hard memory cap** — with reserve-before-allocate; M1.3a records
  that M2/M3 must derive safe limits from resident weights, persistent
  workspace, activation peak, pinned staging, CUDA/runtime overhead, and
  system reserve. It does **not** hardcode the old
  `soft=114 GiB / hard=118 GiB` values.
- **Async / optional component prefetch**.
- **`io_uring` / `preadv`** — async host I/O.
- **GDS / cuFile** — GPU-direct storage transfer.

## Note

M1.3a guarantees the transformer forward never opens model files, never parses
safetensors names, never performs layout conversion, and never loads a weight
on demand from disk — so future loader optimization can replace only the load
mechanism without touching transformer semantics.