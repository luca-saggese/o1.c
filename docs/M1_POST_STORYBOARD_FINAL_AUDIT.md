# M1-post — Final Storyboard Upstream Audit

Date: 2026-09-17
Scope: Final determination of official HiDream-O1-Image storyboard semantics
before M1-post closes. Supersedes the open question in
`docs/M1_POST_MODE_CONTRACTS.md` §6 and
`artifacts/m1post/audit/F_PROMPT_STORYBOARD_AUDIT.md`.

## Verdict: STORYBOARD_SEMANTICS_FOUND

Official executable semantics exist and are recoverable from the **technical
report** (arXiv:2605.11061, "HiDream-O1-Image: A Natively Unified Image
Generative Foundation Model with Pixel-level Unified Transformer"):

> **Storyboard = multi-panel image generation.** A single self-contained
> English prompt describes a *sequence of N panels* (same subject, same
> environment, distinct sequential actions/views); the model renders the whole
> panel sequence **in a single inference pass** as one image. Panel ordering is
> deterministic: it is fixed by the sequential structure of the prompt
> ("In the first panel, … In the second panel, …" / "In panel P1, … P9").

There is **no dedicated storyboard flag, API, tab, or code path** in the frozen
oracle (dev @ `3237a638a5c2c7be106b0175958f4c0db8c2dfbf`) or in the main branch.
Storyboard is realized as ordinary text-to-image generation from a long,
panel-sequential prompt — the model was trained on multi-panel grid data to
render panel sequences in one pass. The repo's own spec question
(`M1_POST_FULL_FEATURE_PARITY.md` §36, options A–D) resolves to **D: another
mechanism described in the technical report** — not a dedicated conditioning
mode (A), not multi-reference personalization (B), not prompt-agent
orchestration over repeated generations (C).

---

## Evidence trail

### 1. Local model cards (this repo)

- `models/base/README.md:42` — "One Model, Many Tasks — Text-to-image,
  long-text rendering, instruction editing, subject-driven personalization,
  and **storyboard generation** in a single architecture." (advertised only;
  no usage section for it).
- `models/dev/README.md` — **no** storyboard mention (grep: 0 matches). The
  dev card documents only text-to-image + the Prompt-Refine agent.

### 2. Upstream GitHub repo (HiDream-ai/HiDream-O1-Image)

- GitHub code search `storyboard repo:HiDream-ai/HiDream-O1-Image` →
  **exactly 1 result**: `README.md` (main branch). No storyboard in any
  `.py` file, any branch.
- Main `README.md:41` — same "One Model, Many Tasks" bullet as the local base
  card (advertised only). No storyboard usage/CLI section anywhere in the
  README (full 23.6 KB read).
- Dev branch (frozen oracle `3237a638…c2dfbf`) `README.md` — **no** storyboard
  mention at all (4 KB read in full).
- `inference.py` (dev `c723ab6…` and main `983b553…`) — single `--prompt` +
  optional `--ref_images`/`--layout_bboxes`; no storyboard/panel argument.
- `app.py` (dev `3964fda…` and main `2e0608d…`) — grep for
  `storyboard|story_board|story board`: **0 matches**. Tabs are
  text-to-image / editing / subject-driven only.
- `prompt_agent_v2.py` (dev `430f2dd…`) — single-prompt rewrite agent
  (`REWRITE_SYSTEM_PROMPT`, `refine_prompt`); no panel decomposition.
- `models/pipeline.py` — no storyboard path (per prior audit
  `F_PROMPT_STORYBOARD_AUDIT.md`).

### 3. Hugging Face model cards

- `https://huggingface.co/HiDream-ai/HiDream-O1-Image/raw/main/README.md` —
  identical to GitHub main README; storyboard only in the Key Features bullet.
- `https://huggingface.co/HiDream-ai/HiDream-O1-Image-Dev-2604/raw/main/README.md`
  — **no** storyboard mention.

### 4. HF Spaces demo (web demo)

- `https://huggingface.co/spaces/HiDream-ai/HiDream-O1-Image/raw/main/app.py`
  — thin Gradio wrapper over a hosted API: single prompt, `wh_ratio`,
  `negative_prompt`, `enable_prompt_refine`, `seed`, `guidance_scale`;
  request body has `"n": 1`. **No storyboard tab, no multi-panel UI.**

### 5. Technical report — arXiv:2605.11061 (AUTHORITATIVE)

PDF text extracted from `https://arxiv.org/pdf/2605.11061v1`; line numbers
below refer to the extracted text.

- **Capability claim** (contributions), line 292: "…subject-driven
  personalization, and **multipanel image generation for storyboard
  production**."
- **Single-pass semantics**, §6.3 lines 1298–1299: "its robust capability in
  multi-panel image generation enables the **coherent creation of storyboards
  within a single inference pass**."
- **Cinematic control**, lines 1292–1297: "fine-grained manipulation across
  15 distinct cinematic shots and camera perspectives": Shot Scales (extreme
  full … extreme close-up), Camera Angles (high, low, eye-level, bird's-eye),
  Subject Orientations (front, side, back, three-quarter).
- **Exact prompt format** (two full examples), lines 1369–1404:
  - "A sequence of **nine panels** depicting a transaction at a bustling
    evening market … **In the first panel**, a woman with dark hair … **In the
    second panel**, an older man … **In the third panel** … **In the ninth
    panel**, the woman turns back to her stall …" (same subject + same market
    environment across all panels; sequential actions).
  - "A sequence of nine panels depicting a stealth mission … **In panel P1**,
    a man … **In P2** … **In P9**, a close-up shows his gloved hand inserting a
    chip …" (same character + same corridor environment).
- **Training data**, lines 347–349: "we collect multi-panel data from two
  complementary sources: **grid images crawled from the Internet** and
  **frame-composition samples constructed from different clips of the same
  video**. These multi-panel data expose the model to **sequential changes,
  panelwise consistency**, and richer spatial organization…"
- **Annotation**, lines 460–462: "For multi-panel samples, Qwen3-VL describes
  both the **global arrangement and the panel-level differences**, enabling the
  model to learn **grid composition and temporal-frame consistency**."
- Figure 3 (line 165) and Figure 9 (lines 1405–1406) showcase multi-panel
  image generation scenarios.

## Recovered semantics (for M1-post implementation)

1. **One prompt → one image containing the panel grid.** No N-image loop, no
   per-panel orchestration, no shared-seed repetition. The frozen oracle's
   `inference.py` already executes this path unchanged (long prompt → single
   `generate_image` call).
2. **Prompt format**: "A sequence of N panels depicting <scenario>. In the
   first panel, <…>. In the second panel, <…>. … In the Nth panel, <…>."
   (or "In panel P1, … P2, …"). Panel descriptions are explicit, self-contained
   English sentences.
3. **Shared conditions**: same subject and same environment persist across all
   panels; only action/view/pose/camera change per panel.
4. **Deterministic ordering**: fixed by the sequential prompt structure
   (first/second/…/Nth, or P1…PN). No seed-based ordering mechanism.
5. **Grid composition**: the model learns grid layout from training data
   (internet grids + video frame sequences); no layout bbox input is required
   for storyboard (layout bboxes exist only for the IP-layout feature).

## Implication for the repo's 3-panel scenario (parity spec §64)

The frozen 3-panel scenario should use the report's prompt format verbatim:
"A sequence of three panels depicting <same subject, same environment>. In the
first panel, <action/view 1>. In the second panel, <action/view 2>. In the
third panel, <action/view 3>." — generated in a single inference pass with the
existing pipeline. This satisfies "same subject, same environment, three
distinct actions/views, deterministic ordering, consistent shared conditions"
without inventing any new mechanism.

## Gaps and uncertainties

- The report does not specify a canonical panel count or grid aspect ratio for
  storyboard; its examples use 9 panels. The repo's 3-panel scenario is a
  deliberate subset, not an upstream-documented count.
- No upstream code validates or post-processes the panel grid (no slicing of
  the output image into panels is documented); the grid is produced as-is by
  the model.
- The report's Figure 3/Figure 9 images were not machine-verified for panel
  count; the text examples (9 panels) are the authoritative format reference.
- Searched: GitHub code search (all branches), both READMEs, both app.py,
  both inference.py, prompt_agent.py, prompt_agent_v2.py, models/pipeline.py,
  both HF model cards, HF Spaces app.py, and the full technical report text.
  No other storyboard reference exists upstream.