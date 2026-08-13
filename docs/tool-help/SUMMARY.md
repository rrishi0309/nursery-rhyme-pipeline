# Tool CLI Surfaces — Task 1 Summary

Captured on 2026-08-12 on a MacBook Pro M4 Max (40-core GPU, 64 GB unified
memory, macOS 26.5.2). Every `--help` capture below is **verbatim real
output** — no flag was invented. Where a `--help` capture alone was not
enough to determine the real invocation shape (ACE-Step), source was read
directly and is called out as such.

**Downstream rule:** every provider task (2–6) must use only flags that
appear in this file or in the matching `docs/tool-help/<tool>.txt`. If a
flag you need is not here, that is a gap — report it, do not guess.

---

## 1. Lyrics — mlx-lm (`mlx_lm.generate`)

- **Stage:** lyrics/writing (Qwen)
- **Install status:** OK. `uv add mlx-lm` succeeded cleanly in the project's
  own venv (`.venv`, Python 3.14). Installed `mlx-lm==0.31.3`.
- **Working invocation:**
  `uv run mlx_lm.generate --help`
  (the console-script form; `uv run python -m mlx_lm.generate` also works
  but prints a deprecation warning telling you to use this form or
  `python -m mlx_lm generate`)
- **Full help:** `docs/tool-help/mlx-lm.txt`
- **Key flags:**
  - `--model MODEL` — local dir or HF repo. No default suffix shown in
    help text; help text says "If no model is specified, then
    mlx-community/Llama-3.2-3B-Instruct-4bit is used."
  - `--prompt, -p PROMPT` — message to process (`'-'` reads stdin)
  - `--max-tokens, -m MAX_TOKENS`
  - `--temp TEMP` (note: `--temp`, not `--temperature`)
  - `--seed SEED`
  - `--system-prompt SYSTEM_PROMPT`
  - `--verbose VERBOSE`
  - **No `--output` flag exists.** Generated text goes to stdout only.
    This matches the plan's design (Task 3 extracts the first balanced
    `{...}` JSON block from stdout) — not a gap, just confirming there is
    no file-output path to rely on.
- **Default model repo used by the pipeline:** `mlx-community/Qwen3.5-4B-MLX-4bit`
  — confirmed to exist on Hugging Face via `HfApi.model_info` (metadata
  only, no weights pulled).

---

## 2. Images — FLUX.2 via mflux (`mflux-generate-flux2`)

- **Stage:** cast sheet + scene stills
- **Install status:** OK, already present. `mflux==0.18.1` was already a
  project dependency and `mflux-generate-flux2` was already on PATH — no
  `uv add --upgrade mflux` was needed.
- **Working invocation:**
  `uv run mflux-generate-flux2 --help`
  `uv run mflux-generate-flux2 --prompt "..." --model flux2-klein-4b --image cast.png 0.4 --output scene_01.png --seed 42 --width 1344 --height 768 --steps <N>`
- **Full help:** `docs/tool-help/mflux-generate-flux2.txt` (FLUX.2 entry
  point) and `docs/tool-help/mflux-generate.txt` (original FLUX.1 entry
  point, captured per brief step 2 for completeness — still used by
  `image_flux.py`).
- **Key flags:**
  - `--prompt PROMPT` / `--prompt-file PROMPT_FILE`
  - `--negative-prompt NEGATIVE_PROMPT`
  - `--seed SEED [SEED ...]` (accepts multiple)
  - `--width WIDTH` / `--height HEIGHT` (default: source image dimensions)
  - `--steps STEPS`
  - `--guidance GUIDANCE` (default 3.5 for most, 30 for fill tools, 10 for depth)
  - `--model, -m MODEL` — choices include `flux2-klein-4b`, `flux2-klein-9b`,
    `flux2-klein-9b-kv`, `flux2-klein-base-4b`, `flux2-klein-base-9b`, plus
    non-FLUX.2 models (dev, schnell, qwen, z-image, ...), or any HF
    `org/model` repo, or a local path
  - `--quantize, -q {3,5,4,6,8}`
  - `--low-ram`, `--mlx-cache-limit-gb`
  - `--image PATH [STRENGTH ...]` — see critical finding below
  - `--output OUTPUT` (default `"image.png"`)
  - `--metadata` — export generation metadata as JSON sidecar
- **Default model repo:** the `flux2-klein-4b` alias resolves (confirmed by
  reading `mflux/models/common/config/model_config.py`) to
  `black-forest-labs/FLUX.2-klein-4B` on Hugging Face — confirmed to exist
  via `HfApi.model_info` (no weights pulled).

### CRITICAL FINDING — FLUX.2 reference-image flag

**`--image PATH [STRENGTH ...]`** is the only image-conditioning flag on
`mflux-generate-flux2`. Its help text: *"Init image as an atomic PATH with
optional STRENGTH (default 0.4): --image photo.jpg 0.6. Preferred over
--image-path/--image-strength."* Reading `mflux/cli/parser/parsers.py`
confirms it is declared with `nargs="+"` as a **single non-repeatable**
argument (unlike `--lora`, which the help text explicitly marks
"Repeatable").

This means: FLUX.2 Klein 4B via `mflux` exposes **one init-image / img2img
strength control, not a dedicated multi-reference IP-adapter for character
identity.** Passing the cast sheet via `--image cast.png <strength>` will
influence the output through image-to-image denoising (structure/color
bleed-through controlled by strength), not through an explicit
identity-preserving reference-conditioning path. This is exactly the risk
the design spec flagged as unverified ("FLUX.2 Klein 4B reference
conditioning is unverified... whether the distilled Klein 4B exposes the
same reference-image path through mflux must be tested, not assumed").
**Task 5 should treat `--image` as the best-available approximation, not a
guaranteed character-consistency mechanism**, and should be prepared to
fall back to text-only cast descriptions if visual results are poor.

---

## 3. Song — ACE-Step 1.5 (`cli.py`)

- **Stage:** song (vocals + music)
- **Install status:** OK, but **not runnable from the project's Python
  3.14 venv.** Cloned to `~/.cache/nursery-tools/ACE-Step-1.5` and ran
  `uv sync` there; it auto-selected **Python 3.12.8** for that repo's own
  `.venv` because `pyproject.toml` declares
  `requires-python = ">=3.11,<3.13"`. 125 packages installed cleanly
  (torch, transformers, mlx, mlx-lm, diffusers, gradio, etc. — no model
  weights were pulled).
- **Working invocation (uses ACE-Step's own venv, not the project's):**
  `~/.cache/nursery-tools/ACE-Step-1.5/.venv/bin/python cli.py --config <path/to/config.toml>`
  (run with cwd = the repo root, or pass an absolute path to `cli.py`)
- **Full captured help:** `docs/tool-help/acestep.txt`
  (`uv run python cli.py --help`, run inside the cloned repo)

### CRITICAL FINDING — no direct generate flags; wizard/config-only CLI

`cli.py --help` exposes only `-c/--config`, `--configure`,
`--backend {vllm,pt,mlx}`, `--log-level`. **There are no `--prompt`,
`--lyrics`, `--output`, `--seed`, `--duration` flags on the command line at
all.** Per `docs/en/CLI.md` in the cloned repo, ACE-Step's CLI is
"wizard/config only": you either run the interactive wizard (`python
cli.py`) or pass a `.toml` config file (`python cli.py --config
config.toml`). The song provider (Task 4) **must write a TOML file** and
invoke `cli.py --config <file>`, not construct a flag-based command line.

The full set of recognized TOML keys was recovered by reading the
`defaults` dict in `cli.py::main()` and the `GenerationParams` /
`GenerationConfig` dataclasses in `acestep/inference.py` (verbatim copy
kept at the bottom of `docs/tool-help/acestep.txt`). The fields the song
provider needs are, at minimum:

  - `save_dir` (str, default `"output"`) — output directory
  - `audio_format` (str, default `"flac"`; also `"wav"`, `"mp3"`, ...)
  - `caption` (str) — style/genre description
  - `lyrics` (str|None) — the song lyrics text
  - `duration` (float, default `-1.0` = auto)
  - `instrumental` (bool)
  - `task_type` (str, default `"text2music"`)
  - `inference_steps` (int, default `8`)
  - `seed` (int, default `-1` = random)
  - `guidance_scale` (float, default `7.0`)
  - `backend` (str: `"vllm"|"pt"|"mlx"`; auto-detects `"mlx"` on Apple
    Silicon — matches this machine)

- **Default model repo:** main checkpoint bundle is
  `ACE-Step/Ace-Step1.5` (vae, Qwen3-Embedding-0.6B, `acestep-v15-turbo`
  DiT, `acestep-5Hz-lm-1.7B` LM), auto-downloaded on first run. The design
  spec calls for the XL turbo variant: `ACE-Step/acestep-v15-xl-turbo`
  (≥12 GB VRAM/unified-memory recommended; this machine's 64 GB comfortably
  qualifies for the "unlimited" tier the tool detected: `GPU Configuration
  Detected: ... GPU Memory: 51.84 GiB ... Configuration Tier: unlimited`).

---

## 4. Alignment — whisperx-mlx (`whisperx`)

- **Stage:** align (word-level timestamps for lyric lines)
- **Install status:** OK, but **not on PyPI and not runnable from the
  project's Python 3.14 venv.** `uv add whisperx-mlx` in the project venv
  **FAILED**:
  ```
  × No solution found when resolving dependencies for split (markers:
  │ python_full_version >= '3.14' and sys_platform == 'darwin'):
  ╰─▶ Because whisperx-mlx was not found in the package registry and your
      project depends on whisperx-mlx, we can conclude that your project's
      requirements are unsatisfiable.
  ```
  Confirmed via direct PyPI JSON API lookup: `whisperx-mlx` returns 404 —
  it is genuinely not published. Per the brief's fallback, cloned
  `https://github.com/taavi223/whisperx-mlx` (a fork of
  `sooth/whisperx-mlx`) to `~/.cache/nursery-tools/whisperx-mlx` and ran
  `uv sync` there. It auto-selected **Python 3.10.11** (repo's
  `pyproject.toml` declares `requires-python = ">=3.9,<3.13"`, package
  name is actually `"whisperx"` v3.4.2). 115 packages installed cleanly.
- **Working invocation (uses its own venv):**
  `~/.cache/nursery-tools/whisperx-mlx/.venv/bin/whisperx <audio.wav> --backend mlx -o <out_dir> -f json`
- **Full captured help:** `docs/tool-help/whisperx-mlx.txt`
  (`uv run whisperx --help`, run inside the cloned repo)
- **Key flags:**
  - `audio` — positional, one or more audio file paths
  - `--backend {auto,mlx,standard,batch,lightning,mlx_lightning}`
    (default `auto`; pass `mlx` explicitly to guarantee Apple Silicon path)
  - `--model MODEL` (default `"small"`; e.g. `large-v3`)
  - `--output_dir, -o OUTPUT_DIR` (default `"."`)
  - `--output_format, -f {all,srt,vtt,txt,tsv,json,aud}` (default `all`;
    use `json` for word-level timestamps)
  - `--word_timestamps WORD_TIMESTAMPS` (default `False`; **note: help
    text says this "requires backend='lightning'"** — if word-level JSON
    timestamps come out empty under `--backend mlx`, this is likely why;
    Task 4 may need `--backend lightning` or `--backend mlx_lightning`
    instead of plain `mlx` to get word timestamps)
  - `--no_align` (skip phoneme-level alignment)
  - `--language LANGUAGE` (default `None` = auto-detect)
  - No `--seed` flag (ASR/alignment is deterministic given model + audio;
    not a generative sampling process, so this is expected, not a gap).

---

## 5. Video — ltx-2-mlx (`ltx-2-mlx generate`)

- **Stage:** animate (image-to-video)
- **Install status:** OK, but required a packaging workaround, and **not
  runnable from the project's Python 3.14 venv.** Cloned
  `https://github.com/dgrauet/ltx-2-mlx` to
  `~/.cache/nursery-tools/ltx-2-mlx`. `uv sync --all-extras` **initially
  FAILED**:
  ```
  ValueError: Readme path must be within the project directory: ../../README.md
  ```
  Each workspace package (`ltx-core-mlx`, `ltx-pipelines-mlx`,
  `ltx-trainer`) declares `readme = "../../README.md"` in its own
  `pyproject.toml`, which hatchling's editable-build validation rejects
  because the resolved path escapes that package's own directory.
  **Workaround applied inside the cloned tool checkout only** (not in the
  nursery project repo): copied the repo-root `README.md` into each
  package directory and changed each package's `readme` field to
  `"README.md"`. After that, `uv sync --all-extras` completed
  successfully, auto-installing **Python 3.11.15** (repo requires
  `>=3.11`; no upper bound, but classifiers only list up to 3.13 — 3.14
  was not tested and uv did not select it).
- **Working invocation (uses its own venv):**
  `~/.cache/nursery-tools/ltx-2-mlx/.venv/bin/ltx-2-mlx generate --prompt "..." --image scene.png --output clip.mp4 --frame-rate 24 ...`
- **Full captured help:** `docs/tool-help/ltx-2-mlx.txt` (top-level
  `ltx-2-mlx --help` plus `generate --help` and `info --help`, all
  verbatim)
- **Key flags on `generate`:**
  - `--prompt, -p PROMPT` (required)
  - `--output, -o OUTPUT` (required, `.mp4`)
  - `--image, -i ARG [ARG ...]` — reference image for I2V:
    `PATH [FRAME_IDX STRENGTH [CRF]]`; PATH alone = `FRAME_IDX=0
    STRENGTH=1.0`; **repeatable**, so both start and end frames can be
    anchored (e.g. `--image foo.jpg 0 1.0 --image foo.jpg 96 1.0`)
  - `--model, -m MODEL` (default `dgrauet/ltx-2.3-mlx-q8`)
  - `--seed, -s SEED` (`-1` = random)
  - `--height, -H` / `--width, -W` (defaults 480 / 704)
  - `--frames, -f FRAMES` (default 97)
  - `--frame-rate FRAME_RATE` — **mandatory, no default** (help text warns
    LTX-2.3 was trained at 24 fps; values far from that drift out of
    distribution)
  - `--steps STEPS` (one-stage denoising steps, default 8)
  - `--two-stage` / `--two-stages-hq` / `--distilled` / `--one-stage` —
    pipeline mode selectors; `--stage1-steps` / `--stage2-steps` tune each
  - `--cfg-scale`, `--stg-scale`
  - `--low-ram` — streams transformer blocks from mmap'd safetensors,
    cuts peak Metal memory ~75% (e.g. q8 ~10-12 GB → ~2.8 GB)
  - `--lora PATH STRENGTH` (repeatable)
- **Default model repo:** `dgrauet/ltx-2.3-mlx-q8` — confirmed to exist on
  Hugging Face via `HfApi.model_info` (22 siblings, no weights pulled).
  `dgrauet/ltx-2.3-mlx-q4` also exists (referenced in the tool's own `info`
  example) and is closer to the design spec's "int4 ~12 GB" sizing; Task 2
  should default to the q4 repo unless quality testing favors q8.

### CRITICAL FINDING — no audio-skip flag on `generate`

`ltx-2-mlx generate` has **no flag to skip, mute, or discard audio
generation.** The full flag list (~30 flags, reproduced above and in full
in `docs/tool-help/ltx-2-mlx.txt`) contains nothing audio-related. Per the
tool's own README: *"Text-to-Video — generate video + stereo 48kHz audio
from a text prompt"* — LTX-2.3 is a joint audio-video diffusion model, and
`generate` always bakes an audio track into the output `.mp4`. The only
audio-related flags in the whole CLI (`--audio`, `--audio-start`,
`--no-regen-audio`) live on the separate `a2v` subcommand, which is
audio-*conditioned* generation (an existing audio file steers the video),
not applicable to the `generate` stage used here.

**Conclusion: LTX's audio cannot be suppressed at generation time.** The
pipeline must strip it downstream — exactly as the existing plan already
anticipates (`nursery/stages/assemble.py`: use `-an` on the LTX clip
inputs so only the ACE-Step song is heard). This is a confirmation of the
plan's existing design, not a new blocker, but it is worth stating
explicitly: no `--no-audio` flag exists to name.

---

## Cross-cutting notes for Task 2 (config schema)

- **Every non-mlx-lm, non-mflux tool needs its own interpreter.** The
  project venv is Python 3.14; ACE-Step 1.5 requires `<3.13`, whisperx-mlx
  requires `<3.13`, and ltx-2-mlx auto-selected 3.11 (untested above
  3.13). Each of these was installed into its own `uv`-managed venv under
  `~/.cache/nursery-tools/<tool>/.venv/`. **Every provider config for
  ACE-Step, whisperx-mlx, and ltx-2-mlx needs a `python` (interpreter) path
  field pointing at that tool's own venv** — this is exactly the scenario
  the brief anticipated when it said "the provider config already supports
  a per-tool `python` path for exactly this reason." `mlx-lm` and `mflux`
  are the only two tools that run fine from the project's own `.venv`.
- **No model weights were downloaded** for any tool. All `--help` captures
  ran instantly; the two Hugging Face repo existence checks
  (`mlx-community/Qwen3.5-4B-MLX-4bit`, `black-forest-labs/FLUX.2-klein-4B`,
  `dgrauet/ltx-2.3-mlx-q8`, `dgrauet/ltx-2.3-mlx-q4`) used
  `HfApi.model_info()`, which fetches metadata only.

## Install status at a glance

| Tool | Package/repo | Install status | Runs from project venv (3.14)? |
|---|---|---|---|
| mlx-lm | `mlx-lm` (PyPI) | OK | Yes |
| FLUX.2 | `mflux` (PyPI, already installed) | OK | Yes |
| ACE-Step 1.5 | `github.com/ace-step/ACE-Step-1.5` | OK (own venv, Python 3.12.8) | No — requires `<3.13` |
| whisperx-mlx | `github.com/taavi223/whisperx-mlx` | OK (own venv, Python 3.10.11) | No — requires `<3.13`; not on PyPI |
| ltx-2-mlx | `github.com/dgrauet/ltx-2-mlx` | OK (own venv, Python 3.11.15; required a readme-path packaging workaround) | No — 3.14 untested/unselected by uv |

No tool failed to install outright. The two build-relevant surprises were
(1) ACE-Step's CLI being config-file-only with no direct generate flags,
and (2) ltx-2-mlx's workspace packages needing a local readme-path patch
before `uv sync --all-extras` would succeed.
