# Handoff — video + audio nursery rhyme pipeline

Paused 2026-08-12. Branch `feat/video-audio-pipeline`, pushed to
`github.com/rrishi0309/nursery-rhyme-pipeline` (private).

Goal: `nursery build <slug>` produces `out/<video_id>/final.mp4`. No YouTube.

- Spec: `docs/superpowers/specs/2026-08-12-video-audio-pipeline-design.md`
- Plan: `docs/superpowers/plans/2026-08-12-video-audio-pipeline-plan.md`
- Ledger: `.superpowers/sdd/2026-08-12-video-audio-pipeline-plan/progress.md`
- Tool CLI ground truth: `docs/tool-help/SUMMARY.md` — **read this before
  writing any provider command line. Never invent a flag.**

## Pipeline

```
lyrics (Qwen3.5-4B) -> song (ACE-Step 1.5) -> align (whisperx)
   -> cast sheet (FLUX.2) -> stills (FLUX.2) -> animate (LTX-2.3) -> assemble (ffmpeg)
```

Timing flows from alignment: whisperx recovers per-line `(start_s, end_s)` from
the sung track, and those ARE the scene boundaries. There is no melody grid.

## Status

| Task | State | Commit |
|---|---|---|
| 1. Toolchain install + CLI capture | done, reviewed clean | `2bf27e4` |
| 2. Config schema | done, reviewed | `b64b1c2` |
| 3. Lyrics provider + validator | done, reviewed, **fixes partial** | `b64b1c2`, `7616133` |
| 4. Song + align providers | done, **not reviewed** | `b4b6425` |
| 5. FLUX.2 image provider | done, reviewed | `fb08df4` |
| 6. LTX provider rewrite | done, reviewed | `fb08df4` |
| 7. Orchestration + assemble + cleanup | **NOT STARTED** | — |
| 8. Smoke command + README | **NOT STARTED** | — |

All tests green at `7616133`. Nothing uncommitted.

## Models — all downloaded, ~92 GB in the HF cache

| Model | Size | Role |
|---|---|---|
| `mlx-community/Qwen3.5-4B-MLX-4bit` | 2.9 GB | lyrics, scenes, cast |
| `ACE-Step/Ace-Step1.5` | 9.4 GB | sung song + instrumental |
| `mlx-community/whisper-small-mlx` | 459 MB | line timings |
| `black-forest-labs/FLUX.2-klein-4B` | 15 GB | cast sheet + stills |
| `dgrauet/ltx-2.3-mlx-q4` | 19 GB | video clips |
| `mlx-community/gemma-3-12b-it-4bit` | 7.5 GB | **LTX text encoder — not in the q4 repo** |

Three tools cap at Python <3.13 and live in their own venvs under
`~/.cache/nursery-tools/`: ACE-Step (3.12.8), whisperx-mlx (3.10.11, not on
PyPI — cloned from `taavi223/whisperx-mlx`), ltx-2-mlx (3.11.15, needed a
readme-path patch inside its own checkout). Their interpreter paths are config
fields. `mlx-lm` and `mflux` run in the project venv (3.14).

~38 GB of the cache is the old FLUX.1-schnell / SDXL from the previous
pipeline. Reclaimable once FLUX.2 is wired in.

## OPEN — do these first

### 1. CRITICAL: silent resolution drift in `animate_ltx.py`

`snap_size()` rounds to a 64-pixel quantum. That was a constraint of the
**previous** tool (`mlx-video`) and does not apply to `ltx-2-mlx`.

Verified in the tool's source
(`~/.cache/nursery-tools/ltx-2-mlx/packages/ltx-core-mlx/src/ltx_core_mlx/components/patchifiers.py:15,127-180`):
`SPATIAL_COMPRESSION = 32`; the real modulus is **32 for single-stage, 64 only
under `--two-stage`**, and the tool's own `snap_output_dimensions()` is a pure
floor that never rounds up.

Consequence: `config.yaml` sets `height: 480`, `snap_size` emits **512**, and
the pipeline renders 704x512 at aspect 1.375 instead of the trained 704x480 at
1.467. Nothing warns. The config says one thing and the mp4 is another.

Fix: floor to 32 (64 when `two_stage`), or delete the snap and let the tool
handle it. Then amend the now-wrong advice in `task-6-brief.md:7` and
`docs/tool-help/ltx-2-mlx.txt:253-255`, and the self-refuting docstring at
`animate_ltx.py:22-24`.
`tests/test_animate_ltx.py:114` currently asserts `--height == "512"` and locks
the bug in as expected behaviour — flip it to `"480"`.

### 2. Open review findings on the lyrics provider

Findings 2 and 4 are fixed (`7616133`). Still open, all against
`nursery/providers/lyrics_qwen.py` unless noted:

1. **`:190`** — `entry.get("text", "")` assumes every `lines` entry is a dict.
   A model returning `"lines": ["twinkle twinkle...", ...]` raises an uncaught
   `AttributeError` out of `generate`. Treat a non-dict entry as a parse
   violation so it retries instead of crashing.
3. **`:191`** — `target_lines` is never reconciled with `len(scheme)`, and
   `check` takes its line count from `scheme`. `target_lines=8` with
   `scheme="AABB"` rejects all 8 lines every attempt, burning every retry on
   multi-minute model runs. Guard with a `ValueError` up front.
5. **`:181`** — `seed` is constant across retries. Identical violations produce
   a byte-identical prompt, and `mlx_lm --seed` is deterministic, so the retry
   is a guaranteed-identical wasted generation. Use `seed + attempt`.
6. **`nursery/config.py`** — nothing expands `~` if a user writes
   `song.python: ~/.cache/...` in their own config, which `config.yaml`'s
   comments invite. Add a `field_validator` applying `Path(v).expanduser()` to
   `song.python`, `song.repo_dir`, `align.bin`, `animate.bin`.
7. **`tests/test_lyrics_qwen.py:64`** — asserts only `--model`, so a regression
   to `--temperature` (which does not exist in mlx-lm) would pass.

### 3. Important: cast text is documented but not implemented

`image_flux2.py:144-160` — `generate()` never injects cast text; it delegates
to a caller that does not exist yet. The brief calls textual cast descriptions
the *primary* character-consistency mechanism, and `--image` only a low-strength
nudge, because `mflux --image` is single-shot img2img and NOT an identity
adapter. Right now nothing implements the primary mechanism. Make `_cast_text`
(`:62`) public, or add `cast: list[dict] | None = None` to `generate()`.

### 4. Dead config field

`SongConfig.variant` is accepted but unused. ACE-Step's real DiT selector is the
TOML key `config_path`, which takes a bare `SUBMODEL_REGISTRY` name, not the
`"org/name"` HF repo id `variant` holds. The plan forbids dead config keys —
either wire the translation or drop the field.

### 5. Task 4 was never reviewed

`b4b6425` (song + align providers) has no review pass. Everything else does.

### 6. Minors parked in the ledger

`tests/test_animate_ltx.py:130` has a vacuous assertion (`"--an" not in cmd`
tests an ffmpeg flag against a different tool); `tests/test_image_flux2.py`
uses bare list membership that will not catch a value bound to the wrong flag;
`config.yaml:60,65` pairs `two_stage: true` with the q4 model, but
`--two-stage` help says it requires q8.

## Tasks 7 + 8 — not started

Briefs are written and ready:
`.superpowers/sdd/2026-08-12-video-audio-pipeline-plan/task-7-brief.md` and
`task-8-brief.md`.

Task 7 wires the providers into `nursery/build.py`, updates
`nursery/stages/assemble.py`, adds manifest fields, trims `STAGE_ORDER` to only
registered stages (drop `publish`/`metadata`), and deletes
`nursery/audio/melody.py`, `nursery/audio/mixer.py`,
`nursery/providers/tts_macos.py`, their tests, and the `melody`/`tempo_bpm`
catalog keys.

**Do not lose this:** LTX clips always carry baked-in audio and there is no flag
to disable it. Assemble must pass `-an` on the clip inputs or the song plays
under ~20 fragments of unrelated LTX audio.

Task 8 adds `nursery smoke` — each stage once, shortest catalog entry, 2-scene
cap, per-stage wall time, continuing past failures with a clear marker — plus a
README.

## Known behaviour, not bugs

- `rhyme_key` is spelling-based, not phonetic. `star`/`are` still do not match
  despite rhyming, so some valid lyrics cost a retry. False *accepts* were the
  worse half and are fixed; this residual false *reject* is acceptable. A
  pronunciation dictionary would close it.
- Nothing has been run against real weights yet. Every command line is grounded
  in captured `--help` output and source reads, but no model has actually
  generated anything. `nursery smoke` (Task 8) is what proves it.

## Method

Superpowers subagent-driven development: Sonnet implements, Opus reviews, one
brief per task from the plan. Parallel dispatch worked because providers take
config *values*, never a `Config` object, so they do not import the file being
rewritten. Each agent got an explicit file-ownership list and was forbidden from
`git add -A`.
