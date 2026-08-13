# Implementation Plan: Video + Audio Nursery Rhyme Pipeline

Spec: `docs/superpowers/specs/2026-08-12-video-audio-pipeline-design.md`
Target: `nursery build <slug>` produces `out/<video_id>/final.mp4`. No YouTube.

## Global Constraints

- **Shell-out pattern.** Every model runs via `subprocess.run` against a CLI,
  never an in-process Python API. Follow `nursery/providers/image_flux.py`.
- **Never invent CLI flags.** Task 1 captures real `--help` output to
  `docs/tool-help/`. Every later task reads those files and uses only flags
  that appear there. If a needed flag is absent, report DONE_WITH_CONCERNS
  naming it — do not guess.
- **Provider protocol.** Each provider is a class with a `name` attribute and
  one primary method. Constructor takes config values, never a `Config`.
- **Failure isolation.** Per-scene work catches per scene and degrades; it
  never aborts the video. Existing pattern: `nursery/stages/animate.py:75`.
- **Config is typed.** All settings are pydantic models in `nursery/config.py`.
  No magic numbers in providers.
- **Tests are light.** Unit-test pure logic (parsing, validation, timing math,
  command construction) with fakes. Do NOT write tests that download weights or
  invoke real models. Speed over coverage.
- **Python 3.14, `uv` for deps.** Line length 100, ruff clean.
- **Do not delete `nursery/audio/melody.py`, `mixer.py`, or
  `providers/tts_macos.py` until Task 7** — Task 7 owns their removal.

## Task 1: Install toolchain and capture CLI surfaces

Install the CLI packages (not weights) and record their real flag surfaces.
Everything downstream depends on this being accurate.

1. `uv add mlx-lm` — verify `uv run python -m mlx_lm.generate --help`
2. Verify FLUX.2 in existing mflux: `uv run mflux-generate-flux2 --help`
   (if that entry point does not exist, run `uv add --upgrade mflux` first,
   then re-check; also capture `mflux-generate --help`)
3. ACE-Step 1.5: clone `https://github.com/ace-step/ACE-Step-1.5` to
   `~/.cache/nursery-tools/ACE-Step-1.5`, follow its macOS install, capture the
   CLI help for its generate entry point
4. `uv add whisperx-mlx` (or clone `taavi223/whisperx-mlx` if not on PyPI) —
   capture help
5. ltx-2-mlx: clone `https://github.com/dgrauet/ltx-2-mlx` to
   `~/.cache/nursery-tools/ltx-2-mlx`, `uv sync --all-extras`, capture
   `ltx-2-mlx generate --help`

Write each captured `--help` verbatim to `docs/tool-help/<tool>.txt`.
Write `docs/tool-help/SUMMARY.md`: for each tool, the exact invocation string,
whether it installed cleanly, and any tool that FAILED to install with the error.

A tool that will not install is not a blocker for this task — record it and
continue. Report DONE_WITH_CONCERNS listing failures.

No production code in this task.

## Task 2: Config schema

Rewrite `nursery/config.py` for the new pipeline. Read
`docs/tool-help/SUMMARY.md` first — model repo names and flags come from there.

Add pydantic models:
- `LyricsConfig`: `model_repo` (default `mlx-community/Qwen3.5-4B-MLX-4bit`),
  `max_tokens`, `temperature`, `max_retries` (default 3)
- `SongConfig`: `variant`, `inference_steps`, `guidance_scale`, `audio_format`
  (`"wav"`), `duration` (-1.0 = auto), `backend` (`"mlx"`), plus `python`
  (`~/.cache/nursery-tools/ACE-Step-1.5/.venv/bin/python`) and `repo_dir`
  (`~/.cache/nursery-tools/ACE-Step-1.5`)
- `AlignConfig`: `model` (whisper size, default `"small"`), `backend`
  (default `"lightning"` — plain `mlx` may not emit word timestamps),
  `max_drift_s` (default 0.75), plus `bin` pointing at
  `~/.cache/nursery-tools/whisperx-mlx/.venv/bin/whisperx`
- `ImageConfig`: `model` (default `"flux2-klein-4b"`), `steps`, `width` 1344,
  `height` 768, `quantize` 4, `guidance` 3.5, `cast_strength` (float, default
  0.35 — the img2img strength used when conditioning a scene on the cast sheet)
- `AnimateConfig`: keep `min_source_ssim` 0.55, `min_motion_ssim` 0.35,
  `max_motion_ssim` 0.995. Set `width` 704, `height` 480 (ltx-2-mlx defaults —
  the old 1024x576 was for a different tool), `frame_rate` 24 (**mandatory,
  the tool has no default**; LTX-2.3 was trained at 24fps), `model_repo`
  default `dgrauet/ltx-2.3-mlx-q4`, `steps` 8. Drop `pipeline`, `cfg_scale`,
  `image_strength`. Add `two_stage: bool = False`, `low_ram: bool = True`,
  and `bin` pointing at `~/.cache/nursery-tools/ltx-2-mlx/.venv/bin/ltx-2-mlx`

Note `nursery/video/kenburns.py` and `assemble` still render at
`VideoConfig.width/height` (1920x1080); LTX clips are upscaled to match.

Update the `providers` dict to exactly:
`{"lyrics": "qwen", "song": "acestep", "align": "whisperx", "image": "flux2", "animate": "ltx"}`.
Every key must be read by its stage — no dead keys (the old config had five).

Update `tests/test_config.py` for the new shape.

## Task 3: Lyrics provider + meter validator

New `nursery/providers/lyrics_qwen.py` and `nursery/lyrics/validate.py`.

`validate.py` (pure, fully unit-tested — this is the load-bearing logic):
- `count_syllables(word: str) -> int` — vowel-group heuristic: lowercase, count
  vowel groups `[aeiouy]+`, subtract 1 for a silent trailing `e` when the word
  has >1 group, floor at 1
- `line_syllables(text: str) -> int` — sum over words, ignoring punctuation
- `rhyme_key(text: str) -> str` — last word lowercased, stripped of
  punctuation, from its last vowel group onward (`"star"` -> `"ar"`)
- `check(lines: list[str], target: int, tolerance: int, scheme: str)
  -> list[str]` — returns human-readable violations, empty when valid.
  `scheme` is like `"AABB"`; lines sharing a letter must share a `rhyme_key`.

`LyricsProvider.generate(slug_hint, target_lines, syllables, scheme, seed)`:
- Builds a prompt asking for strict JSON: `title`, `style`, `cast` (list of
  `{name, description}`), `lines` (list of `{text, visual}`)
- Shells out to `mlx_lm.generate` per `docs/tool-help/`
- Extracts the first balanced `{...}` block from stdout, parses it
- Runs `validate.check`; on violation, retries up to `max_retries` appending the
  violation text to the prompt
- Raises `LyricsError` when retries are exhausted

Tests: `validate.py` thoroughly with plain strings. For the provider, one test
with a monkeypatched `subprocess.run` returning canned JSON.

## Task 4: Song provider + alignment provider

New `nursery/providers/song_acestep.py` and `nursery/providers/align_whisperx.py`.
Read `docs/tool-help/` for both invocations.

`SongProvider.generate(lyrics: list[str], style: str, seed: int, out: Path) -> Path`
- **ACE-Step has NO generate flags — it is TOML-config-only.** Write a temp
  `.toml` with the keys SUMMARY.md lists (`save_dir`, `audio_format="wav"`,
  `caption` = style, `lyrics` = lines joined by newline, `duration`,
  `instrumental=false`, `task_type="text2music"`, `inference_steps`, `seed`,
  `guidance_scale`, `backend="mlx"`), then run
  `<song.python> cli.py --config <toml>` with `cwd=<song.repo_dir>`
- ACE-Step writes into `save_dir`; move/rename the produced file to `out`
- Raises `SongError` with the stderr tail on non-zero exit

`AlignProvider.align(audio: Path, lines: list[str], out_dir: Path) -> list[tuple[float, float]]`
- Runs `<align.bin> <audio> --backend <align.backend> --model <align.model>
  -o <out_dir> -f json --word_timestamps True`. Note SUMMARY.md's warning:
  `--word_timestamps` is documented as requiring a `lightning` backend, so
  plain `mlx` may return empty word spans — that is exactly what the fallback
  below is for.
- Maps recognised words back onto the supplied lyric lines in order, returning
  one `(start_s, end_s)` per line
- Sung vowels stretch and ASR drops words, so matching must be tolerant:
  normalise case/punctuation and walk both sequences forward, never requiring an
  exact match
- **Fallback, required:** if alignment fails, or any line's span is
  non-monotonic or zero-length, log a warning and return an even split of the
  audio duration across the lines. Never raise. Scene timing must always exist.

Tests: the word-to-line mapping and the fallback, both with canned JSON.
No real audio.

## Task 5: FLUX.2 image provider with cast sheet

New `nursery/providers/image_flux2.py`. Read `docs/tool-help/` for flags.

`Flux2ImageProvider`:
- `cast_sheet(cast: list[dict], style: str, seed: int, out: Path) -> Path` —
  renders one image of all characters together, prompt built from their names
  and descriptions plus the style
- `generate(prompt, reference, seed, out) -> Path` — one scene still.

**Known limitation, design around it:** `mflux-generate-flux2 --image PATH
[STRENGTH]` is a single non-repeatable **img2img** control, NOT an
identity-preserving multi-reference adapter. Passing the cast sheet at high
strength would copy its composition into every scene, which is wrong. So:
  - Always append the cast descriptions (name + description) to the scene
    prompt text. This is the primary consistency mechanism.
  - Additionally pass `--image <cast_sheet> <cfg.cast_strength>` (default
    0.35, low on purpose) only when `reference` is provided.
  - Use a stable per-video base seed so scenes stay stylistically coherent.

Expose `use_reference: bool = True` on the constructor so the img2img path
can be switched off wholesale if it turns out to hurt.
- Keep the existing `NEGATIVE` prompt string from `image_flux.py`

Leave `image_flux.py` and `image_sdxl.py` in place as fallbacks.

Tests: command construction with a monkeypatched `subprocess.run`, both the
reference and no-reference paths.

## Task 6: Rewrite the LTX animate provider for ltx-2-mlx

Rewrite `nursery/providers/animate_ltx.py` against `ltx-2-mlx generate`
(see `docs/tool-help/`). The old `mlx_video.models.ltx_2.generate` invocation
is gone.

Keep unchanged: `frames_for`, `snap_size`, `FRAME_QUANTUM`, `SIZE_QUANTUM`,
`MOTION_SUFFIX`, the `animate(image, prompt, duration_s, seed, out) -> Path`
signature, and the `LTXUnavailableError` behaviour on a missing module.

Change: `_command()` builds `<cfg.bin> generate` with `--prompt`, `--image
<still>` (PATH alone means frame 0 at strength 1.0), `--output`, `--frames`,
`--width`/`--height`, `--seed`, `--steps`, `--model`, and **`--frame-rate`
which is mandatory and has no tool default**, plus `--two-stage` and
`--low-ram` when configured.

**There is no audio-skip flag** — confirmed against the captured help. LTX
always bakes an audio track into the mp4. Do NOT invent one; assemble strips
it with `-an` (Task 7).

`nursery/stages/animate.py` keeps its SSIM gate and Ken Burns fallback
unchanged — only `build_provider` updates for the new config fields.

Tests: update `tests/` for the new command shape. Keep the existing
`frames_for` tests passing.

## Task 7: Orchestration, assemble, and cleanup

Rewrite `nursery/build.py` to the spec's stage order:

1. lyrics (Qwen) -> catalog entry with `cast`; write `out/<id>/lyrics.json`
2. song (ACE-Step) -> `out/<id>/audio/song.wav`
3. align (whisperx) -> per-line `(start_s, end_s)` -> scene boundaries
4. cast sheet (FLUX.2) -> `out/<id>/images/cast.png`
5. stills (FLUX.2 + cast reference) -> `out/<id>/images/scene_NN.png`
6. animate (LTX) -> clips, via the existing `animate_scenes`
7. assemble -> `out/<id>/final.mp4`

Also:
- `--from-catalog <slug>` keeps the existing YAML path working, skipping step 1
- `nursery/stages/assemble.py`: use `song.wav` as the audio track; drop any LTX
  clip audio (`-an` on clip inputs) so only the song is heard
- `nursery/manifest.py`: add `song_path`, `cast_sheet_path`, `lyrics_path`
- **Delete** `nursery/audio/melody.py`, `nursery/audio/mixer.py`,
  `nursery/providers/tts_macos.py`, `nursery/catalog/*.yaml` melody/tempo keys,
  and their tests (`tests/test_melody.py`, mixer tests)
- Register the new stages so `STAGE_ORDER` entries `lyrics`/`song`/`align`/
  `cast`/`images` resolve, or trim `STAGE_ORDER` to what is registered — no
  declared-but-missing stages
- Remove `publish`/`metadata` from `STAGE_ORDER` (out of scope)

Run the full suite. Delete tests made obsolete by removals rather than leaving
them failing.

## Task 8: End-to-end smoke command

Add `nursery smoke` to `nursery/cli.py`: runs each stage once on the shortest
catalog entry with a 2-scene cap, printing per-stage wall time and the output
path, and continuing past a failed stage with a clear marker.

This is the spec's spike, as a real command. It is how the user finds out which
models actually work on their machine.

Write `README.md`: what the pipeline does, the model table from the spec, the
install steps from `docs/tool-help/SUMMARY.md`, and how to run `nursery smoke`
then `nursery build`.
