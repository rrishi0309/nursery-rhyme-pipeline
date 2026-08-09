# Nursery Rhyme Video Generation Pipeline — Design

**Date:** 2026-08-08
**Status:** Approved (design), model selections pending verification (see §10)

## 1. Purpose

Generate children's nursery-rhyme videos end-to-end on a single Apple Silicon
machine, at a cost of zero per video, using only self-hosted open-weight models.
Output is 1080p MP4 with sung audio, illustrated scenes, and karaoke captions,
plus the metadata needed to publish to YouTube.

The channel goal is volume with monetization, so throughput and marginal cost
dominate. The binding constraint is not compute — it is operator review time.

## 2. Constraints

| Constraint | Value |
|---|---|
| Hardware | MacBook Pro, Apple M4 Max, 64 GB unified memory, ~513 GB free |
| Marginal cost per video | $0 — no paid APIs anywhere in the pipeline |
| Model licensing | Must permit commercial use of generated output |
| Network | Required only to pull model weights, and to upload |
| Target runtime | Under ~20 min per video, unattended |
| Output | 1920x1080, H.264, AAC, 60–120 s duration |

### 2.1 Non-goals for v1

- Image-to-video / animated motion models. Wan, HunyuanVideo, CogVideoX and
  LTX are CUDA-first and impractically slow on MPS. Motion comes from Ken Burns
  pan/zoom in ffmpeg instead. Revisit only if a viable MPS path appears.
- Automated upload. The publish stage prepares an upload bundle; a human runs
  the upload. Automating it is a later, separate decision.
- Multi-language output.
- A web UI. This is a CLI.

## 3. Content strategy

Two rhyme sources share one pipeline and one manifest schema:

**Classics.** Traditional public-domain rhymes from a curated `classics.yaml`
catalog — lyrics only. These carry real search demand because parents search
them by name. Only the *text* is public domain; every melody and arrangement in
this pipeline is newly generated, never an imitation of a specific existing
recording.

**Originals.** Generated from a structured seed sampled from `seeds.yaml`:
`character x setting x activity x lesson x meter template`. Constraining the LLM
this way is what prevents the uniform mush that "write me a random nursery
rhyme" produces. Originals carry no copyright exposure and give an unbounded
catalog, at the cost of near-zero cold search demand.

The mix ratio is a config value, defaulting to 50/50.

## 4. Architecture

Stage-based pipeline. Each video is a directory; a `manifest.json` inside it is
the single source of truth. Stages are independent modules that read the
manifest, do exactly one job, write artifacts, and update the manifest.

```
nursery/
  pyproject.toml
  nursery/
    cli.py              # nursery run | stage | list | review
    manifest.py         # pydantic schema, load/save/validate, atomic writes
    orchestrator.py     # stage sequencing, resume, failure handling
    config.py           # typed config, loaded from config.yaml + env
    stages/
      seed.py  script.py  audio.py  align.py
      cast.py  images.py  assemble.py  metadata.py  publish.py
    providers/
      llm.py            # Ollama or MLX backend
      song.py           # local text-to-song backend
      tts.py            # fallback narration backend
      image.py          # ComfyUI or diffusers backend
      align.py          # forced-alignment backend
    catalog/
      classics.yaml     # public-domain rhyme canon
      seeds.yaml        # sampler vocabulary for originals
      cast.yaml         # recurring characters + reference image paths
  assets/
    cast/               # cached character reference images
    fonts/  music/  branding/
  out/<video-id>/
    manifest.json
    script.json
    audio/{vocal,music,mix}.wav
    align/words.json
    images/scene_00.png ...
    video/{final.mp4,thumbnail.png}
    logs/<stage>.log
  docs/superpowers/specs/
```

`<video-id>` is `YYYY-MM-DD-<slug>-<nn>`, e.g. `2026-08-08-twinkle-twinkle-01`.

### 4.1 Stage contract

Every stage implements the same interface:

```python
class Stage(Protocol):
    name: str
    def is_satisfied(self, m: Manifest) -> bool: ...
    def run(self, m: Manifest, cfg: Config) -> Manifest: ...
```

`is_satisfied` checks that declared output files exist and the manifest records
a successful prior run with a matching input hash. The orchestrator walks the
stage list in order, skips satisfied stages, and halts on the first failure with
the manifest written and intact. This makes every stage independently
re-runnable: `nursery run <id> --from images` re-renders images and everything
downstream, touching nothing before it.

Input hashing matters. If the script changes, the audio stage must not report
itself satisfied by a stale WAV. Each stage records a hash of the manifest
fields it consumed; a mismatch invalidates it.

### 4.2 Provider abstraction

Model backends sit behind narrow interfaces in `providers/`. `image.py` exposes
`generate(prompt, refs, seed) -> Path` and has two implementations: a ComfyUI
HTTP backend and a direct diffusers/MLX backend. Stages never import a model
library directly. This is what makes swapping SDXL for a newer model a one-file
change, and it is what lets the entire pipeline be tested without a GPU.

## 5. Data flow

```
seed ──► script ──► audio ──► align ──► images ──► assemble ──► metadata ──► publish
  │        │          │         │          ▲          │
  │        │          │         │          │          │
  │        └─ scenes ─┼─────────┼──────────┘          │
  │                   │         │                     │
  └─ classic title    └─ sung   └─ word timings ──────┘
     or sampled seed     audio     drive cuts AND captions
```

The non-obvious property: **scene durations are derived, not chosen.** The
script stage decides how many scenes exist and what is in each. The audio stage
renders the song. Alignment then pins each lyric line to real timestamps, and
those become the cut points. Audio must precede images for this reason; it is
what makes visuals land on the beat rather than drift.

### 5.1 Stages

**`seed`** — Picks a classic title from `classics.yaml` (excluding already-used
titles) or samples an original seed from `seeds.yaml`. Writes `manifest.seed`.
Deterministic given an RNG seed, so runs are reproducible.

**`script`** — Local LLM produces structured JSON: title, stanzas, per-scene
visual descriptions, a global art-style string, and the cast referenced. For
classics the lyrics come verbatim from the catalog and the LLM only writes
scenes. Output is validated against a pydantic schema and retried on parse
failure, up to 3 attempts, before failing the stage.

**Review gate.** `nursery run` stops here by default. Lyrics are the cheapest
artifact to inspect and the most expensive to get wrong. `nursery review <id>`
prints the script and accepts approve / regenerate / edit. `--no-review` skips
it for batch runs.

**`audio`** — Renders a sung track from the lyrics via the local song provider,
producing `vocal.wav`, `music.wav` where the model separates them, and a
normalized `mix.wav`. Loudness-normalized to about -14 LUFS for YouTube. If the
song provider fails or is disabled, falls back to the TTS provider over a
backing loop from `assets/music/`, and records the degradation in the manifest.

**`align`** — Forced alignment of the *known* lyric text against the rendered
audio, yielding word-level timestamps. Forced alignment is used rather than open
transcription because the exact text is already known; only timing is unknown,
which is a far more reliable problem — especially for sung audio, where open
transcription degrades badly. Line-level boundaries become scene cut points.

**`cast`** — Resolves each referenced character to a reference image. Known
characters load cached references from `assets/cast/`; new ones are generated
once and cached. This is the backbone of visual consistency: the same bunny in
video 40 uses the same reference as video 1.

**`images`** — One image per scene, conditioned on the art-style string plus
character reference images. Fixed seeds recorded in the manifest so a render is
reproducible. Renders are parallel-safe but serialized by default to avoid
memory pressure.

**`assemble`** — ffmpeg. Ken Burns pan/zoom per scene, cut to alignment
boundaries, crossfade transitions, karaoke captions burned in from word
timings, optional intro/outro. Produces `final.mp4` and a thumbnail. This stage
is pure and deterministic: given the same inputs it always produces the same
video, so it can be iterated on freely against cached artifacts.

**`metadata`** — LLM generates title, description, and tags. Always sets
`made_for_kids: true` (see §8). Writes `upload.json`.

**`publish`** — Assembles an upload bundle and prints a checklist. Does not
upload in v1.

## 6. Manifest schema

Pydantic models, versioned with a `schema_version` field. Sketch:

```python
class Manifest(BaseModel):
    schema_version: int
    video_id: str
    created_at: datetime
    source: Literal["classic", "original"]
    seed: SeedSpec
    script: ScriptSpec | None
    scenes: list[Scene]          # text, visual_prompt, characters, image_path,
                                 # start_s, end_s  (timings filled by align)
    audio: AudioSpec | None      # paths, duration, provider, degraded: bool
    alignment: AlignSpec | None
    video: VideoSpec | None
    metadata: MetadataSpec | None
    stages: dict[str, StageRecord]   # status, started/finished, input_hash, error
```

Writes are atomic — serialize to a temp file, then rename — so an interrupted
run never leaves a corrupt manifest.

## 7. Error handling

- **Stage failure** halts the run for that video, records the error and
  traceback in `manifest.stages[name]`, writes `logs/<stage>.log`, and moves on
  to the next video in a batch. One bad video never stalls a queue.
- **Transient model failures** (OOM, backend timeout) retry up to 3 times with
  backoff. Persistent failures fail the stage.
- **Schema validation failures** from LLM output retry with the validation error
  appended to the prompt, then fail.
- **Degradation is recorded, never silent.** If audio falls back from song to
  TTS, the manifest says so and `nursery list` flags it.
- **Disk pressure**: at roughly 200–500 MB per video, `nursery clean` prunes
  intermediates for completed videos, keeping manifest, final MP4, and
  thumbnail.

## 8. YouTube and legal compliance

These are requirements, not advice, and the pipeline enforces them:

- **`made_for_kids: true` is set unconditionally** on every upload bundle.
  Children's content is COPPA-regulated. This disables personalized ads,
  comments, and end screens, and structurally lowers RPM. It is not optional.
- **Melodies are always newly generated.** Classic rhyme *lyrics* are public
  domain; specific modern arrangements are not. Prompts must never name or
  imitate an existing channel's arrangement.
- **Model licenses are verified before use.** Open weights do not imply
  commercial rights. Known landmines: FLUX.1-dev is non-commercial while
  FLUX.1-schnell is Apache-2.0; XTTS-v2 is non-commercial. Every model in §10
  carries a verified license, and `docs/LICENSES.md` records the evidence.
- **Volume alone is a monetization risk.** YouTube policy targets mass-produced
  repetitive content. The mitigations built into this design are a recurring
  cast, real sung audio, and the human review gate. Noted so the tradeoff is
  explicit; the operator decides the throughput.

## 9. Testing

The pipeline must be fully testable without loading a single model.

- **Fake providers.** Each provider interface gets a fake returning fixture
  artifacts — a 3-second sine WAV, a solid-color PNG, canned JSON. The full
  pipeline runs end-to-end in CI in seconds against fakes. This is the single
  most important testing decision here; without it every test needs 15 minutes
  and 40 GB of RAM.
- **Unit tests per stage** against fixture manifests, covering the satisfied /
  unsatisfied / stale-input-hash paths.
- **Golden-file tests for `assemble`**, asserting on the ffmpeg filter graph
  string rather than on video bytes, which are not stable across ffmpeg builds.
- **Schema round-trip tests** for the manifest, including forward-compatible
  loading of older `schema_version` values.
- **One manual smoke target**, `make smoke`, that runs a real single video with
  real models. Not part of CI.

Development follows TDD per the project's skills: failing test, then
implementation.

## 10. Model selections

Pending verification by research pass. Each entry must record: model, exact
license, commercial-use permission, Apple Silicon viability, measured speed on
M4 Max, and install method. Anything unverified is not used.

| Stage | Candidate | License | Commercial | MPS | Status |
|---|---|---|---|---|---|
| LLM | TBD | | | | pending |
| Song | TBD | | | | pending |
| TTS fallback | TBD | | | | pending |
| Image | TBD | | | | pending |
| Consistency | TBD | | | | pending |
| Alignment | TBD | | | | pending |

## 11. Build order

1. Manifest schema, config, orchestrator, stage contract, fake providers.
2. `assemble` — the deterministic core. Build it first against fixture images
   and audio, because it is pure, fast to iterate, and defines what every
   upstream stage must produce.
3. `seed` and `script` with the real LLM provider, plus catalogs.
4. `audio` and `align`.
5. `cast` and `images`.
6. `metadata` and `publish`.
7. End-to-end smoke run, then batch mode.

Building `assemble` second, before any generative stage, is deliberate: it turns
the vague question "what should the image stage output?" into a contract that
already exists.
