# Video + Audio Generation Pipeline

Date: 2026-08-12
Status: design approved, not implemented

Replaces the still-image + Ken Burns pipeline with generated video clips, and
replaces the synthesized melody + robotic `say` narration with a generated sung
song. Adds a local text model for lyrics and character design.

## Pipeline

```
  USER          nursery build baa-baa-rainbow-sheep
                            |
  1 WRITE       Qwen3.5-4B (4-bit MLX)  ~2.5 GB  Apache-2.0
                writes : lyrics, per-scene description, CHARACTER DESIGN, title, style
                code validates: syllable count, rhyme scheme, schema, safe vocab
                            |  retry with the specific violation on failure
                            v
  2 SING        ACE-Step 1.5 (MLX)  MIT
                lyrics + style -> ONE continuous song, sung vocals + instrumental
                *** replaces melody.py AND macOS `say` ***
                            |
                            v
  3 ALIGN       whisperx-mlx  (or ACE-Step's own lyric timestamps)
                recovers per-line start/end from the sung track
                *** this sets every scene's start and end ***
                            |
                            v
  4 CAST SHEET  FLUX.2 Klein 4B (MLX)  Apache-2.0
                ONE canonical image of the characters, from Qwen's design
                *** the consistency anchor for the whole video ***
                            |
                            v
  5 STILLS      FLUX.2 Klein 4B, cast sheet passed as reference
                one illustration per scene
                prompt = scene description + style,  reference = cast sheet
                            |
                            v
  6 ANIMATE     LTX-2.3 distilled (int4 MLX)  ~12 GB
                image-to-video, one clip per scene
                inputs: still + scene description + duration
                VIDEO STREAM ONLY - LTX audio is discarded, the song owns sound
                SSIM gate: reject -> scene falls back to Ken Burns pan
                            |
                            v
  7 ASSEMBLE    ffmpeg
                xfade clips over the song, captions burned in, thumbnail
                            |
                            v
  8 PUBLISH     YouTube Data API   (declared in STAGE_ORDER, not built)
                            |
                            v
                        final.mp4 -> YouTube
```

## Models

| Stage | Model | Size | Runner | License |
|---|---|---|---|---|
| Lyrics, scenes, cast | `mlx-community/Qwen3.5-4B-MLX-4bit` | ~2.5 GB | `mlx_lm.generate` | Apache-2.0 |
| Song (vocals + music) | ACE-Step 1.5, XL turbo | ~12-20 GB | `acestep` | MIT |
| Alignment | whisperx-mlx | ~1 GB | `whisperx-mlx` | BSD/MIT |
| Cast sheet + stills | FLUX.2 Klein 4B | ~4 GB | `mflux-generate-flux2` | Apache-2.0 |
| Video | LTX-2.3 distilled, int4 | ~12 GB | `ltx-2-mlx generate` | LTX-2.x Community (<$10M ARR) |

Every model is MLX-native and driven by shelling out to a CLI, matching the
existing `mflux-generate` / `ffmpeg` pattern. LTX weights are the only
non-permissive license in the stack.

## Key decisions

**A song model replaces both the melody and the narration.** `melody.py`, the
catalog note sequences, and `MacTTSProvider` all go away. ACE-Step takes the
lyrics and produces one continuous track with sung vocals and instrumental.
This removes the robotic `say` voice, and because the whole video shares one
song, it also removes the risk of 20 independently generated LTX audio beds
with no common key or tempo.

**LTX audio is discarded.** LTX still generates it - the pass is joint - but the
song owns the soundtrack. Optionally keep the LTX track mixed very low as foley;
default is off.

**Timing comes from alignment.** The song is generated as one track, so
per-line boundaries have to be recovered from it rather than decided up front.
This finally gives the `align` stage in `STAGE_ORDER` a real job. ACE-Step
advertises automatic lyric timestamps; whisperx-mlx is the fallback and the
verification path.

**Consistency is anchored by a cast sheet, not by seeds.** Qwen writes the
character design; FLUX.2 renders it once; every scene still is generated with
that sheet as a reference image. This replaces the current `seed=1000+i` plus
style-string approach, which has no actual mechanism holding a character
together across scenes. It supplies the `cast` stage declared in `STAGE_ORDER`
and the `cast.yaml` referenced in `animate_ltx.py`, neither of which existed.

**FLUX.1-schnell is replaced by FLUX.2 Klein 4B.** Same Apache-2.0 licensing,
2026 model, and reference-image conditioning is what makes the cast sheet work.
Klein 4B is the only FLUX.2 checkpoint that is not Non-Commercial: Klein 9B and
dev 32B allow commercial *outputs* but require a paid license to self-host.

**Meter is enforced in code, never by the model.** LLMs count syllables badly at
any size because tokenization hides the information. Code sets the syllable
target and rhyme scheme, the model drafts, a validator rejects and retries with
the specific violation. This is what makes 4B sufficient.

**One heavy model resident at a time.** Unified memory means Qwen, ACE-Step,
FLUX.2 and LTX draw from the same 64 GB as macOS. Stages run strictly in
sequence and release before the next loads.

## Failure behaviour

Per-scene, independent, already implemented in `nursery/stages/animate.py`:

- Clip render throws, or fails the SSIM gate -> `animation = "kenburns"`, that
  scene pans its still instead.
- `assemble` picks `ClipSource` or `StillSource` per scene; both normalise to
  `width x height @ fps`, so the xfade chain is identical either way.
- One bad scene costs one scene, never the video.

The SSIM gate checks `SSIM(clip_frame_0, source_still)` (did LTX honour the
reference image) and `SSIM(clip_frame_0, clip_frame_last)` (is there motion, and
did it stay on model).

## Hardware

MacBook Pro M4 Max, 40-core GPU, 64 GB unified, 454 GB free, macOS 26.5.2.

Roughly 2.5 min per scene for LTX on this class of machine, so a 20-line rhyme
is about 50 minutes, dominated entirely by stage 6. Stages 1-5 run once each,
not per scene.

## Not in scope

- LTX-2.5. Released 2026-08-11, CUDA/ComfyUI only, no MLX port. If one lands,
  the swap is `_command()` in the provider; the stage, gate, and fallback are
  unaffected.
- Lip sync. No character is shown singing, so vocals need not match mouths.

## Open risks

1. **i2v may ignore the reference image.** Reported against a sibling MLX stack
   and unresolved. If `SSIM(first, still)` comes back near 0.2 instead of 0.55+,
   LTX is silently running text-to-video, and the cast sheet buys nothing past
   the first frame.
2. **FLUX.2 Klein 4B reference conditioning is unverified.** The multi-reference
   character-consistency claims are documented for FLUX.2 [dev]. Whether the
   distilled Klein 4B exposes the same reference-image path through `mflux` must
   be confirmed before the cast sheet design is load-bearing. Fallback is
   FLUX.1 Kontext, which does but is heavier.
3. **ACE-Step speed on Apple Silicon is unmeasured.** Published figures are CUDA
   (~90s on an RTX 3060). No M-series numbers found.
4. **ACE-Step lyric intelligibility is unmeasured.** The entire point of the
   swap is that a child can hear the words. Sung output must be checked against
   the source lyrics, not assumed.
5. **Alignment accuracy on sung audio.** Forced aligners are tuned for speech;
   singing stretches vowels and shifts stress. If whisperx-mlx drifts on sung
   vocals, scene boundaries drift with it.
6. **Unmeasured LTX wall time.** The 2.5 min/scene figure is from a 128 GB M4
   Max, not this 64 GB machine.

## First step

A standalone script, outside `nursery/`, run in this order because each step
gates the next:

1. ACE-Step: generate one song from existing catalog lyrics. Listen. Are the
   words intelligible? How long did it take? (risks 3, 4)
2. whisperx-mlx: align that song. Do the line boundaries land correctly on sung
   audio? (risk 5)
3. FLUX.2 Klein 4B: render a cast sheet, then a scene still using it as a
   reference. Does the character carry over? (risk 2)
4. LTX: animate that still. Report wall time, `SSIM(first, still)`,
   `SSIM(first, last)`. (risks 1, 6)

Nothing under `nursery/` changes until all four pass.
