# Audio tools audit, 2026-10-16

Scope: source-code audit at origin/main 64fa4a2, plus additive lyricsync integration. No existing tools were deleted. This distinguishes a useful editing cue from a proven semantic interpretation. A measurement that appears only in the song report is still available to the director; lack of an automated renderer consumer does not make it useless.

## Verdict

Keep the core song measurements and the two specialized separators. Add reference-conditioned forced alignment alongside raw recognition. The highest-value simplification would be making expensive texture tagging opt-in, not removing beats/stems/turns. Avoid adding another section/mood model until a concrete editing failure shows that the present measurements cannot support the decision.

## Inventory and data flow

| Component | What it actually produces / current consumer | Cost / limitations | Verdict |
|---|---|---|---|
| analyze_edit.py + librosa beat/onset pass | Tempo, alternative tempo warning, beat times, strong onset times. Used in report, JSON, sync comparisons; frames.py reads beats. | Beat grid can be half/double tempo or drift; onset is a transient, not necessarily a meaningful accent. | Keep. Check the intended cue by ear/frames; don't score edits by nearest beat. |
| RMS / low-band loudness | 0.25 s mix and low-band dBFS windows, plots/report; music.py also uses envelope changes to build turns. | Envelope resolution is not sample-accurate sync. Not perceived loudness or emotion. | Keep. Shared primitive; don't add duplicate energy-curve extraction. |
| stems.py, htdemucs | Content-addressed 4-stem mono FLAC cache at 22050 Hz. Feeds activity, turns, bass/voice features, Whisper, and now lyricsync. | First-use model download and separation; bleed affects downstream estimates. | Keep as the ONE song stem cache. New alignment reuses it. |
| separate.py, BS-RoFormer | Voice-only stereo-capable soundtrack extraction, used for renderer dialogue layers. | Separate checkpoint and normalization/cropping contracts. Dialogue vs song stem objectives differ. | Keep. Not interchangeable with htdemucs merely because both output vocals. Share low-level decode code only if a maintenance problem warrants it. |
| music.activity / loop_breaks / turns / drops | Stem presence/runs, entries/exits, loop changes, grouped structural turns, ranked drops. JSON and report; drops additionally feed frames.py and section logic. | Threshold/debounce/beat snapping can hide brief events. A ranked drop is a candidate event, not the emotional climax. | Keep. These answer “what changed?” more directly than a generic mood label. |
| music.stops | Whole-band stops/gaps, feeding event boundaries and report. | Quiet/processed recordings and vocal-only sections can confuse thresholds. | Keep as separate from a stem's exit. |
| _novelty / boundaries / build_sections | Spectral novelty candidates, snapped against stem/stop events, then audio/video section groupings. Reports/JSON, not renderer instructions. | No supervised verse/chorus meaning; video sections also depend on shots and recognized phrase grouping. | Keep with current descriptive labels. Do not promise semantic section recognition. |
| music.bass_roots | Estimated per-bar bass roots in active bass spans, harmony report/JSON. | Bass root is not a full chord or harmonic analysis; separator errors and beat drift propagate. | Keep as optional planning evidence. “Bass-root changes” in high-level copy is stronger than the data supports; prefer “bass-root changes.” |
| voice_sections / voice_shifts | Vocal pitch/register/level summaries and shifts by section; report/JSON. | Pitch tracking under layered, processed or unvoiced vocals is uncertain. Doesn't infer character POV, emotion or singer identity. | Keep but treat as review hints, not automatic mood or POV. |
| music.texture / Essentia TensorFlow | Whole-track key strength, instrument/genre tags; report/JSON. Does NOT provide timed mood changes. Current analyze_edit.py runs it unless --no-texture. | Optional compiled runtime plus tagger weights, substantially broader dependency surface than key alone. A top tag isn't stem activity. | Best candidate to demote to opt-in or split key from taggers. Audit only; behavior unchanged here. |
| analyze_edit.transcribe_words / faster-whisper | Raw vocal-stem recognized text, start/end and recognition probabilities; phrase segmentation, report, timeline.json, frames.py. | Hallucinations/omissions on singing. Recognition confidence is not timestamp accuracy. | Keep for unknown audio/dialogue and mismatch discovery. Not the authoritative sung reference or final timing source. |
| index.py dialogue transcription | Per-source dialogue timings/text used in show indexes, search/story workflows. Uses VAD unless disabled. | VAD can omit lines under score. This is a different job/input from song alignment. | Keep; shared Whisper wrapper is a possible small maintenance merge, not an output merge. |
| lyrics.py reference reader | Published/corrected text with provenance and optional unverified LRC times. Consumed by analysis reports and frames.py. | A reference does not establish what was sung in a clip. No alignment by itself. | Keep as canonical reference parser; lyricsync reuses it. |
| lyricsync.py, added | One global forced CTC path through ordered English reference; word index/line identity/times/CTC scores, uncertain-word audit, SRT/ASS and optional caption MP4. frames.py --alignment overlays words separately. | Needs correct sung sequence, matching torch/torchaudio and model; no simultaneous vocal alignment, no timing ground truth, no general mashup reconstruction. | Keep as the reference-timing path. No score-based word deletion or per-line windows. |
| LRCLIB / Shazam in lyricsync | Exact identity/duration-guarded lyric lookup, separately opted-in fingerprint identification candidates. | Network services can return wrong versions. Fingerprint lookup leaves the machine. | Keep explicit, fail conservatively. No auto-accepting fuzzy search or mashup winner. |
| render.py audio shaping | Song pieces/speed, gain/filter automation, compressor/EQ, ducking, dialogue layers, echo/reverb tails; consumes EDL, NOT an inferred musical meaning. | ffmpeg feature/version compatibility; renderer listening still required. | Keep. This produces the edit's sound; unrelated to deleting analysis features. |

## Overlap and useless-output assessment

There is overlap in decoding, level calculation and transcription setup, but the outputs serve distinct consumers. Consolidating every audio operation would risk conflating clocks, stereo rendering with mono analysis, and recognition confidence with CTC fit. The new code deliberately reuses the lyric parser and song-stem cache while leaving raw recognition untouched.

Only beats, drops and words are read into the cut-pair strip today. Stem activity, turns, roots, vocal shifts and tags reach the director through report.md/timeline.json. They aren't automated renderer inputs. If an agent ignores those reports, the remedy is to inspect whether the information changes its shot/cue choices, not to infer that every unrendered field is dead code. No repository-wide runtime usage study or human timing accuracy benchmark was performed.

The previous lyricsync prototype's automatic mashup anchor selection is genuinely not ready for production: its delivered mashup retained only 9 selected words. It was NOT imported. Nor were batch hardcoded song names, previous result archives, or a parallel Demucs pipeline imported. Supply the actual sung sequence for edited material.

## Validation and known blockers

New tests exercise actual torchaudio forced_align on synthetic emissions, repeated words, low scores, final-frame end timing, original-clock inference concatenation including a tiny tail, exact-reference retrieval guards, and a real ffmpeg SRT/ASS-caption-video smoke test. frames.py's integration test checks source identity, timeline conversion, retained low-score words and distinct CTC labels without overwriting the raw map.

Strict mypy passes all 15 configured source files. compileall and git diff --check pass. Targeted lyricsync tests pass 6/6; frames tests pass 17/17.

The complete initial suite reached 224 tests but exhausted the sandbox filesystem during the existing echo-tail render test. Rerunning all tests EXCEPT that one gives 224 tests, 2 failures, 2 errors, 7 skipped. A clean archive of unchanged 64fa4a2 with the identical exclusion gives 217 tests with exactly the same 2 failures, 2 errors, 7 skips. The 7 new tests add no regression on this run. The pre-existing failures concern a 21 ms embedded-subtitle offset in test_video; the errors are the installed ffmpeg lacking afir's irnorm option in reverb tests. The echo path was not successfully validated on this host and should be investigated with bounded ffmpeg output before rerunning it. No unrelated renderer/index changes were made.

No new full-song inference/separation or paid API experiment was run as part of this audit. Existing v3 song results motivated the implementation but do not establish current integration timing accuracy. Synthetic word preservation is not an accuracy claim.

## Approved follow-up implemented

Essentia key/tags now require --texture and its optional requirements-texture.txt. --lyrics now aligns the exact sung reference and embeds timed lines/words/uncertainty in timeline.json, with --alignment for reuse and --lyric-video for explicit caption rendering. Planning cues are grouped in the same file; downbeats remain explicitly unestimated. Voice shifts are labeled review hints. Existing audio.harmony JSON remains compatible.
