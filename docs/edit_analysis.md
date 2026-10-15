# Edit analysis

How `scene_frames/analyze_edit.py` turns a short reference edit (15–40 s) into the measured timeline a breakdown is written from: the skeleton and the cut-by-cut table in `breakdowns/`.

## Usage

```sh
python scene_frames/analyze_edit.py VIDEO OUT_DIR [--caption-band 0.75:1] [--no-words]
```

Requires ffmpeg on PATH and `requirements.txt`. Word timings use faster-whisper `small.en` (about 460 MB, downloaded on first use). Audio and video stay local.

Writes to OUT_DIR:

- `report.md`: the skeleton (tempo, drop, vocal phrases, pacing per section) and one row per shot, flash or stretch of black. Each row gives in/out, length, how the shot was entered, and the offset to the nearest beat, strong onset and word edge.
- `timeline.json`: the same data for later stages: segments with frame numbers, in-shot events, dissolves, the audio map, word timings, phrases and sections.
- `frames.json`: per-frame luma, white share, contrast, histogram distance, pixel difference, structure and edge distance, and the frame's label.
- `sheets/shots_NN.jpg`: one keyframe per shot, labelled `S4 6.95-7.85s`.
- `sheets/captions_NN.jpg`: the caption band every 0.25 s at up to 1280 px wide, so burned-in captions can be read. Use `--caption-band TOP:BOTTOM` (fractions of the frame height) to move or tighten it, and `--caption-step` to change the interval.

No lyric or dialogue text is written anywhere. Whisper's words are dropped as soon as their timings are read.

## How it works

1. **Frames.** Every frame is decoded through ffmpeg's `fps` filter at 128 px on the long side, so frame *i* is at *i* / fps. Per frame the script records mean luma, the share of pixels above 200, the 98th-percentile luma, an RGB histogram distance and the mean pixel difference to the previous frame. It also records two brightness-normalised signatures: standardised half-size luma ("structure") and standardised gradient magnitude ("edges").
2. **Labels.** A frame is *white* when its mean luma is at least 140 and at least 25% of its pixels are above 200. It is *black* when 98% of its pixels are below 30 and the mean is below 10. A white run longer than 0.5 s is bright footage, not a flash. A run of white or black frames becomes a transition span. Any footage inside it shorter than the minimum shot (two frames) is treated as a strobe and stays in the span.
3. **Cuts.** Between adjacent footage frames, a cut needs both the structure distance and the edge distance to reach 0.45 (`--cut-threshold`). The distances are the smallest over the two frames on each side, so one flicker frame cannot fake a cut, and a mean pixel change under 3 never cuts. Across a flash or black span the default flips: it is a cut unless both distances fall below 0.35.
4. **Short pieces.** A piece shorter than two frames becomes a flash if it is at least 40 luma brighter than both neighbours. Otherwise it joins the incoming shot. When a flash or black span separates it from the incoming shot, it joins whichever side it resembles.
5. **In-shot events.** A flash or black span with the same shot on both sides is listed as an in-shot event, not a cut.
6. **Dissolves.** Inside each shot, windows of ±0.1–0.4 s whose ends differ like a cut are fitted as a linear blend of their two end frames. A window passes when the blend residual is at most 0.35 of the distance to the nearer end and the mix only moves one way. The dissolve is the union of the blended frames of the passing windows that overlap the best one. It belongs to the incoming shot, which starts on its first blended frame.
7. **Audio.** librosa gives the tempo and beat grid, and the onsets at or above the 90th percentile of onset strength. RMS and a 150 Hz low-pass are reported in dBFS per 0.25 s window. The drop is the largest loudness jump between the 0.25 s before and after a moment. The first 0.5 s and moments preceded by silence are skipped.
8. **Words.** faster-whisper runs with `word_timestamps=True` and VAD off. Words closer than 1 s form a vocal phrase.
9. **Report.** The edit is split into sections at phrase edges and the drop, each snapped to a shot boundary within 0.5 s. Every row is measured at its first frame; for a shot after a dissolve, that is the first blended frame. "On" means within 0.10 s. Events more than 1 s from a row are left out of the table but kept in `timeline.json`.

## Design choices

**Structure, not histograms, decides cuts.** Anime footage flickers inside shots: fireworks, lightning, explosion light, a face going from dark to lit. On the reference edit, flicker inside the fireworks-kiss shot scores a histogram distance of 0.66, as high as real cuts. Brightness-normalised structure stays above 0.86 correlation through that flicker and drops below 0.3 at cuts. The histogram distance and pixel difference are still recorded in `frames.json`, and the pixel difference gates out noise on static dark frames.

**Flashes and black are transitions, not shots.** They get their own rows so the table shows where each picture starts, and they never count towards shot totals or average shot length. A shot entered through a flash starts after the flash.

**`small.en` with VAD off.** On the reference edit, multilingual `small` misses the quiet spoken intro entirely, and Silero VAD drops every vocal that sits under music. `small.en` without VAD recovers all three voices. Seven of the eight word times quoted in the breakdown come out within 0.1 s; the start of the second half of the last line comes out 0.3 s early. Pass `--whisper-model small` (and optionally `--lang`) for edits that aren't in English.

**ffmpeg pipes, not OpenCV.** opencv-python and PyAV both bundle libavdevice, which clash when loaded in one process (faster-whisper imports PyAV). Decoding through ffmpeg keeps OpenCV out of this script.

## Against the hand-made breakdown

On `denreze-whitelight.mov` the script finds the breakdown's 19 shots. Every hard cut and flash it lists lands within one frame, as do the kiss's 2.85 s hold, the one cross-dissolve, 123 BPM and the drop at 4.0 s. Where they differ, the script follows the frames:

- The opening fade-up is reported as 0.65 s of black before shot 1.
- The bloom at 3.95 stays in the city shot. The white flash proper runs 4.00–4.10.
- The hands-against-fireballs shot cuts to near-black at 11.75. A dark city shot then fades up and dissolves into the silhouette from 12.20 to 13.00. The breakdown has one shot running to about 12.2.
- The pin blast is a single bright frame at 14.10, after one black frame. The café shot flickers in from 14.15, not 14.30.
- One-frame black cuts at 7.85 and 18.45 get their own rows.

## Caching

Picture analysis, the audio map and word timings are cached in `OUT_DIR/cache/` with the same signature scheme as `index.py`: source path, size and mtime, the relevant options, and an analysis version. Rerunning reuses whatever still matches. Sheets and reports are rebuilt every run.

## Tests

`tests/test_analyze_edit.py` checks the segmentation on synthetic frames on every run. With `denreze-whitelight.mov` at the repo root, or `ANALYZE_EDIT_VIDEO` pointing at it, it also checks the breakdown's ground truth within one frame: the hard cuts, the flashes, the kiss as one shot, the dissolve, the tempo and the drop.
