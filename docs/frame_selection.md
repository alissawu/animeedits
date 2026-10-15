# Frame selection

How `scene_frames/detect.py` turns a video into shots, and what it records about each one.

## Usage

```
python scene_frames/detect.py INPUT OUT_DIR [--faces live|anime] [--sample N] [--proxy WIDTH]
```

Per shot it writes three stills, named by shot number and frame time:

- `scene_0004_005.90s.jpg`: the best keyframe, for thumbnails and captioning.
- `scene_0004_first_005.45s.jpg` and `scene_0004_last_006.85s.jpg`: how the shot starts and ends, for match cuts.

Agents can rename stills with a label as long as the timestamp stays at the end, e.g. `04_fireworks_kiss_005.90s.jpg`.

`scenes.json` is a list with one entry per shot:

| Field | Meaning |
|---|---|
| `i`, `start`, `end` | Shot number and time range in seconds (`end` exclusive) |
| `best`, `file` | Keyframe time and filename |
| `first`, `last` | `{time, file}` of the edge stills |
| `flashes` | `[{time, frames}]`: bright flashes in the shot |
| `motion` | `{fps, curve, peaks, mean, max}`, see [Motion](#motion) |
| `look` | `{luma, hue, saturation, palette}`, see [Look](#look) |
| `face`, `face_checks` | Only with `--faces` |

## How it works

1. **Find shots.** PySceneDetect's ContentDetector compares each frame's hue/saturation/brightness to the previous one; a big jump is a camera cut. This finds shots (camera cuts), not scenes, so one scene can produce several shots. Slight over-splitting is preferred to missing a cut.
2. **Read every frame once.** Frames stream in order and only the current shot's candidates stay in memory. Each frame is downscaled to 160 px wide for motion, flash and look measurements.
3. **Pick a keyframe.** The first and last ~1/6 of each shot are skipped to avoid dissolves and flashes. Each remaining frame is scored on sharpness (Laplacian variance), contrast (luma std) and closeness to mid exposure. The highest score wins, unless it's part of a flash.
4. **Faces (optional).** With `--faces live` (YuNet) or `--faces anime` (lbpcascade_animeface), the shot is split into 3 time chunks and the best frame of each chunk becomes a candidate. Candidates are face-checked in order of score, stopping at the first face. If none has a face, the shot is treated as faceless and the top-scoring frame is kept.
5. **Edges, flashes, motion, look.** Measured from the same pass, as below.

### Edge frames

`first` and `last` are the shot's first and last frames, except when the shot opens or closes on a flash: then they're the first and last frames outside it. A white frame says nothing about how a shot can match the one before or after it.

### Flashes

A flash is a run of at most 0.25 s of frames whose mean luma is at least 0.25 (of full scale) above the frames on both sides and above the shot's median frame. A run at the start or end of a shot has only one side to compare with. Slower brightening is lighting, not a flash, and so is the frame after a dip to black.

PySceneDetect cuts at the start of a bright flash and ignores the cut back out of it, because the second change comes sooner than `--min-len`. So a flash between two shots opens the second one, and a flash inside a shot splits it in two at the flash. In both cases the flash is listed in the second shot's `flashes`, and its `first` frame skips it.

### Motion

`motion.curve` has one value per shot frame: the mean Farneback optical-flow speed from the previous frame, in frame widths per second. A frame at `start + k / fps` has the value `curve[k]`. It is `null` where flow means nothing: the shot's first frame, and frames into, during and right out of a flash, because optical flow assumes constant brightness.

`motion.peaks` are the times of motion bursts, the impact points an editor syncs to a beat. The curve is averaged over 0.25 s first. Otherwise anime drawn on twos or threes, where every other frame repeats a drawing, would give a peak per drawing. A peak is a local maximum of that average, at least 1.5× the shot's median and 0.05 widths/s. Peaks are more than 0.25 s apart, and each one is moved to the fastest frame under its window. `mean` and `max` ignore `null`s.

### Look

Measured on 8 evenly spaced usable frames, excluding flashes, at 80 px wide, so shots can be matched by grade:

- `luma`: mean luma, 0–1.
- `hue`: saturation-weighted mean hue in degrees, `null` when mean saturation is under 0.05. Gray pixels have no meaningful hue.
- `saturation`: mean HSV saturation, 0–1.
- `palette`: up to 5 `{hex, share}` colors from k-means in Lab, by share.

On the White Light reference edit, the night classroom, fights and pool come out at hues of 205–230°, and the fireworks, fire and daylight shots at 15–35°. That is the cold-blue grade continuity the breakdown describes.

## Design choices

**Time chunks, not top-N.** The top 5 frames by score cluster in the same steady second, so they say little about the rest of the shot. Taking the best per chunk spreads candidates across the shot.

**No color-diverse candidates.** We considered adding frames whose color histogram differs most from the chunk picks (farthest-point: maximize distance to the nearest existing pick), to catch a pan from a wall to a face. Shots in edits are ~1s and barely change color, so the time chunks already cover them. Worth adding for long takes or full films.

**Face mode rarely changes the pick.** On 4 test edits (1 live action, 3 anime, 94 shots), face mode changed 2 frames, and both picks already had faces. The sharpness/contrast score tends to land on faces anyway, so treat it as a safety net. The anime cascade is also weak: it found 10 of ~15 face shots on one edit.

**Flashes are known only when a shot ends.** So each chunk keeps one more alternate than a flash can have frames, and the first and last 0.25 s of frames are kept for the edges. At 24 fps that is 21 full-resolution frames in memory per shot, or 35 with `--faces`.

Requires `opencv-python-headless<5` (5.0 dropped CascadeClassifier).

Defaults: threshold 23, minimum shot length 8 frames. Override with `--threshold` and `--min-len`. With `--sample N`, only N evenly spaced usable frames are scored for the keyframe; motion, flashes and look still read every frame. A `--proxy WIDTH` speeds up cut detection; everything else uses the original video. Finished-shot stills are written immediately rather than kept for the entire video. See [pipeline.md](pipeline.md) for captioning and dialogue indexing.
