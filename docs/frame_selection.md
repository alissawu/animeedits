# Frame selection

How `scene_frames/detect.py` turns a video into one thumbnail per shot.

## Usage

```
python scene_frames/detect.py INPUT OUT_DIR [--faces live|anime]
```

Writes `scene_XX_SSS.SSs.jpg` (shot number + timestamp of the chosen frame) and `scenes.json` with each shot's start, end and chosen-frame time. Agents can rename frames with a label as long as the timestamp stays at the end, e.g. `04_fireworks_kiss_005.90s.jpg`.

## How it works

1. **Find shots.** PySceneDetect's ContentDetector compares each frame's hue/saturation/brightness to the previous one; a big jump is a camera cut. This finds shots (camera cuts), not scenes, so one scene can produce several shots. Slight over-splitting is preferred to missing a cut.
2. **Pick a frame per shot.** The first and last ~1/6 of each shot are skipped to avoid dissolves and flashes. Each remaining frame is scored on sharpness (Laplacian variance), contrast (luma std) and closeness to mid exposure. The highest score wins.
3. **Faces (optional).** With `--faces live` (YuNet) or `--faces anime` (lbpcascade_animeface), the shot is split into 3 time chunks and the best frame of each chunk becomes a candidate. Candidates are face-checked in order of score, stopping at the first face. If none has a face, the shot is treated as faceless and the top-scoring frame is kept.

## Design choices

**Time chunks, not top-N.** The top 5 frames by score cluster in the same steady second, so they say little about the rest of the shot. Taking the best per chunk spreads candidates across the shot.

**No color-diverse candidates.** We considered adding frames whose color histogram differs most from the chunk picks (farthest-point: maximize distance to the nearest existing pick), to catch a pan from a wall to a face. Shots in edits are ~1s and barely change color, so the time chunks already cover them. Worth adding for long takes or full films.

**Face mode rarely changes the pick.** On 4 test edits (1 live action, 3 anime, 94 shots), face mode changed 2 frames, and both picks already had faces. The sharpness/contrast score tends to land on faces anyway, so treat it as a safety net. The anime cascade is also weak: it found 10 of ~15 face shots on one edit.

Requires `opencv-python-headless<5` (5.0 dropped CascadeClassifier).
