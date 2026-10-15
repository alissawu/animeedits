# Edit pipeline

## Detect
```sh
pip install -r scene_frames/requirements.txt
python scene_frames/detect.py VIDEO OUT_DIR --sample 5 --proxy 480
```

Requires ffmpeg on PATH. Detect finds camera cuts with ContentDetector (threshold 23, minimum 8 frames). It then scores original-resolution frames and writes a keyframe as soon as each shot ends. Only the current shot's candidates stay in memory.

Writes `scenes.json` and `scene_0004_005.90s.jpg`-style filenames. Omit `--sample` to score every usable frame. `--sample N` scores N evenly spaced frames inside the edge-padded range. `--proxy WIDTH` uses a temporary downscaled video for cut detection only. `--threshold`, `--min-len` and `--faces live|anime` also work in index.

## Index
```sh
export ANTHROPIC_API_KEY="your key"
python scene_frames/index.py VIDEO OUT_DIR --sample 5 --proxy 480
```

Builds numbered contact sheets of up to 24 keyframes, captions them with Claude in parallel, and transcribes the original audio locally with faster-whisper. The contact sheets are sent to Anthropic; audio stays local. Captioning costs API credits; Whisper downloads its model on first use.

Writes `sheets/sheet_0001.jpg` and `shots.json`, an array of `{id,start,end,keyframe,caption,dialogue}`. Times are seconds relative to the input video, keyframe paths are relative to OUT_DIR, and dialogue contains timestamped segments overlapping the shot. A line spanning a cut appears in both shots. Captions and transcription are machine-generated, not verified subtitles.

Defaults are `--caption-model claude-haiku-4-5`, `--workers 4`, `--whisper-model small`, and automatic language detection. Use `--lang en` to force English or `--whisper-model small.en` for an English-only model.

Detection, each sheet's captions, and the transcript are cached in `cache/`. Rerun the same command to resume after a failure. Completed sheets are saved individually; transient API errors and invalid caption JSON are retried. Cache signatures include source path/size/mtime and relevant options, so changed settings invalidate the affected stage. Use a separate OUT_DIR for each video. Whisper resumes at the transcription stage, not within a partly transcribed audio track.

## Render
```sh
python scene_frames/render.py EDL.json OUT.mp4
```

Renders an edit decision list to a frame-accurate MP4. It also writes `edl_table.md` (time, source time, sync, why) and `proof.jpg` (the frames either side of every cut, labeled with time and why). `--check` validates the EDL and its sources without rendering. Every problem is listed with its JSON path, e.g. `clips[3].src_out`. Each clip is cached as an exact-length segment in `render_cache/`, so re-renders redo only changed clips. One ffmpeg filtergraph then composites the rest. Use `--verbose` to print the ffmpeg commands as a starting point for effects the schema lacks.

The EDL is JSON; the full schema is the docstring of `scene_frames/edl.py`. `output` sets width, height and fps. `song` is one continuous window (`source`, `src_in`, `src_out`, optional `speed` and fades), and its length ÷ speed is the output length. `clips` give `source`, `src_in`, `timeline_in`, the required `sync` (beat, drop, word, phrase_end, motion, free) and `why`, plus `timeline_out` or `src_out`. Optional clip fields are `speed` (0 freezes), `reverse`, `crop` (zoom and focus point), `color` and `transition` (cut, white or black flash, or crossfade, in frames).

Clip motion uses one mechanism: `keyframes`, each with a time inside the clip, an ease (linear, in, out, in_out or hold) and any of `speed`, `zoom`, `x`, `y` and `shake`. Speed keys make ramps and freezes, and the source span is the speed curve's integral. When `src_out` is omitted it is derived, and when it is given it must match. Zoom and position keys make punch-ins and pans, and `shake` keys set a wobble amplitude that decays when keyed to 0. `slowmo` chooses where slowed footage gets its in-between frames. The default `repeat` holds source frames, while `blend` and `motion` (minterpolate) synthesize new ones. Anime is usually drawn on 2s or 3s (8–12 drawings a second), so synthesized frames can smear line art; that choice is left to the agent.

`audio` layers lift dialogue and can duck the song. `overlays` composite images or video, `captions` support a word or character typewriter reveal, and `ending.fade_to_black` closes the edit. Timeline times snap to output frames, and gaps between clips are black.

## Future stages
Director takes shots.json, song lyrics/beats and an edit move library. Opus chooses the story and writes an edit decision list (EDL) containing source in/out points, timing and effects.

Critic watches the result and gives notes for another director/render pass. Director and Critic are planned, not implemented here.
