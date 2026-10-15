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

## Future stages
Director takes shots.json, song lyrics/beats and an edit move library. Opus chooses the story and writes an edit decision list (EDL) containing source in/out points, timing and effects.

Render turns that EDL into an MP4 with ffmpeg.

Critic watches the result and gives notes for another director/render pass. These three stages are planned, not implemented here.
