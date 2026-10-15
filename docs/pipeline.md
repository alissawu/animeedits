# Edit pipeline

## Detect
```sh
pip install -r scene_frames/requirements.txt
python scene_frames/detect.py VIDEO OUT_DIR --sample 5 --proxy 480
```

Requires Python 3.11+ and ffmpeg/ffprobe on PATH. Detect finds camera cuts with ContentDetector (threshold 23, minimum 8 frames). It then streams the original video once. For each shot it saves a keyframe plus the first and last frames, and measures a per-frame motion curve with its peaks, bright flashes, and look stats (luma, hue, saturation, palette). Only the current shot's frames stay in memory.

Writes `scenes.json` and `scene_0004_005.90s.jpg`-style stills. Omit `--sample` to score every usable frame for the keyframe. `--sample N` scores N evenly spaced frames inside the edge-padded range. `--proxy WIDTH` uses a temporary downscaled video for cut detection only. `--threshold`, `--min-len` and `--faces live|anime` also work in index. Fields and measurements are described in [frame selection](frame_selection.md).

## Index
```sh
export ANTHROPIC_API_KEY="your key"
python scene_frames/index.py VIDEO OUT_DIR --sample 5 --proxy 480 --context story.md
```

Builds numbered contact sheets of keyframes and captions them with Claude in parallel. It also transcribes one audio stream locally with faster-whisper, with word timings, and can read an embedded text subtitle stream. The contact sheets are sent to Anthropic; audio and subtitles stay local. Captioning costs API credits; Whisper downloads its model on first use.

### Captions

`--context FILE` is markdown the captioner reads before every sheet: character names with what they look like, and scene or chapter summaries. With it, captions name characters ("Reze", not "girl with purple hair"), list them in `characters`, and tag the likely `scene`. Without it, `characters` is empty and `scene` is null. Text inside the frames is never treated as an instruction. The context is a cached prompt block, so a long file is billed at full price on the first sheet and at the cache rate after that, when it's above the model's minimum cacheable length.

`--sheet-size N` sets shots per sheet (default 12). Sheets are three 512 px tiles wide, so fewer shots per sheet means larger tiles after the API's downscaling. Dark night scenes caption badly when tiles are small.

### Dialogue

The first audio stream (`0:a:0`) is transcribed unless `--audio-stream N` picks another. Films often carry several tracks, and reference edits often use the English dub. Index prints the audio streams with language and title tags, and lists any text subtitle streams it finds. `--subs-stream N` adds that subtitle stream (`0:s:N`: SRT, ASS/SSA, mov_text or WebVTT) to each shot as `subtitles`. Image subtitles (PGS, VobSub) can't be read. `--no-transcribe` skips Whisper, for subtitles instead of a transcript or for a video without audio.

Whisper runs with voice-activity detection by default, which keeps it from inventing lines over music and silence. On edits, where dialogue is mixed under a song, it can drop every line: on the White Light reference edit it removed both. Use `--no-vad` there.

Defaults are `--caption-model claude-haiku-4-5`, `--workers 4`, `--whisper-model small`, and automatic language detection. Use `--lang en` to force English or `--whisper-model small.en` for an English-only model.

### shots.json

An array of shots. Times are seconds relative to the input video, and still paths are relative to OUT_DIR.

| Field | Meaning |
|---|---|
| `id`, `start`, `end` | Shot number and time range (`end` exclusive) |
| `keyframe`, `first_frame`, `last_frame` | `{time, file}` stills: the best frame, and how the shot starts and ends |
| `caption`, `characters`, `scene` | From the captioner; `scene` may be null |
| `motion` | `{fps, curve, peaks, mean, max}`: per-frame flow speed in frame widths/s and the times of motion bursts |
| `flashes` | `[{time, frames}]` bright flashes |
| `look` | `{luma, hue, saturation, palette}` for matching color grades |
| `dialogue` | Whisper segments overlapping the shot: `{start, end, text, words: [{start, end, word}]}`. Absent with `--no-transcribe` |
| `subtitles` | Subtitle cues overlapping the shot: `{start, end, text}`. Only with `--subs-stream` |

A line spanning a cut appears in both shots; its word times show which words fall in which. Word text is as Whisper emits it, including the leading space, so joining the words rebuilds the line. Captions and transcription are machine-generated, not verified subtitles. Subtitles converted from ASS keep typeset signs as cues, though vector drawings are dropped.

### Caching

Detection, each sheet's captions, the transcript and the subtitles are cached in `cache/`. Rerun the same command to resume after a failure. Completed sheets are saved individually; transient API errors and invalid caption JSON are retried. Cache signatures include source path/size/mtime and the options that affect each stage. That covers the detection options, caption model, sheet contents, context file and prompt version, plus the Whisper model, language, VAD and audio stream, and the subtitle stream. Changed settings rerun only the affected stage. Use a separate OUT_DIR for each video. Whisper resumes at the transcription stage, not within a partly transcribed audio track.

## Future stages
Director takes shots.json, song lyrics/beats and an edit move library. Opus chooses the story and writes an edit decision list (EDL) containing source in/out points, timing and effects.

Render turns that EDL into an MP4 with ffmpeg.

Critic watches the result and gives notes for another director/render pass. These three stages are planned, not implemented here.
