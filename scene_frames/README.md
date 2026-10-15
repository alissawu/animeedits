# scene_frames

Detect shots with `detect.py VIDEO OUT_DIR`: per shot, a keyframe, first and last frames, a motion curve with its peaks, flashes and look stats. Build the full index for an edit director with `index.py VIDEO OUT_DIR`, which adds captions, word-timed dialogue and optional subtitles, from the video or a subtitle file, in `shots.json`. By default, captions come from `claude -p` on your Claude Code sign-in, with no API key. `--captioner api` uses the Claude API and `--captioner agent` hands sheets to Claude Code agents.

Measure a short reference edit's cuts, flashes, beats and word timings, or map a song file's beats, drops, sections and word timings, with `analyze_edit.py MEDIA OUT_DIR`; see [edit analysis](../docs/edit_analysis.md).

Needs Python 3.11+, `requirements.txt` and ffmpeg/ffprobe. See [frame selection](../docs/frame_selection.md) and [the pipeline](../docs/pipeline.md) for options and outputs.

`sheet.py OUT_DIR [--labels labels.json]` draws one contact sheet of every shot in a detect.py output directory, optionally with a line of text under each tile. The indexer builds its own numbered sheets for captioning, a few shots at a time.

Tests: `python -m unittest discover -s tests` (the video tests generate their input with ffmpeg). Types: `mypy` from this directory (strict, configured in `mypy.ini`).
