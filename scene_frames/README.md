# scene_frames

Detect shots with `detect.py VIDEO OUT_DIR`: per shot, a keyframe, first and last frames, a motion curve with its peaks, flashes and look stats. Build the full index for an edit director with `index.py VIDEO OUT_DIR`, which adds captions, word-timed dialogue and optional subtitles, from the video or a subtitle file, in `shots.json`. By default, captions come from `claude -p` on your Claude Code sign-in, with no API key. `--captioner api` uses the Claude API and `--captioner agent` hands sheets to Claude Code agents.

Index a whole show from its manifest in `shows/<show>/` with `show.py prepare SHOW`, then `show.py finish SHOW` once its cast notes and synopses are written; see [indexing a show](../docs/indexing.md). It runs index.py and story.py on every video a few at a time, with the options every edit shares, and reports the names the captions give.

Place every indexed shot in the story with `story.py OUT_DIR`. Claude reads a synopsis with the index's captions and lines, as text only, and writes `story.json`: scenes with a label, characters, a confidence, flags on caption names the scene contradicts, and every shot's names with those flags applied.

Measure a short reference edit's cuts, flashes, beats and word timings, or map a song file's beats, drops, sections and word timings, with `analyze_edit.py MEDIA OUT_DIR`; see [edit analysis](../docs/edit_analysis.md).

Needs Python 3.11+, `requirements.txt` and ffmpeg/ffprobe. See [frame selection](../docs/frame_selection.md) and [the pipeline](../docs/pipeline.md) for options and outputs.

`frames.py SOURCE IN OUT OUT.jpg` draws a board of every shot in a span, each with its first and last frames and in-between frames placed where the picture changes; `--dense` draws frames at a fixed rate with a change curve for picking cut points; `--edl EDL.json OUT.jpg` draws the frames either side of each cut in an edit, as render.py renders them. See [frames](../docs/pipeline.md#frames).

`sheet.py OUT_DIR [--labels labels.json]` draws one contact sheet of every shot in a detect.py output directory, optionally with a line of text under each tile. The indexer builds its own numbered sheets for captioning, a few shots at a time.

Tests: `python -m unittest discover -s tests` (the video tests generate their input with ffmpeg). Types: `mypy` from this directory (strict, configured in `mypy.ini`).
