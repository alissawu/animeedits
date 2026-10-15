# scene_frames

Detect shots and select keyframes with `detect.py VIDEO OUT_DIR`. Build captions and timestamped dialogue with `index.py VIDEO OUT_DIR`.

Measure a short reference edit's cuts, flashes, beats and word timings with `analyze_edit.py VIDEO OUT_DIR`; see [edit analysis](../docs/edit_analysis.md).

Install `requirements.txt` and ffmpeg. See [frame selection](../docs/frame_selection.md) and [the pipeline](../docs/pipeline.md) for options and outputs.

`sheet.py` remains the small manual contact-sheet script. Run inside an output directory, optionally with an argument to use `labels.json`. The indexer builds its own 24-frame sheets without loading a whole movie’s keyframes into memory.
