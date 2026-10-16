# Reference-conditioned lyric alignment

Run with the repo Python 3.12 environment. Optional dependencies are separate so indexing/rendering do not require an aligner:

```sh
uv pip install --python .venv -r scene_frames/requirements-lyricsync.txt
python scene_frames/lyricsync.py media/song.m4a out/song-lyrics --lyrics media/sung.txt --lyrics-source "corrected sung reference" --video
```

Supply exactly the words sung, in order, with one display line per text line. Plain text and LRC use the existing lyrics.py reader. LRC timestamps are retained as reference metadata, but are NOT alignment windows. Repeats must appear in the reference. For a mashup, edited clip, instrumental gap, added dialogue or missing verse, transcribe/correct the actual sung sequence first; do not blindly align a full published song. English A-Z/apostrophes only; numbers and other scripts are rejected rather than silently discarded.

The pipeline reuses stems.py's content-addressed htdemucs vocal cache, resamples its 22050 Hz vocal track to 16000 Hz and runs WAV2VEC2_ASR_BASE_960H. Acoustic inference uses 20-second chunks for memory; ONE CTC alignment covers the entire concatenated emission and entire reference. No per-line windows, overlap deletion or confidence filtering. Chunk edges can affect precision, especially a tiny final chunk. Low scores flag review, not deletion. The first alignment downloads the acoustic model unless cached; first separation also downloads a separator checkpoint.

If vocals already exist, pass `--vocals vocals.wav` to skip separation. The file must have the same origin and duration as the original input, not a trimmed vocal excerpt. Duration is checked; an equal-length shifted or wrong vocal file cannot be detected automatically.

Outputs are alignment.json (reference provenance, model/runtime versions, word identities/times/scores), words.json, low_score_words.json, lyrics.srt and lyrics.ass. `--video` adds a black-screen lyrics.mp4 with word highlights. It needs ffmpeg's libass/libx264 support. Captions keep uncertain words. Existing output files are overwritten on a rerun; use separate output directories for comparisons. Scores are uncalibrated CTC fit, not probabilities that a word was sung or that a timestamp is correct.

For a complete, unedited recording, `--title TITLE --artist ARTIST` fetches an exact normalized LRCLIB identity match with a duration guard. It saves reference.json/reference.lrc. Exact identity/duration do not establish accuracy. Network failures or ambiguous versions require a supplied reference; there is no fuzzy auto-selection.

`--identify` is a separate opt-in Shazam fingerprint lookup at start/middle/end. It writes identification.json and exits without alignment. It sends information derived from your audio to an external service; recognition can be wrong, particularly on mashups. Select/correct a reference explicitly afterward. The experimental v3 mashup anchoring heuristic was not imported: its 9-word result was not a working solution.

## Use in edit review

```sh
python scene_frames/frames.py --edl out/edit/edl.json out/edit/cuts.jpg --song-map out/song/timeline.json --alignment out/song-lyrics/alignment.json
```

This overlays aligned reference words on the cut-pair word strip without modifying raw Whisper caches. The alignment must name the EDL's song file; its original-source times are mapped through song splices. The score label is CTC, not Whisper recognition confidence. Beats/drops remain from the existing song map. Without a map, aligned words can still be displayed but no beats/drops are invented. Existing `--lyrics` reference lines remain independent; don't treat their LRCLIB line stamps as validated aligned times.

This is an additive review input, not a replacement for analyze_edit.py's raw recognition output. analyze_edit.py --lyrics now computes alignment, or --alignment reuses it, and stores timed lines/words in timeline.json; --lyric-video explicitly requests captions. It does not inject reference-conditioned text into dialogue indexes, section detection or the renderer. Listen/check against the actual track before claiming word-level sync.

## Checks

`python -m unittest discover -s tests -p test_lyricsync.py -v` tests real synthetic CTC paths, retained low-score words, repeats, final-frame timing, tiny-tail inference, LRCLIB guards and ffmpeg subtitle/video rendering. It requires torch/torchaudio but no downloaded models or paid APIs. These tests establish implementation invariants, not alignment accuracy on real songs.
