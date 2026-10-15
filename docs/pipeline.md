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
python scene_frames/index.py VIDEO OUT_DIR --sample 5 --proxy 480 --context context.md
```

Builds numbered contact sheets of keyframes and captions them with Claude in parallel. It also transcribes one audio stream locally with faster-whisper, with word timings, in the background while shots are detected and captioned, and can read an embedded text subtitle stream. The contact sheets are sent to Anthropic; audio and subtitles stay local. Whisper downloads its model on first use.

### Captioners

`--captioner` picks how sheets reach Claude. All three send the same prompt and sheet and validate replies the same way:

- `claude` (default): one `claude -p` call per sheet, on your Claude Code sign-in, so there's no key to manage. It loads no tools, settings, MCP servers or CLAUDE.md, and drops `ANTHROPIC_API_KEY` from its environment so the sign-in is what gets used. It pins `--caption-model` exactly: aliases such as "haiku" can be remapped in Claude Code settings, so a full model ID is checked against the model that actually ran. A mismatch stops the run. Each sheet takes about 15 s with Haiku 5.5.
- `api`: the Anthropic API, billed as API usage, for running without Claude Code. Needs `ANTHROPIC_API_KEY` or an `ant auth login` profile; Claude Code's sign-in doesn't reach the API.
- `agent`: writes a jobs file for Claude Code subagents to answer, see [Captioning with agents](#captioning-with-agents). It's for when an interactive session should do the work. Subagents run whatever model their alias maps to.

`api` and `claude` share cached captions for the same `--caption-model`, so switching between them doesn't re-caption.

### Captions

Captions name the characters the model recognizes, write "possibly" before a name it's unsure of, and call people it can't place "unidentified" instead of guessing. Recognition fails most on small, dark, back-view and silhouetted figures. Nobody is named, even as "possibly", when what's visible contradicts how that character looks, such as a different hair color. `characters` lists everyone a caption names by bare name, with `certain: false` for the "possibly"s, and leaves out the unidentified. Captions describe only what's pictured, and text inside the frames is never treated as an instruction. Placing a shot in the story is left to a later text-only pass over shot order and dialogue.

Nothing in the prompt is specific to a show. Two inputs tell the model what it's looking at:

- `--context FILE`: the usual input, a markdown file from a research step for the title. Only its `## Cast` section goes into the prompt, so sheets stay cheap, and the model matches people against the visual identifiers there.
- `--title TEXT`: the light option when there's no context file, e.g. `--title "Show Name, season 2"`. It defaults to the context file's `# ` heading.

Both are in the caption cache signature (the cast section by hash), so changing either re-captions. Editing other sections of the context file doesn't. The system prompt is cached, which only matters once the cast makes it longer than the model's minimum cacheable length.

The context file looks like this:

```markdown
# Show title

## Cast
### Character name
- Hair: color, length, style
- Eyes: color, anything distinctive
- Outfit: what they usually wear, and alternates
- Accessories: anything signature, like a choker, earrings, glasses or a scar
- Other forms: transformations, monster or armored forms, masks

## Scenes
...

## Moments
...
```

Each cast entry needs identifiers that survive a thumbnail: hair color and style, eye color, outfit, signature accessories, and alternate or monster forms. `## Scenes` and `## Moments` are optional and not used for captions. A file without a non-empty `## Cast` section is rejected.

`--sheet-size N` sets shots per sheet (default 12). Sheets are three 512 px tiles wide, so fewer shots per sheet means larger tiles after the API's downscaling. The model matters more than tile size: Haiku 4.5 misreads dark night tiles even at 6 shots per sheet, where Haiku 5.5 reads them at 12.

### Captioning with agents

`--captioner agent` captions with Claude Code instead of the API, so no key is needed and usage counts against the Claude Code plan. Index builds the sheets as usual. For every sheet without cached captions, it writes `OUT_DIR/caption_jobs.json` and stops before `shots.json`. Transcription and subtitles still run, so the next pass is quick.

The jobs file is self-contained:

- `instructions`: what the agent should do.
- `prompt`: the exact system prompt the API path uses, with title and cast.
- `schema`: the reply format.
- `rerun`: the command to run afterwards.
- `jobs`: for each sheet, the image path, the question to answer (`request`, the shot IDs on that sheet) and the `reply` path to write the JSON to.

In Claude Code, ask for something like "caption the jobs in OUT_DIR/caption_jobs.json with one Haiku subagent per sheet, then run its rerun command". Index validates each reply exactly like an API reply and caches it. It names any reply it rejects, which stays pending until it's rewritten. Once every sheet has captions, it writes `shots.json` and deletes the jobs file.

Reply filenames include a hash of the sheet's cache signature, so a reply written for an older sheet, title, cast or prompt is never picked up. Agent captions are cached separately from `--captioner api` captions, so switching captioners re-captions. Subagents don't get the API's structured-output guarantee, so malformed replies are possible and get rejected.

### Dialogue

The first audio stream (`0:a:0`) is transcribed unless `--audio-stream N` picks another. Films often carry several tracks, and reference edits often use the English dub. Index prints the audio streams with language and title tags, and lists any text subtitle streams it finds. `--subs-stream N` adds that subtitle stream (`0:s:N`: SRT, ASS/SSA, mov_text or WebVTT) to each shot as `subtitles`. Image subtitles (PGS, VobSub) can't be read. `--no-transcribe` skips Whisper, for subtitles instead of a transcript or for a video without audio.

Whisper runs with voice-activity detection by default, which keeps it from inventing lines over music and silence. On edits, where dialogue is mixed under a song, it can drop every line: on the White Light reference edit it removed both. Use `--no-vad` there.

Defaults are `--captioner claude`, `--caption-model claude-haiku-5-5`, `--workers 10` (sheets captioned at once), `--whisper-model small`, and automatic language detection. Use `--lang en` to force English or `--whisper-model small.en` for an English-only model.

### shots.json

An array of shots. Times are seconds relative to the input video, and still paths are relative to OUT_DIR.

| Field | Meaning |
|---|---|
| `id`, `start`, `end` | Shot number and time range (`end` exclusive) |
| `keyframe`, `first_frame`, `last_frame` | `{time, file}` stills: the best frame, and how the shot starts and ends |
| `caption`, `characters` | From the captioner. `characters` is `[{name, certain}]` |
| `motion` | `{fps, curve, peaks, mean, max}`: per-frame flow speed in frame widths/s and the times of motion bursts |
| `flashes` | `[{time, frames}]` bright flashes |
| `look` | `{luma, hue, saturation, palette}` for matching color grades |
| `dialogue` | Whisper segments overlapping the shot: `{start, end, text, words: [{start, end, word}]}`. Absent with `--no-transcribe` |
| `subtitles` | Subtitle cues overlapping the shot: `{start, end, text}`. Only with `--subs-stream` |

A line spanning a cut appears in both shots; its word times show which words fall in which. Word text is as Whisper emits it, including the leading space, so joining the words rebuilds the line. Captions and transcription are machine-generated, not verified subtitles. Subtitles converted from ASS keep typeset signs as cues, though vector drawings are dropped.

### Caching

Detection, each sheet's captions, the transcript and the subtitles are cached in `cache/`. Rerun the same command to resume after a failure. Completed sheets are saved individually; transient API errors and invalid caption JSON are retried. Cache signatures include source path/size/mtime and the options that affect each stage. That covers the detection options, caption model, sheet contents, title, cast and prompt version, plus the Whisper model, language, VAD and audio stream, and the subtitle stream. Changed settings rerun only the affected stage. Use a separate OUT_DIR for each video. Whisper resumes at the transcription stage, not within a partly transcribed audio track.

## Render
```sh
python scene_frames/render.py EDL.json OUT.mp4
```

Renders an edit decision list to a frame-accurate MP4. It also writes `edl_table.md` (time, source time, sync, why) and `proof.jpg` (the frames either side of every cut, labeled with time and any `why`). `--check` validates the EDL and its sources without rendering. Every problem is listed with its JSON path, e.g. `clips[3].src_out`. Each clip is cached as an exact-length segment in `render_cache/`, so re-renders redo only changed clips. One ffmpeg filtergraph then composites the rest. Use `--verbose` to print the ffmpeg commands as a starting point for effects the schema lacks.

The EDL is JSON; the full schema is the docstring of `scene_frames/edl.py`. `output` sets width, height and fps. `song` is one continuous window (`source`, `src_in`, `src_out`, optional `speed` and fades), and its length ÷ speed is the output length. `clips` give `source`, `src_in` and `timeline_in`, plus `timeline_out` or `src_out`. Optional clip fields are `speed` (0 freezes), `reverse`, `crop` (zoom and focus point), `color` and `transition` (cut, white or black flash, or crossfade, in frames). Clips can also carry `sync` (beat, drop, word, phrase_end, motion, free) and `why` as notes for critique; the table and proof sheet show them.

Clip motion uses one mechanism: `keyframes`, each with a time inside the clip, an ease (linear, in, out, in_out or hold) and any of `speed`, `zoom`, `x`, `y` and `shake`. Speed keys make ramps and freezes, and the source span is the speed curve's integral. When `src_out` is omitted it is derived, and when it is given it must match. Zoom and position keys make punch-ins and pans, and `shake` keys set a wobble amplitude that decays when keyed to 0. `slowmo` chooses where slowed footage gets its in-between frames. The default `repeat` holds source frames, while `blend` and `motion` (minterpolate) synthesize new ones. Anime is usually drawn on 2s or 3s (8–12 drawings a second), so synthesized frames can smear line art; that choice is left to the agent.

`audio` layers lift dialogue and can duck the song. `overlays` composite images or video, `captions` support a word or character typewriter reveal, and `ending.fade_to_black` closes the edit. Timeline times snap to output frames, and gaps between clips are black.

## Future stages
Director takes shots.json, song lyrics/beats and an edit move library. Opus chooses the story and writes an edit decision list (EDL) containing source in/out points, timing and effects.

Critic watches the result and gives notes for another director/render pass. Director and Critic are planned, not implemented here.
