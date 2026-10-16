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
python scene_frames/index.py VIDEO OUT_DIR --sample 5 --proxy 480 --faces anime --context CAST.md
```

A show's videos are indexed with `show.py`, which runs this with the options every edit shares; see [indexing.md](indexing.md). This section is what index.py does.

Builds numbered contact sheets of keyframes and captions them with Claude in parallel. It also transcribes one audio stream locally with faster-whisper, with word timings, in the background while shots are detected and captioned, and can add text subtitles from the video or a subtitle file. The contact sheets are sent to Anthropic; audio and subtitles stay local. Whisper downloads its model on first use. `--no-captions` stops before captioning: it detects, saves the stills, draws the sheets, transcribes and reads subtitles, with no Claude and no `shots.json`, so cast notes can be checked against the sheets before anything is captioned.

### Captioners

`--captioner` picks how sheets reach Claude. All three send the same prompt and sheet and validate replies the same way:

- `claude` (default): one `claude -p` call per sheet, on your Claude Code sign-in, so there's no key to manage. It loads no tools, settings, MCP servers or CLAUDE.md, and drops `ANTHROPIC_API_KEY` from its environment so the sign-in is what gets used. It pins `--caption-model` exactly: aliases such as "haiku" can be remapped in Claude Code settings, so a full model ID is checked against the model that actually ran. A mismatch stops the run. Each sheet takes about 15 s with Haiku 5.5.
- `api`: the Anthropic API, billed as API usage, for running without Claude Code. Needs `ANTHROPIC_API_KEY` or an `ant auth login` profile; Claude Code's sign-in doesn't reach the API.
- `agent`: writes a jobs file for Claude Code subagents to answer, see [Captioning with agents](#captioning-with-agents). It's for when an interactive session should do the work. Subagents run whatever model their alias maps to.

`api` and `claude` share cached captions for the same `--caption-model`, so switching between them doesn't re-caption.

### Captions

Captions name the characters the model recognizes, write "possibly" before a name it's unsure of, and call people it can't place "unidentified" instead of guessing. Recognition fails most on small, dark, back-view and silhouetted figures. Nobody is named, even as "possibly", when what's visible contradicts how that character looks, such as a different hair color. `characters` lists everyone a caption names by bare name, with `certain: false` for the "possibly"s, and leaves out the unidentified. Names are written in the cast notes' spelling: a name that matches a `### ` heading ignoring case, accents and punctuation, or that is part of exactly one heading ("Gojo", "Gojo Satoru"), becomes that heading ("Satoru Gojo"). A trailing parenthetical in a heading written as names gives aliases, so "Master" becomes "Café owner" under `### Café owner (Master)`; one in lowercase, like `(in Suguru Geto's body)`, is a note. This happens when `shots.json` is built, so cached captions get it too. Names the notes don't have are kept as the caption gives them. Captions describe only what's pictured, and text inside the frames is never treated as an instruction. Placing a shot in the story is a separate text-only pass over shot order and dialogue, [Story](#story).

Nothing in the prompt is specific to a show. Two inputs tell the model what it's looking at:

- `--context FILE`: the usual input, the cast notes for the video's film, season or arc, from `shows/<show>/cast/`. Only its `## Cast` section goes into the prompt, so sheets stay cheap, and the model matches people against the visual identifiers there.
- `--title TEXT`: the light option when there's no context file, e.g. `--title "Show Name, season 2"`. It defaults to the context file's `# ` heading.

Both are in the caption cache signature (the cast section by hash), so changing either re-captions. Editing other sections of the context file doesn't. The system prompt is cached, which only matters once the cast makes it longer than the model's minimum cacheable length.

The context file looks like this:

```markdown
# Show title, season or arc (studio, year)

## Cast

Lighting: ...
Setting: ...

### Character name
- Hair: shape, length, color
- Eyes: color, anything distinctive
- Outfit: what they usually wear, and alternates
- Accessories: anything signature, like a choker, earrings, glasses or a scar
- Other forms: transformations, monster or armored forms, masks, other ages
- Tell apart from: each look-alike, with the one visible difference
```

How to research and check one is the [cast-notes recipe](indexing.md#recipe-cast-notes). Other sections are ignored. A file without a non-empty `## Cast` section is rejected.

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

The first audio stream (`0:a:0`) is transcribed unless `--audio-stream N` picks another. Films often carry several tracks, and reference edits often use the English dub. Index prints the audio streams with language and title tags, and lists any text subtitle streams it finds. `--subs-stream N` adds that subtitle stream (`0:s:N`: SRT, ASS/SSA, mov_text or WebVTT) to each shot as `subtitles`. Image subtitles (PGS, VobSub) can't be read.

`--subs FILE` adds a subtitle file instead: SRT, ASS/SSA or WebVTT, in UTF-8. Index lists subtitle files named after the video that sit beside it, such as `EP01.srt` or `EP01.en.ass` next to `EP01.mp4`. A file made for a different release of the video can be offset by a recap, an intro or a different cut. So when there are both subtitles and a transcript, index checks their timing against each other, by start times alone since the two are often in different languages. It finds the shift, up to 2 minutes either way, at which the most cue starts land within 0.5 s of a Whisper line's start, and takes the median gap over those cues. It prints a line starting `warning: ` when that median is over 1 s, with the `ffmpeg -itsoffset` command that shifts the file back, or when fewer than half the cues match at any shift, which usually means the file is for another video. In-sync subtitles match about 65–85% of their cues; at a wrong shift, about a third match by chance. `--no-transcribe` skips Whisper, for subtitles instead of a transcript or for a video without audio.

Whisper runs with voice-activity detection by default, which keeps it from inventing lines over music and silence. On edits, where dialogue is mixed under a song, it can drop every line: on the White Light reference edit it removed both. Use `--no-vad` there.

Whisper transcribes in the language of the audio track's tag (`jpn` gives `ja`, `eng` gives `en`), and detects the language itself when the tag is missing, undetermined or unusual. `--lang CODE` overrides the tag, and `--lang auto` asks for detection anyway. Index prints which it used.

Defaults are `--captioner claude`, `--caption-model claude-haiku-5-5`, `--workers 10` (sheets captioned at once) and `--whisper-model small`. Use `--whisper-model small.en` for an English-only model.

### shots.json

An array of shots. Times are seconds relative to the input video, and still paths are relative to OUT_DIR.

| Field | Meaning |
|---|---|
| `id`, `start`, `end` | Shot number and time range (`end` exclusive) |
| `keyframe`, `first_frame`, `last_frame` | `{time, file}` stills: the best frame, and how the shot starts and ends |
| `caption`, `characters` | From the captioner. `characters` is `[{name, certain}]`, names in the cast notes' spelling |
| `motion` | `{fps, curve, peaks, mean, max}`: per-frame flow speed in frame widths/s and the times of motion bursts |
| `flashes` | `[{time, frames}]` bright flashes |
| `look` | `{luma, hue, saturation, palette}` for matching color grades |
| `dialogue` | Whisper segments overlapping the shot: `{start, end, text, words: [{start, end, word}]}`. Absent with `--no-transcribe` |
| `subtitles` | Subtitle cues overlapping the shot: `{start, end, text}`. Only with `--subs-stream` or `--subs` |

A line spanning a cut appears in both shots; its word times show which words fall in which. Word text is as Whisper emits it, including the leading space, so joining the words rebuilds the line. Captions and transcription are machine-generated, not verified subtitles. Subtitles converted from ASS keep typeset signs as cues, though vector drawings are dropped.

### Caching

Detection, each sheet's captions, the transcript and the subtitles are cached in `cache/`. Cast notes and titles are keyed by their text, not their file path, so moving a notes file re-captions nothing. Rerun the same command to resume after a failure. Completed sheets are saved individually; transient API errors and invalid caption JSON are retried. Cache signatures include source path/size/mtime and the options that affect each stage. That covers the detection options, caption model, sheet contents, title, cast and prompt version, plus the Whisper model, language, VAD and audio stream, and the subtitle stream. A subtitle file is read again on every run. Changed settings rerun only the affected stage. Use a separate OUT_DIR for each video. Whisper resumes at the transcription stage, not within a partly transcribed audio track.

## Story
```sh
python scene_frames/story.py OUT_DIR --context CAST.md --synopsis SYNOPSIS.md
```

Places every indexed shot in the story. `show.py finish` runs it for each of a show's videos. It reads `OUT_DIR/shots.json` and a synopsis of the video, `OUT_DIR/synopsis.md` unless `--synopsis FILE` names another. Claude reads them as text, with no images, and splits the shots into scenes. Each scene gets a label for what happens, who is in it and a confidence. Caption names that the scene contradicts are flagged, and every shot's names are listed again with the flags applied. The result goes to `OUT_DIR/story.json`. `shots.json` is never changed: captions stay a record of what each still shows, and the story sits beside them.

Captions leave the story out on purpose. Given a synopsis while looking at a thumbnail, a captioner labels scenes from the thumbnail, confidently and often wrongly. This pass sees what a captioner can't: every shot in order, the lines over them and the synopsis. It trusts the lines most (who speaks, who is addressed by name), then the synopsis, then the captions.

Inputs:

- The synopsis: the episode's or film's plot in order, as plain text or markdown, from `shows/<show>/synopses/`; the [synopsis recipe](indexing.md#recipe-synopsis) is how to write one. Without `--synopsis`, story.py reads `OUT_DIR/synopsis.md`.
- `--context FILE`: the cast notes given to `index.py --context`. Scenes name characters as the notes do, which matches the captions. The visual identifiers in the notes help the model say who a misnamed person really is. The notes' `# ` heading is the default `--title`.
- Lines: each shot's subtitles when the index has them, otherwise its Whisper dialogue. A line spanning cuts is given once, under the shot it begins in.

`story.json` is an array of scenes in order, covering every shot exactly once:

| Field | Meaning |
|---|---|
| `id` | Scene number, from 1 |
| `first_shot`, `last_shot` | Shot IDs, inclusive |
| `start`, `end` | Seconds, from the first shot's start to the last shot's end |
| `label` | One sentence on what happens, e.g. "Gojo and Geto escort Riko to Okinawa". Opening and ending credits, title cards and previews are labeled as what they are |
| `characters` | Everyone who appears or speaks, by the cast notes' names |
| `confidence` | `high`: lines or an unmistakable event from the synopsis confirm it. `medium`: it fits the synopsis and captions, but nothing confirms it. `low`: a guess |
| `flags` | `[{shot, name, likely, reason}]`: a name the shot's caption gives that the scene contradicts, who the person likely is (`null` when unclear), and the line or synopsis point that says so. `name` is spelled as in that shot's `characters` |
| `shots` | `[{id, characters, caption_characters}]` for every shot in the scene: its names with the scene's flags applied, and, only when they changed, the caption's own names from `shots.json` |

Use it to find a moment by what happens rather than by how a still looks, and to leave out credits, recaps and previews. Use `shots[].characters` for who is in a shot. A name is flagged only when something contradicts it, not just because the caption was unsure.

The flags are applied like this. A flag replaces the name on its shot with the `likely` person, or drops it when `likely` is `null`. The model often flags a wrong name on one shot and misses the same name on the shots around it, so when a scene's `characters` leave the flagged name out, every shot in the scene that gives it changes too: to the flags' `likely` person when they agree, or dropped when they don't. When the scene's `characters` include the name, that person is really in the scene (the model flagged Reze once in the pool scene, where every other "Reze" is right), so only the flagged shots change. A replacement is always unsure (`certain: false`), since the story pass reads lines and the synopsis but never sees the picture. Applying flags needs no model call, so it reruns on cached scenes.

A scene is a continuous stretch of story in one place and time. One that runs past about 40 shots is split where the action turns, except for credits. Boundaries at changes of place or time and at credits are stable from run to run. Where a long scene splits into beats can move by a shot or a few.

Each request holds up to `--chunk-size` shots (default 300, about 30k tokens with captions and subtitles). A chunk's last scene may run on past its last shot, so that scene is dropped. The next chunk starts at the dropped scene's first shot, with the labels of every scene so far. So chunking never places a scene boundary unless a single scene fills a whole chunk. A 24-minute episode of about 480 shots takes two requests, about 45 s in all with Opus 5.5. A 2,000-shot film takes about eight, one after another.

`--model` defaults to `claude-opus-5-5`. On a 24-minute episode with six planted wrong names, Sonnet 5.5 matched it (all six caught, no false flags, two scene splits a shot off) in about half the time. On the Reze film, though, a third of Sonnet's flags were ones Opus didn't make, and the ones checked were weak or wrong. Haiku 5.5 missed names and scene cuts even on the episode. Calls go through `claude -p` on the Claude Code sign-in, set up exactly as for `--captioner claude`: no tools, settings, MCP servers or CLAUDE.md, and the model is pinned. A reply must cover every shot once with scenes in order, and may flag only names that the scene's captions give. A reply that fails this is retried up to three times.

Each chunk's scenes are cached in `cache/story_NNNN.json`. The cache key is the model plus a hash of the whole request: instructions, synopsis, cast, title, the chunk's shots with their captions and lines, and the scenes before it. Rerunning is free. A changed synopsis or cast reruns every chunk, and changed captions rerun from the first chunk they touch. A failed run resumes after the last finished chunk.

## Render
```sh
python scene_frames/render.py EDL.json OUT.mp4
```

Renders an edit decision list to a frame-accurate MP4. Give `OUT.wav` instead to render only the audio mix (and `edl_table.md`), which is quicker while shaping the song. The MP4 render also writes `edl_table.md` (time, source time, sync, why) and `proof.jpg` (the frames either side of every cut, labeled with time and any `why`). `--check` validates the EDL and its sources without rendering. Every problem is listed with its JSON path, e.g. `clips[3].src_out`. Each clip is cached as an exact-length segment in `render_cache/`, so re-renders redo only changed clips. One ffmpeg filtergraph then composites the rest. Use `--verbose` to print the ffmpeg commands as a starting point for effects the schema lacks.

The EDL is JSON; the full schema is the docstring of `scene_frames/edl.py`. `output` sets width, height and fps. `song` is one continuous window (`source`, `src_in`, `src_out`) or a list of `pieces` joined in order, with optional `speed`, `gain_db`, fades, `keyframes` and a `tail` (see [Shaping the song](#shaping-the-song)). Its length ÷ speed, plus any tail, is the output length. `clips` give `source`, `src_in` and `timeline_in`, plus `timeline_out` or `src_out`. Optional clip fields are `speed` (0 freezes), `reverse`, `crop` (zoom and focus point), `color` and `transition` (cut, white or black flash, or crossfade, in frames). Clips can also carry `sync` (beat, drop, word, phrase_end, motion, free) and `why` as notes for critique; the table and proof sheet show them.

Clip motion uses one mechanism: `keyframes`, each with a time inside the clip, an ease (linear, in, out, in_out or hold) and any of `speed`, `zoom`, `x`, `y` and `shake`. Speed keys make ramps and freezes, and the source span is the speed curve's integral. When `src_out` is omitted it is derived, and when it is given it must match. Zoom and position keys make punch-ins and pans, and `shake` keys set a wobble amplitude that decays when keyed to 0. `slowmo` chooses where slowed footage gets its in-between frames. The default `repeat` holds source frames, while `blend` and `motion` (minterpolate) synthesize new ones. Anime is usually drawn on 2s or 3s (8–12 drawings a second), so synthesized frames can smear line art; that choice is left to the agent.

`audio` layers lift dialogue or place sound effects from any audio file, and can duck or muffle the song while they play; `stem: "voice"` strips the show's score and sound effects from a line (see [Voice stems](#voice-stems)). `overlays` composite images or video, `captions` support a word or character typewriter reveal, and `ending.fade_to_black` closes the edit. Timeline times snap to output frames, and gaps between clips are black.

### Shaping the song

The song can be edited, not just trimmed. All of these are optional; an EDL that uses none of them renders exactly as before.

- **Splices.** `pieces: [{src_in, src_out, crossfade}]` replaces `src_in`/`src_out` and plays source windows one after another, e.g. to cut a verse. A piece starts on the output where the ones before it end, so output times are the sum of the pieces' lengths ÷ speed. `crossfade` (output seconds, default 5 ms, not on the first piece) blends the join into that piece. It is centred on the join: each side plays half of it past its edge, so the source needs that much either side, and `--check` says when it doesn't. The 5 ms default only stops the click of two waveforms meeting; a longer one blends sections.
- **Keyframes.** `keyframes: [{t, ease, gain_db, lowpass, highpass, bass_db, compress}]` on the song work like clip keyframes: `t` is output seconds (the tail included), and each property eases from its previous key (`linear`, `in`, `out`, `in_out`, or `hold` to jump at `t`). `gain_db` adds to the song's gain and is interpolated in dB. `lowpass` and `highpass` are Butterworth cutoffs in Hz, interpolated in octaves, so a sweep from 400 to 20000 sounds even. 20000 is a fully open low-pass and 20 an open high-pass. An open filter is switched off, so it changes nothing. `song.filter_slope` (12, 24, 36 or 48 dB per octave, default 24) sets how steeply both filters and the layers' muffles cut past the cutoff. 12 dulls gently and leaves some air above the cutoff; 48 is close to a wall.
- **Bass and compression.** A `bass_db` key (−24 to 24) sets the gain of a low shelf at 120 Hz, interpolated in dB like `gain_db`. It moves everything below the corner by that much, half of it at 120 Hz and almost nothing from 400 Hz up. `compress: {threshold_db, ratio, attack, release, makeup_db}` on the song runs it through a compressor (ffmpeg `acompressor`, RMS detection, a soft knee). Where the song's level is over `threshold_db` (default −20 dBFS) it comes out `ratio` (default 4) times nearer to the threshold. It reacts within `attack` seconds (default 0.01) and recovers over `release` seconds (default 0.15), and `makeup_db` (default 0) lifts the result. The compressor follows the shelf, so a bass lift into it is held down on the kicks and stays up between them. A `compress` key (0 to 1) sets how much of the song goes through the compressor, mixed with the rest left dry: key it from 0 to 1 with `hold` to compress from the drop on. With no `compress` keys the compressor is fully on throughout.
- **Muffling under a layer.** `muffle_hz` on an `audio` layer low-passes the song at that cutoff, at `filter_slope`, while the layer plays, ramping over the same 0.08 s as `duck_db`, so a line sits on muffled music whose bass carries on. It combines with the song's `lowpass` keys by taking the lower cutoff.
- **Tail.** `tail: {seconds, type, input, wet_db, delay, feedback, decay}` lets the song ring out for `seconds` past its end, which lengthens the output by that much. The song's last `input` seconds (default 0.5) feed the effect and the song itself stops at its end. `type: "echo"` (default) repeats them every `delay` seconds (default 0.25; a beat or half-beat keeps it in time), each `feedback` (default 0.5) times the one before. `type: "reverb"` sends them through a synthetic room that falls 60 dB in `decay` seconds (default 2). `wet_db` is the level of the first repeat or of the reverb. The last quarter of the tail fades to silence. Keyframes apply over the tail, so a low-pass can close as the echoes fade.
- **Sound effects.** An `audio` layer's `source` can be any audio file, such as a riser, impact or whoosh the user provides. `src_in` defaults to 0 and `src_out` to the end of the file, so `{source, timeline_in, gain_db}` places the whole effect. A layer that would run past the end of the output is an error, and the message gives the `src_out` that cuts it there.

The song is processed in this order: pieces, speed, `gain_db`, fades, tail, keyframes (bass shelf, compressor, gain, low-pass, high-pass), then ducks and muffles. After that, the layers are mixed in and the peak limiter applies. Gain, bass, compression and filter changes are applied every 256 samples (5.3 ms) through `asendcmd`. The commands go to `render_cache/song_commands.txt`, which `--verbose` shows being read. A value changes at the start of the block nearest its time.

### Voice stems

An audio layer with `"stem": "voice"` plays only the voices in its span: the lines, breaths and screams, without the score and effects mixed under them. Use it when a line plays over the song, where the show's own music clashes with it. The default, `"mix"`, plays the span's soundtrack as it is.

Render separates the span, plus a second either side for context, then cuts the stem back to the span and caches it in `render_cache/` by source file and span. Gain, fades, ducking and placement are applied in the mix, so changing them doesn't separate again. The separator is BS-RoFormer (the ViperX 1297 checkpoint) through the `audio-separator` package, in `scene_frames/separate.py`. The first voice layer downloads the model, about 640 MB, to `~/.cache/animeedits/separator/`. On an M2 Max, an 8 s line takes about 6 s plus a few seconds to load the model. Check a stem by ear: a vocal model keeps sung vocals in a score (insert songs, choirs) as voice.

The model was chosen on known mixes. Each mix is dialogue from a film's centre channel in a scene with almost no score, plus a stretch of the same film's score with no dialogue: *Chainsaw Man – The Movie: Reze Arc* in English and *Jujutsu Kaisen 0* in Japanese, with the score at the dialogue's level and 6 dB under it. "Score left" is the score's energy in the voice stem relative to the mix. "Voice kept" is the dialogue's level in the stem. On the real lines, the RoFormers and BandIt output silence between lines, while CDX23 leaks score there at about −75 dBFS.

| Separator | Score left | Voice kept | SDR | 8 s line, M2 Max |
|---|---|---|---|---|
| BS-RoFormer ViperX 1297 (used) | −26 to −38 dB | 0.0 dB | 17–26 dB | 6 s |
| MelBand RoFormer (Kim) | −27 to −38 dB | 0.0 to −0.2 dB | 16–25 dB | 3 s |
| HTDemucs ft, vocals | −23 to −34 dB | −0.1 to −0.3 dB | 15–24 dB | 0.5 s (vocals model only) |
| BandIt v2, DnR v3 multilingual, speech | −23 to −32 dB | −0.1 to −2.1 dB | 6–23 dB | 6 s |
| CDX23 (Demucs on DnR), dialog | −22 to −31 dB | −0.2 to −3.0 dB | 5–18 dB | 1.5 s |

The two RoFormers are within a dB of each other. BS-RoFormer had the higher SDR on all six mixes, and MelBand kept some faint non-speech sound in one long gap. The dialogue/music/effects (DnR) models leak as much score as HTDemucs and drop up to 3 dB of the Japanese lines, treating shouted or breathy delivery as effects. `audio-separator` 0.47 has no DnR model; CDX23 and BandIt v2 were run from their own checkpoints.

## Frames
```sh
python scene_frames/frames.py SOURCE IN OUT OUT.jpg                  # scene board
python scene_frames/frames.py SOURCE IN OUT OUT.jpg --dense --fps 6  # shot detail
python scene_frames/frames.py --edl EDL.json OUT.jpg --cut 10-14     # cut pairs
```

Draws frames to look at. It measures and draws; it never captions or judges a picture. IN and OUT are source seconds (`1255.5`) or clock times (`20:55.5`). Every tile is stamped with the source time of the frame it shows, to 0.01 s: the frame's own timestamp from the start of the file, which is what `src_in` means in an EDL. Takes about 5 s for 25 s of 1080p.

The one measurement is **change**: the mean absolute RGB difference between consecutive frames, on frames scaled to analyze_edit.py's analysis size (128 px on the long side), 0–255. Anything that moves pixels counts. A pan, a zoom, camera shake or a lighting change registers as much as a character moving, and a held drawing reads as 0. White and black frames are flagged with analyze_edit.py's flash thresholds. Change into and out of them, and across cuts, is set to 0 so it doesn't swamp the curve.

**Scene board** (default). One row per shot in IN..OUT. Shots come from the index, `out/index/NAME/shots.json`, where NAME is the source file name without its extension. frames.py looks for `out/index` in the working directory and every folder above the script, so it finds the main checkout's index from a worktree. `--index DIR` points elsewhere, and `--no-index` ignores it. With no index the shots are detected with analyze_edit.py's cut detection. A row shows the shot's first and last frames, and in-between frames spaced evenly by accumulated change rather than by time. Change below a noise floor (1.5) doesn't count. A still shot gets one middle frame. Each further 12 of change earns another, up to 6, and they land where the picture changes. A strip under each row plots change per frame, with ▼ on the frames shown and flashes drawn as white or black cells. A shot that starts before IN or ends after OUT is labeled with the part shown. The index's dialogue words (Whisper, with their times) and subtitle cues are printed under the shot they fall in. Up to 30 rows fit on one board.

**Shot detail** (`--dense`, `--fps` default 6). A grid of the frames nearest to every 1/fps seconds, numbered. Consecutive tiles whose mean difference from the first tile of their run is under 2 merge into that tile, marked `held 0.8 s`. A run never merges across a cut or a flash. Under the grid, change per frame on one source-time axis, with shot boundaries in red, flashes, the tiles' positions and the index's dialogue words. Stretches where frames keep changing by at least 2.5, with holds under 0.35 s bridged (anime is drawn on 2s and 3s), are shaded and listed: where change starts, where it peaks, and where it settles. Up to 60 tiles fit; past that it asks you to narrow the span or lower `--fps`.

**Cut pairs** (`--edl`). One row per cut, numbered as in `edl_table.md` and `proof.jpg`: cut N is where clip #N enters. `--cut` takes `N`, `A-B` or a comma list, and defaults to every cut. The left side holds the outgoing clip's last 3 frames, the right side the incoming clip's first 3, each stamped with timeline and source time. A gap of black between them is labeled. Frames are rendered by render.py's own segment pipeline, at tile size, so speed, ramps, reverse, keyframed crop and shake, and color match the render. Composited layers are not drawn. That covers flash and crossfade transitions, overlays and captions; the transition is printed in the row's title and on the tiles it covers. Under each row, a strip covers 1 s either side of the cut, with the song's beats as ticks, sung words as bars and the 6 frames shown. Then the beat and word-start offsets from the cut, singing pauses, the lyric line in effect and any that starts within 1 s (with `--lyrics FILE.lrc`), audio layers playing, and the clip's `why`. Beats and words come from an analyze_edit.py `timeline.json`, passed as `--song-map` or found in the EDL's folder or its parent, one folder down at most. A map of the song file is preferred, converted to the timeline through `song.src_in` and `speed`. Next comes a map of a render in the EDL's folder, already on the timeline. LRC times are song-file times. analyze_edit.py stores word timings, not text, so the words' text comes only from `--lyrics`.

`--crop-x F` previews a 9:16 crop centred at F (0 is the left edge, 1 the right) with render.py's crop math. On a scene board or detail sheet it crops the source, and change is measured inside the crop. With `--edl` it crops the edit's 16:9 output, as a vertical cut of the same edit would; a vertical EDL already sets each clip's crop, so it is refused there.

## Future stages
Director takes shots.json, song lyrics/beats and an edit move library. Opus chooses the story and writes an edit decision list (EDL) containing source in/out points, timing and effects.

Critic watches the result and gives notes for another director/render pass. Director and Critic are planned, not implemented here.
