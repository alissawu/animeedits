# animeedits

A harness for seeing how well a model can make short (15–40 s) TikTok-style anime edits that actually mean something. The user hands you a song, some source footage and an idea ("angsty Ethel Cain SatoSugu", "DenReze to this"). Sometimes they hand you less and leave the rest to you. You make the edit, start to finish.

## The bar

Read [breakdowns/denreze-white-light.md](breakdowns/denreze-white-light.md) before planning. It takes apart an edit the user loves: what the song is about, which scenes the editor picked, and what those scenes mean in the story. It shows the level of thought the user responds to. It isn't a structure to follow; your edit should come from what this song and these characters are about.

What the user responds to:
- **One story your edit tells about the characters**, where the song meets the show, with the edit built around it.
- **Something true about the characters** that a fan would recognize, and ideally notice more of on a rewatch.
- **Craft that serves the idea:** timing, flow and look.

You'll be judged on whether it moves the user, whether it means something, whether it's well made, and whether it surprises.

## How to work

Aim to make something emotionally evocative for the people who care about these characters. Research what moves them: the moments they return to, the feelings they describe, and the details or interpretations that give those moments weight. Read actual audience responses rather than relying only on plot summaries or your own assumptions about what people feel. Research the song and relationship before detailed episode and shot-index searches.

Bring that understanding together with what the song expresses to develop the edit's core thesis. Understand the lyrics, their delivery, and the emotional movement of the music. Use the connection between the song and the characters to shape the edit's perspective, progression, choice of song section, and footage.

Don't commit to a thesis until every research subagent has returned: write it in `notes.md` first, and don't start the EDL before it's there. Song mapping, indexing and scouting footage can run while research is out. Have research subagents write findings to a file under `out/<edit-name>/research/` as they go, so you can read partial results instead of waiting blind. If research that changes the thesis arrives after you've built, rethink from the thesis rather than patching the finished cut.

Develop this connection while researching and examining scenes. Consider what those moments mean in the original story and what they convey together in the edit. Let discoveries refine or change the idea. Treat lyrics as part of the song's meaning, rather than a list of things to illustrate.

One thing is not up to you: **review your render before presenting it.** First watch it as a viewer, without reading your intention notes. What relationship do the pictures establish with the music or words? What develops across the sequence? Where is the intended connection difficult to perceive? A theme you can explain is not evidence that the edit communicates it. Storytelling can work through contrast, repetition, revelation or association; it does not require a plot turn or a new clip for every lyric.

Beats, drops, lyrics, vocal phrasing, motion and pauses are optional cues, chosen by taste. Holding through a cue, anticipating it or following it can serve the edit. Revise the sequence when the observed effect does not support the idea, not only its framing and transitions.

Pitching is optional. If you want the user's steer, pitch 2–3 genuinely different concepts (the stretch of song and what it feels like, what the edit is about, the moments it hinges on, what the viewer goes through) and wait for a pick. Otherwise choose the strongest and build it.

Episode summaries, cast notes and indexes support factual checking and footage navigation. The show's shared cast notes and episode synopses are in `shows/<show>/`.

Good places to look. These are starting points, not a checklist:
- **Canon:**
  - The show's Fandom wiki. Its episode and chapter summaries are the most reliable facts.
  - Creator and staff interviews, for intent.
  - For anime, Japanese-language reviews and explainers often have details English sources lack.
- **How fans feel:**
  - Reddit episode-discussion threads. Reddit blocks scripts and the built-in web fetch; see the browser note below.
  - MyAnimeList forum threads. These are readable and full of shot-level fan readings.
  - Tumblr meta posts.
  - AO3 relationship and freeform tags. A ship's most-used tags map its dynamics and recurring motifs. For the works themselves, sort by bookmarks or kudos and prefer fics centered on the ship, which usually list it as their first relationship tag. A ship's tag also lists long ensemble fics where it's a side pairing, and those top every popularity sort.
- **The song:**
  - Genius annotations.
  - Artist interviews and album notes.
  - Timestamped YouTube comments, which show where a song hits people.

## Tools

This is the tool overview; the linked docs cover commands, output fields and limits.

Setup: everything runs in the repo's `.venv` (Python 3.12). Use `.venv/bin/python` wherever these docs say `python`, since an activated venv doesn't carry over between shell commands. To rebuild it with lyric alignment: `uv venv --clear --python 3.12 .venv && uv pip install --python .venv -r scene_frames/requirements.txt -r scene_frames/requirements-lyricsync.txt -r tools/requirements.txt mypy`. The requirements keep torch/torchaudio matched and select a compatible separator on Apple Silicon. ffmpeg and ffprobe must be on PATH; lyric caption videos also need ffmpeg built with libass and libx264. On Homebrew, install and link `ffmpeg-full`; see [caption-video setup](docs/lyricsync.md#ffmpeg-for-caption-videos).

| Need | Command | Docs |
|---|---|---|
| Index a show's videos: shots, stills, motion, flashes, look, captions, dialogue, then scenes and corrected names | `python scene_frames/show.py prepare SHOW`, research, then `finish SHOW` | [indexing.md](docs/indexing.md) |
| Map a song or measure an edit: tempo, beats, downbeats and bar positions, aligned lyric lines and words with uncertainty, stem entries/exits, drops, stops, loop breaks, bass-root changes and voice-shift review hints | `python scene_frames/analyze_edit.py MEDIA OUT_DIR [--lyrics sung.txt] [--alignment FILE] [--lyric-video]` | [edit_analysis.md](docs/edit_analysis.md) |
| Align a checked English sung reference to word times; optionally render black-screen lyric captions | `python scene_frames/lyricsync.py MEDIA OUT_DIR --lyrics FILE.txt [--video]` | [lyricsync.md](docs/lyricsync.md) |
| See source frames or frames either side of an edit's cuts; overlay aligned words with `--alignment FILE` | `python scene_frames/frames.py SOURCE IN OUT OUT.jpg [--dense]`, `python scene_frames/frames.py --edl EDL.json OUT.jpg [--song-map FILE] [--alignment FILE]` | [pipeline.md](docs/pipeline.md#frames) and [lyric review](docs/lyricsync.md#use-in-edit-review) |
| Render an edit decision list (`--check` validates without rendering). Besides cutting the picture, the EDL can shape the song (splices, volume, low- and high-pass filters, bass, compression and echo or reverb tails, all changeable over time) and lay audio over it: film dialogue, which can keep only the voices, or sound effects from files | `python scene_frames/render.py EDL.json OUT.mp4` | [pipeline.md](docs/pipeline.md#render) and [shaping the song](docs/pipeline.md#shaping-the-song); full schema in the `scene_frames/edl.py` docstring |

### Reviewing a render

Use the proof sheet, or `frames.py --edl`, and `analyze_edit.py` to investigate what you notice. Review sheets hide `why` notes and declared sync claims by default; `--show-intent` adds them for comparison afterward. Numerical proximity is descriptive, not a quality score. If you claim a particular sync, identify its target and inspect the relevant cut or action within the shot. Named `sync_points` can record that comparison.

Listen to the finished mix when audio perception is available. If it isn't, distinguish the visual and measured checks you performed from an audiovisual review; do not claim to have heard or verified what you could not assess.

### Research browser

**When a page won't load:** use the built-in web search and fetch first. Only for pages they can't read (Reddit, logged-in or JavaScript-drawn pages), fall back to `python tools/browser.py text URL`. It reads through the user's own logged-in Chrome in background tabs, so their screen never moves. At the start of research, run `python tools/browser.py status`, which checks without prompting. `connected` means reads already work. `ready` means the first read makes Chrome ask the user to Allow the connection: as soon as you know the research needs one of those sites, tell the user, then make that read. Later reads in the session reuse it. `off`, or a read that exits 3, means Chrome isn't set up for it or the Allow wasn't clicked: ask the user to turn on `chrome://inspect/#remote-debugging` and wait for their answer. Fall back to `--profile`, a separate headless Chrome without the user's logins, only if they say no or there's no one to ask. Exit 2 means the site blocked the read: note it and move on. List every source you couldn't read in `notes.md` and when you present the edit, so the user knows what it was made without. Details are in [tools/README.md](tools/README.md).

### Tool notes

- **Song planning:** `timeline.json` groups beats, downbeats, bar positions, strong/weak labels, accent candidates, sections, drops, stops and stem events in `planning`; aligned lines and timed words are in `aligned_lyrics`. Beat This! estimates downbeats without forcing four beats per bar. Bar 1 is the first detected downbeat; pickup positions stay unknown. Strong/weak means downbeat/non-downbeat, not loudness. Bass-root grouping still uses its separate four-beat heuristic. Voice shifts are review hints. Key and genre/instrument tags have been removed. See [edit_analysis.md](docs/edit_analysis.md) for fields and [audio_tools_audit.md](docs/audio_tools_audit.md) for limits.
- **Sung lyrics:** Whisper can get both words and timing wrong. Raw recognition stays in `words`. `analyze_edit.py --lyrics FILE --lyrics-source SOURCE` aligns a checked English sung reference independently, reusing the vocal stem cache and keeping every word with uncertainty flags. Supply the actual sung sequence, including repeats; [LRCLIB](https://lrclib.net) references still need checking. `--alignment FILE` reuses an existing alignment. Caption videos require `--lyric-video` in `analyze_edit.py` or `--video` in standalone [lyricsync.py](docs/lyricsync.md). `frames.py --edl EDL.json OUT.jpg --song-map FILE --alignment alignment.json` overlays aligned words separately from recognition. Inspect uncertain timing before relying on a lyric cue.
- **Dialogue buried under music in footage:** before transcribing, `index.py` skips stretches it thinks have no speech (voice activity detection, or VAD). That also drops lines mixed under a song or loud score. If a scene's dialogue is missing from `shots.json`, rerun with `--no-vad`. `analyze_edit.py` already runs without VAD.
- **Vertical crops:** a 9:16 crop of 16:9 footage keeps only the middle third. Subjects at the edges disappear, like two hands at either side of the frame. Set each clip's `crop` focus and check it on the proof sheet.
- **Indexes are shared between edits.** Each video's index is in `out/index/NAME/` (NAME is the file name without its extension), built from the show's research in `shows/<show>/` by `show.py`. If its captions keep misnaming someone, follow [docs/indexing.md](docs/indexing.md) rather than running `index.py` with your own options.
- **A video you need isn't indexed yet.** Index it in the background and keep working on the edit meanwhile. If it's already in its show's `shows/<show>/show.toml`, run `show.py prepare` and then `finish` with `--only NAME` in the background. If it isn't, hand its indexing research to a subagent briefed with [docs/indexing.md](docs/indexing.md#from-inside-an-edit): the manifest entry, the synopsis, and cast notes checked against the contact sheets. Don't write cast notes inline in a hurry; that's how the Reze film's Angel Devil came out as Makima. Use a subagent rather than a new session, so this works outside the desktop app too.
- **Use `story.json`'s names, not the captions'.** `shots.json` says what each still shows. `story.json` places each shot in its scene and lists every shot's names corrected against the dialogue and the plot (`shots[].characters`), with the caption's originals beside any it changed.
- **Captioning needs no API key.** By default it runs through `claude -p` on the Claude Code sign-in, and so does `story.py`.

## Files

- `media/`: songs and source video. `shows/`: each show's shared indexing research (manifest, cast notes, synopses), committed. `out/index/`: the shared indexes built from it. `out/<edit-name>/`: create this folder when starting an edit. Put research notes, sources and observations, the song map, EDL, intermediate files and final render here. `media/` and `out/` are gitignored, so they exist only in the main checkout: make edits there, not in a git worktree. Never commit media.
- `breakdowns/`: dissected reference edits, the taste reference.
- For each finished edit, keep in `out/<edit-name>/`:
  - `edl.json`
  - the render
  - a short `notes.md` with the concept you went with, what each part of the song does for it, and the moments you planned and which of them you found in the footage

## Code

Python 3.11+, full type hints, strict mypy (`scene_frames/mypy.ini`). Tests: `.venv/bin/python -m unittest discover -s tests`.
