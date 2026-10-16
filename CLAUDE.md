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

How you research, plan and build is up to you. What tends to work is planning from understanding rather than from browsing: knowing the song and the story well enough to say which stretch of the song you'll use, what it feels like, what the edit is about, and which moments it hinges on, then going to the footage to find those moments and discover what you didn't know to look for. It also helps to work out what each part of the stretch (a verse, a build, a drop, a quiet stretch; the song map shows where they fall) does for that idea, so the whole stretch carries it rather than only the moment it builds to. Then think through what the viewer goes through against the song: what they expect, where it turns, where it peaks, how it leaves them.

One thing is not up to you: **review your render before presenting it.** First watch it as a viewer, without reading your intention notes. What relationship do the pictures establish with the music or words? What develops across the sequence? Where is the intended connection difficult to perceive? A theme you can explain is not evidence that the edit communicates it. Storytelling can work through contrast, repetition, revelation or association; it does not require a plot turn or a new clip for every lyric.

Use the proof sheet, or `frames.py --edl`, and `analyze_edit.py` to investigate what you notice. Review sheets hide `why` notes and declared sync claims by default; `--show-intent` adds them for comparison afterward. Beats, drops, lyrics, vocal phrasing, motion and pauses are optional cues, chosen by taste. Holding through a cue, anticipating it or following it can serve the edit. Numerical proximity is descriptive, not a quality score. If you claim a particular sync, identify its target and inspect the relevant cut or action within the shot. Named `sync_points` can record that comparison. Listen to the finished mix when audio perception is available. If it isn't, distinguish the visual and measured checks you performed from an audiovisual review; do not claim to have heard or verified what you could not assess. Revise the sequence when the observed effect does not support the idea, not only its framing and transitions.

Pitching is optional. If you want the user's steer, pitch 2–3 genuinely different concepts (the stretch of song and what it feels like, what the edit is about, the moments it hinges on, what the viewer goes through) and wait for a pick. Otherwise choose the strongest and build it.

Investigate what people find moving about these characters: the specific moments, tensions and interpretations they return to, and why. Explore how the song connects to those details. Treat a plausible interpretation as something to develop and question. Neither confidence in an explanation nor a count of sources establishes that the research is sufficient. Keep researching while you plan: when a question comes up, search for it yourself or send another agent. Subagents work well for this, but they report back in summary, so ask them for the specifics you need (scenes, timestamps, lines, what fans say), not just conclusions. The show's cast notes and episode synopses are already in `shows/<show>/`; your research is what's specific to this edit, in `out/<edit-name>/`.

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

**When a page won't load:** use the built-in web search and fetch first. Only for pages they can't read (Reddit, logged-in or JavaScript-drawn pages), fall back to `python tools/browser.py text URL`. It reads through the user's own logged-in Chrome in background tabs, so their screen never moves. At the start of research, run `python tools/browser.py status`, which checks without prompting. `connected` means reads already work. `ready` means the first read makes Chrome ask the user to Allow the connection: as soon as you know the research needs one of those sites, tell the user, then make that read. Later reads in the session reuse it. `off`, or a read that exits 3, means Chrome isn't set up for it or the Allow wasn't clicked: ask the user to turn on `chrome://inspect/#remote-debugging` and wait for their answer. Fall back to `--profile`, a separate headless Chrome without the user's logins, only if they say no or there's no one to ask. Exit 2 means the site blocked the read: note it and move on. List every source you couldn't read in `notes.md` and when you present the edit, so the user knows what it was made without. Details are in [tools/README.md](tools/README.md).

## Tools

Setup: everything runs in the repo's `.venv` (Python 3.12). Use `.venv/bin/python` wherever these docs say `python`, since an activated venv doesn't carry over between shell commands. To rebuild it: `uv venv --python 3.12 .venv && uv pip install --python .venv -r scene_frames/requirements.txt -r tools/requirements.txt mypy`. ffmpeg and ffprobe must be on PATH.

| Need | Command | Docs |
|---|---|---|
| Index a show's videos: shots, stills, motion, flashes, look, captions, dialogue, then scenes and corrected names | `python scene_frames/show.py prepare SHOW`, research, then `finish SHOW` | [indexing.md](docs/indexing.md) |
| Map a song, or measure any edit (including your own). For a song: key and tempo, beats, when the sung words fall, when drums, bass and vocals come in and drop out, the drop, where the band stops, where the voice jumps, where a loop breaks, and chord changes | `python scene_frames/analyze_edit.py MEDIA OUT_DIR` | [edit_analysis.md](docs/edit_analysis.md) |
| Align a checked English sung reference to word times; render black-screen lyric captions | `python scene_frames/lyricsync.py MEDIA OUT_DIR --lyrics FILE.txt --video` | [lyricsync.md](docs/lyricsync.md) |
| See what happens in a stretch of source or across an edit's cuts, as frames stamped with source times: a board of every shot, a dense strip for picking a cut point, or the frames either side of each cut | `python scene_frames/frames.py SOURCE IN OUT OUT.jpg [--dense]`, `python scene_frames/frames.py --edl EDL.json OUT.jpg` | [pipeline.md](docs/pipeline.md#frames) |
| Render an edit decision list (`--check` validates without rendering). Besides cutting the picture, the EDL can shape the song (splices, volume, low- and high-pass filters, bass, compression and echo or reverb tails, all changeable over time) and lay audio over it: film dialogue, which can keep only the voices, or sound effects from files | `python scene_frames/render.py EDL.json OUT.mp4` | [pipeline.md](docs/pipeline.md#render) and [shaping the song](docs/pipeline.md#shaping-the-song); full schema in the `scene_frames/edl.py` docstring |

Things you can't see from the code:
- **Sung lyrics:** Whisper can get both words and timing wrong. The song map preserves its recognized text, times and confidence separately from published lyrics supplied with `--lyrics FILE --lyrics-source SOURCE`. [LRCLIB](https://lrclib.net) has time-synced lyrics; Genius pages are often incomplete. Reference line times are unverified and are not word alignment. For word timing from a checked reference, use [lyricsync.py](docs/lyricsync.md); it keeps uncertain words instead of dropping them. `frames.py --edl --alignment alignment.json` overlays these words separately from raw recognition. Inspect disagreements, missing words and uncertain timing before relying on a lyric cue. `frames.py --edl --song-map FILE` shows recognized words and any reference stored in the map; `--lyrics FILE` can supply a separate reference.
- **Dialogue buried under music in footage:** before transcribing, `index.py` skips stretches it thinks have no speech (voice activity detection, or VAD). That also drops lines mixed under a song or loud score. If a scene's dialogue is missing from `shots.json`, rerun with `--no-vad`. `analyze_edit.py` already runs without VAD.
- **Vertical crops:** a 9:16 crop of 16:9 footage keeps only the middle third. Subjects at the edges disappear, like two hands at either side of the frame. Set each clip's `crop` focus and check it on the proof sheet.
- **Indexes are shared between edits.** Each video's index is in `out/index/NAME/` (NAME is the file name without its extension), built from the show's research in `shows/<show>/` by `show.py`. If its captions keep misnaming someone, follow [docs/indexing.md](docs/indexing.md) rather than running `index.py` with your own options.
- **A video you need isn't indexed yet.** Index it in the background and keep working on the edit meanwhile. If it's already in its show's `shows/<show>/show.toml`, run `show.py prepare` and then `finish` with `--only NAME` in the background. If it isn't, hand its indexing research to a subagent briefed with [docs/indexing.md](docs/indexing.md#from-inside-an-edit): the manifest entry, the synopsis, and cast notes checked against the contact sheets. Don't write cast notes inline in a hurry; that's how the Reze film's Angel Devil came out as Makima. Use a subagent rather than a new session, so this works outside the desktop app too.
- **Use `story.json`'s names, not the captions'.** `shots.json` says what each still shows. `story.json` places each shot in its scene and lists every shot's names corrected against the dialogue and the plot (`shots[].characters`), with the caption's originals beside any it changed.
- **Captioning needs no API key.** By default it runs through `claude -p` on the Claude Code sign-in, and so does `story.py`.

## Files

- `media/`: songs and source video. `shows/`: each show's research (manifest, cast notes, synopses), committed. `out/index/`: the shared indexes built from it. `out/<edit-name>/`: everything else a run produces (research notes, song map, EDL, renders). `media/` and `out/` are gitignored, so they exist only in the main checkout: make edits there, not in a git worktree. Never commit media.
- `breakdowns/`: dissected reference edits, the taste reference.
- For each finished edit, keep in `out/<edit-name>/`:
  - `edl.json`
  - the render
  - a short `notes.md` with the concept you went with, what each part of the song does for it, and the moments you planned and which of them you found in the footage

## Code

Python 3.11+, full type hints, strict mypy (`scene_frames/mypy.ini`). Tests: `.venv/bin/python -m unittest discover -s tests`.
