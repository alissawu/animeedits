# animeedits

A harness for seeing how well a model can make short (15–40 s) TikTok-style anime edits that actually mean something. The user hands you a song, some source footage and an idea ("angsty Ethel Cain SatoSugu", "DenReze to this"). Sometimes they hand you less and leave the rest to you. You make the edit, start to finish.

## The bar

Read [breakdowns/denreze-white-light.md](breakdowns/denreze-white-light.md) before planning. It takes apart an edit the user loves: what the song is about, which scenes the editor picked, and what those scenes mean in the story. It shows the level of thought the user responds to. It isn't a structure to follow; your edit should come from what this song and this story are about.

What the user responds to:
- **One idea where the song meets the story**, with the edit built around it.
- **Something true about the characters** that a fan would recognize, and ideally notice more of on a rewatch.
- **Craft that serves the idea:** timing, flow and look.

You'll be judged on whether it moves the user, whether it means something, whether it's well made, and whether it surprises. Clean but empty loses to rough but meaningful.

## How to work

How you research, plan and build is up to you. What tends to work is planning from understanding rather than from browsing: knowing the song and the story well enough to say which stretch of the song you'll use, what it feels like, what the edit is about, and which moments it hinges on, then going to the footage to find those moments and discover what you didn't know to look for. It also helps to think through what the viewer goes through against the song: what they expect, where it turns, where it peaks, how it leaves them.

One thing is not up to you: **watch your render before presenting it.** Look at the proof sheet and run `analyze_edit.py` on your output. Confirm the cuts land where you meant them to. Then watch it as a viewer: what each cut makes you expect and feel, and whether the edit does what you planned. Revise until it does.

Pitching is optional. If you want the user's steer, pitch 2–3 genuinely different concepts (the stretch of song and what it feels like, what the edit is about, the moments it hinges on, what the viewer goes through) and wait for a pick. Otherwise choose the strongest and build it.

Research the show and the song as deeply as the edit needs, and keep researching while you plan: when a question comes up, search for it yourself or send another agent. Subagents work well for this, but they report back in summary, so ask them for the specifics you need (scenes, timestamps, lines, what fans say), not just conclusions. Research on a show's cast can double as `index.py --context`; the format is in [docs/pipeline.md](docs/pipeline.md#captions).

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
| Index source video: shots, stills, motion, flashes, look, captions, dialogue | `python scene_frames/index.py VIDEO out/index/NAME ...`, full command below | [pipeline.md](docs/pipeline.md) |
| Place each shot in the story: scenes, what happens, who's there, caption names the scene contradicts | `python scene_frames/story.py out/index/NAME --context out/index/SHOW.md` | [pipeline.md](docs/pipeline.md#story) |
| Map a song, or measure any edit (including your own) | `python scene_frames/analyze_edit.py MEDIA OUT_DIR` | [edit_analysis.md](docs/edit_analysis.md) |
| Render an edit decision list (`--check` validates without rendering) | `python scene_frames/render.py EDL.json OUT.mp4` | [pipeline.md](docs/pipeline.md#render); full schema in the `scene_frames/edl.py` docstring |

Things you can't see from the code:
- **Sung lyrics:** Whisper gets the timing right but often the words wrong. Take the words from the published lyrics and the timing from Whisper. [LRCLIB](https://lrclib.net) has time-synced lyrics; Genius pages are often incomplete.
- **Dialogue buried under music in footage:** before transcribing, `index.py` skips stretches it thinks have no speech (voice activity detection, or VAD). That also drops lines mixed under a song or loud score. If a scene's dialogue is missing from `shots.json`, rerun with `--no-vad`. `analyze_edit.py` already runs without VAD.
- **Vertical crops:** a 9:16 crop of 16:9 footage keeps only the middle third. Subjects at the edges disappear, like two hands at either side of the frame. Set each clip's `crop` focus and check it on the proof sheet.
- **Indexes are shared between edits.** Index each film or episode into `out/index/NAME/`, where NAME is the video's file name without its extension, with exactly this command:
  `python scene_frames/index.py VIDEO out/index/NAME --sample 5 --proxy 480 --faces anime --lang LANG --context out/index/CAST.md`
  LANG is the audio's language (`en`, `ja`). Add `--subs FILE` when index.py lists a subtitle file beside the video. Every stage is cached, so the command finishes a partial index, and on a finished one it only rebuilds the contact sheets, a minute or two. Other options redo detection for the whole video.
- **Cast notes are shared too.** `out/index/CAST.md` holds the cast notes for one film, season or arc, used by every video of it, e.g. `jujutsu_kaisen_0.md` and `jujutsu_kaisen_s2_hidden_inventory.md`. Characters look different across them (ages, outfits), and listing characters who never appear makes captions guess them, so each describes only its own. Use the existing file when there is one. If captions keep misnaming someone, fix the notes, but every video that uses them gets re-captioned.
- **Captions say what a still shows; `story.json` says where it falls in the plot.** After indexing, write the episode's or film's plot in order to `out/index/NAME/synopsis.md` (a Fandom wiki episode summary works well), then run `story.py`. It's cached and shared like the index. Before trusting a caption's character name, check `story.json` for a flag on that shot.
- **Captioning needs no API key.** By default it runs through `claude -p` on the Claude Code sign-in, and so does `story.py`.

## Files

- `media/`: songs and source video. `out/index/`: shared indexes and cast notes. `out/<edit-name>/`: everything else a run produces (research notes, song map, EDL, renders). `media/` and `out/` are gitignored, so they exist only in the main checkout: make edits there, not in a git worktree. Never commit media.
- `breakdowns/`: dissected reference edits, the taste reference.
- For each finished edit, keep in `out/<edit-name>/`:
  - `edl.json`
  - the render
  - a short `notes.md` with the concept you went with, the moments you planned and which of them you found in the footage, and what you'd change next time

## Code

Python 3.11+, full type hints, strict mypy (`scene_frames/mypy.ini`). Tests: `.venv/bin/python -m unittest discover -s tests`.
