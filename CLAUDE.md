# animeedits

A harness for seeing how well a model can make short (15–40 s) TikTok-style anime edits that actually mean something. The user hands you a song, some source footage and an idea ("angsty Ethel Cain SatoSugu", "DenReze to this"). Sometimes they hand you less and leave the rest to you. You make the edit, start to finish.

## The bar

Read `breakdowns/` before planning, starting with [denreze-white-light.md](breakdowns/denreze-white-light.md). That edit is the bar. It sets a song about a dangerous first high against a first love who is literally a bomb. The drop lands on a detonation that turns into a kiss, and every cut carries meaning. The breakdowns show depth; they aren't templates. Don't reuse their moves; find the ones this song and this story call for.

What the user responds to:
- **One idea where the song meets the story**, with the edit built around it.
- **A reason behind every cut** that you can say out loud.
- **Something true about the characters** that a fan would recognize, and ideally notice more of on a rewatch.
- **Craft that serves the idea:** timing, flow and look.

You'll be judged on whether it moves the user, whether it means something, whether it's well made, and whether it surprises. Clean but empty loses to rough but meaningful.

## How to work

How you research, plan and build is up to you. Two things are not:

1. **Pitch before building.** Once you know the song and the source well, pitch 2–3 genuinely different concepts. Each one is a sentence on where the song meets the story, the stretch of song you'd use, and the moment or image pair the edit hinges on. Then wait for the user to pick or redirect.
2. **Watch your render before presenting it.** Look at the proof sheet and run `analyze_edit.py` on your output. Confirm every cut lands where you meant it to and every `why` still holds. Revise until they do.

Research the show and the song as deeply as the edit needs; subagents work well for this. Research on a show's cast can double as `index.py --context`; the format is in [docs/pipeline.md](docs/pipeline.md#captions).

Good places to look. These are starting points, not a checklist:
- **Canon:**
  - The show's Fandom wiki. Its episode and chapter summaries are the most reliable facts.
  - Creator and staff interviews, for intent.
  - For anime, Japanese-language reviews and explainers often have details English sources lack.
- **How fans feel:**
  - Reddit episode-discussion threads. Reddit blocks scripts and the built-in web fetch; see the browser note below.
  - MyAnimeList forum threads. These are readable and full of shot-level fan readings.
  - Tumblr meta posts.
  - AO3 relationship and freeform tags. A ship's most-used tags map its dynamics and recurring motifs.
- **What's been done:** popular edits of the character or ship, and the songs they use. Yours should surprise. To study one:
  1. Find links through TikTok search with `tools/browser.py html`.
  2. Download with `tools/tiktok.py URL media/refs/NAME.mp4`.
  3. Measure it with `analyze_edit.py`.
- **The song:**
  - Genius annotations.
  - Artist interviews and album notes.
  - Timestamped YouTube comments, which show where a song hits people.

**When a page won't load:** use the built-in web search and fetch first. Only for pages they can't read (Reddit, TikTok search, logged-in or JavaScript-drawn pages), fall back to `python tools/browser.py text URL`. It reads through the user's own logged-in Chrome in background tabs, so their screen never moves. The first read makes Chrome ask the user to Allow the connection, so tell them before it; later reads in the session reuse it. If it exits 3, ask the user to turn on `chrome://inspect/#remote-debugging`, or use `--profile`. Exit 2 means the site blocked the read: note it and move on. Details are in [tools/README.md](tools/README.md).

## Tools

Setup: `pip install -r scene_frames/requirements.txt`, plus ffmpeg/ffprobe on PATH.

| Need | Command | Docs |
|---|---|---|
| Index source video: shots, stills, motion, flashes, look, captions, dialogue | `python scene_frames/index.py VIDEO OUT_DIR --context FILE` | [pipeline.md](docs/pipeline.md) |
| Map a song, or measure any edit (including your own) | `python scene_frames/analyze_edit.py MEDIA OUT_DIR` | [edit_analysis.md](docs/edit_analysis.md) |
| Render an edit decision list (`--check` validates without rendering) | `python scene_frames/render.py EDL.json OUT.mp4` | [pipeline.md](docs/pipeline.md#render); full schema in the `scene_frames/edl.py` docstring |

Things you can't see from the code:
- **Sung lyrics:** Whisper gets the timing right but often the words wrong. Take the words from the published lyrics and the timing from Whisper. [LRCLIB](https://lrclib.net) has time-synced lyrics; Genius pages are often incomplete.
- **Dialogue buried under music in footage:** before transcribing, `index.py` skips stretches it thinks have no speech (voice activity detection, or VAD). That also drops lines mixed under a song or loud score. If a scene's dialogue is missing from `shots.json`, rerun with `--no-vad`. `analyze_edit.py` already runs without VAD.
- **Vertical crops:** a 9:16 crop of 16:9 footage keeps only the middle third. Subjects at the edges disappear, like two hands at either side of the frame. Set each clip's `crop` focus and check it on the proof sheet.
- **Index each film or episode once.** Reuse its output directory; every stage is cached.
- **Captioning needs no API key.** By default it runs through `claude -p` on the Claude Code sign-in.

## Files

- `media/`: songs and source video. `out/<edit-name>/`: everything a run produces (research notes, indexes, song map, EDL, renders). Both are gitignored. Never commit media.
- `breakdowns/`: dissected reference edits, the taste reference.
- For each finished edit, keep in `out/<edit-name>/`:
  - `edl.json`, where every cut has its `why` and `sync`
  - the render
  - a short `notes.md` with the pitch the user chose and what you'd change next time

## Code

Python 3.11+, full type hints, strict mypy (`scene_frames/mypy.ini`). Tests: `python -m unittest discover -s tests`.
