# Indexing a show

An edit session finds its clips in an index: every shot of every episode or film, with stills, a caption, who's in it, what's said and where it falls in the story. This doc is how to build those indexes for a show so they come out as good as the Jujutsu Kaisen ones, whoever builds them. Follow it start to finish for a new show. For a show already in `shows/`, skip to [A show that's already researched](#a-show-thats-already-researched).

## Where things live

| Folder | What | In git |
|---|---|---|
| `shows/<show>/` | The show's research: `show.toml` (the manifest), `cast/` (cast notes, one file per film, season or arc) and `synopses/` (one per video) | Yes |
| `media/<show>/` | Your video and subtitle files | No, never |
| `out/index/<NAME>/` | One index per video, `NAME` being the video's file name without its extension: stills, contact sheets, `shots.json`, `story.json`, caches and `index.log` | No; it's rebuilt from `shows/` plus your media |

`media/` and `out/` are gitignored, so they exist only in the main checkout. Run everything from there, with the repo's `.venv` (setup is in [CLAUDE.md](../CLAUDE.md)). `.venv/bin/python` is written `python` below.

### Which research goes here and which goes with an edit

- **Here, in `shows/<show>/`:** facts about the footage that every edit of the show needs, whatever the song. The cast notes say what each character looks like in this footage and feed the captions. The synopses say what happens in each video, in order, and feed the story pass. Write them once, commit them, and every edit and every friend with the repo uses them.
- **With the edit, in `out/<edit>/`:** the song (lyrics, what it's about, its beats and sections), fan readings of the characters or the ship, the angle, and which moments carry it. Keep fan readings out of `shows/` on purpose. A shared list of famous moments would put the same scenes in front of every edit, and each edit finds different things when it researches fans from its own angle.

## New show

| Step | Command or file | Claude? | Time |
|---|---|---|---|
| 1. Add the videos | `media/<show>/` | | |
| 2. Write the manifest | `shows/<show>/show.toml` | | |
| 3. Prepare | `python scene_frames/show.py prepare <show>` | No | ~10 min per episode, 2 at a time |
| 4. Research | `shows/<show>/cast/*.md`, `shows/<show>/synopses/*.md` | You | |
| 5. Finish | `python scene_frames/show.py finish <show>` | Haiku, then Opus | ~4 min per episode, ~12 per film |
| 6. Check, fix, commit | the summary from step 5 | | |

### 1. Add the videos

Put the files in `media/<show>/`, in folders if you like (`media/jujutsu_kaisen/season_2/`). Use one lowercase folder name per show, such as `chainsaw_man`, and the same name for `shows/<show>/`. Give every video a different file name across all shows, since the name is also its index folder's name.

If you have subtitles for a video, put them beside it as `<video>.en.srt` (or `.ass`, `.vtt`), named exactly like the video: `Jujutsu_Kaisen_S02E10.BluRay.Japanese.en.srt` beside `Jujutsu_Kaisen_S02E10.BluRay.Japanese.mp4`. Subtitles are worth having: they're accurate where Whisper's transcript isn't, above all for Japanese, and the story pass trusts them over everything else. They must be UTF-8 text; image subtitles (PGS, VobSub) can't be read. Machine-made subtitles, such as YouTube's auto captions, add nothing over Whisper's transcript, so leave them out.

### 2. Write the manifest

`shows/<show>/show.toml` lists every video, grouped by film, season or arc. Each group has one cast-notes file, used for every video in it.

```toml
# Jujutsu Kaisen: the research every edit of this show shares. docs/indexing.md explains each field.
# Video and subtitle paths start in media/; cast notes and synopses in this folder.
title = "Jujutsu Kaisen"

[[groups]]
name = "Hidden Inventory / Premature Death"
cast = "cast/s2_hidden_inventory.md"

[[groups.videos]]
video = "jujutsu_kaisen/season_2/Jujutsu_Kaisen_S02E01.BluRay.Japanese.mp4"
subs = "jujutsu_kaisen/season_2/Jujutsu_Kaisen_S02E01.BluRay.Japanese.en.srt"
synopsis = "synopses/s2e01.md"

[[groups.videos]]
video = "jujutsu_kaisen/season_2/Jujutsu_Kaisen_S02E02.BluRay.Japanese.mp4"
subs = "jujutsu_kaisen/season_2/Jujutsu_Kaisen_S02E02.BluRay.Japanese.en.srt"
synopsis = "synopses/s2e02.md"
```

| Key | Where | Meaning |
|---|---|---|
| `title` | top | The show's name, for the summaries |
| `[[groups]]` | top | One per film, season or arc, in story order |
| `name` | group | A short name, unique in the manifest. `--only` takes it |
| `cast` | group | The group's cast notes, relative to `shows/<show>/` |
| `[[groups.videos]]` | group | One per video, in story order |
| `video` | video | The video file, relative to `media/` |
| `subs` | video | Optional: its subtitle file, relative to `media/` |
| `synopsis` | video | Its synopsis, relative to `shows/<show>/`, such as `synopses/s2e01.md` |
| `lang` | video | Optional: the dialogue's language as a Whisper code (`ja`, `en`). Leave it out: index.py reads it from the audio track's language tag (`jpn` gives `ja`), and detects it when the tag is missing. Set it only when the tag is wrong |

Paths use `/`, never start with `/` and never use `..`, so the manifest means the same in every checkout. The files they name needn't exist yet: the cast notes and synopses are written in step 4.

**How to group.** Make a group wherever the characters' looks or the cast change: each film, each season, and each arc within a season that jumps in time. Jujutsu Kaisen has three: the film *Jujutsu Kaisen 0*, season 2's Hidden Inventory arc (2006, the main characters as teenagers) and season 2's Shibuya arc (2018). A cast-notes file that lists characters who don't appear makes the captioner guess them: one season-wide file that also described 2018 characters made Haiku caption teenage Geto as "possibly Kenjaku" and teenage Shoko as "possibly Nobara". Splitting the file by arc fixed it.

Check the manifest with `python scene_frames/show.py prepare <show> --dry-run`, which prints every command without running anything.

### 3. Prepare

```sh
python scene_frames/show.py prepare <show>
```

For each video this finds the shots, saves three stills per shot, draws numbered contact sheets of the keyframes (`out/index/<NAME>/sheets/sheet_0001.jpg`, 12 shots each), transcribes the dialogue with Whisper and reads the subtitles. It uses no Claude. Two videos run at a time (`--jobs N` changes it; past 3, each gets slower), and every stage is cached, so rerunning the command after a failure or Ctrl-C picks up where it stopped. Each video's full output goes to `out/index/<NAME>/index.log`.

The summary at the end lists each video with its shot count, the dialogue language and any warnings, then the research still missing. Look at the warnings now:

- **"the subtitles start 2.3 s late against the dialogue"**: the file was timed for another release of the video, with a different intro or recap. Shift it with the `ffmpeg -itsoffset` command the warning gives, point `subs` at the shifted file, and rerun prepare.
- **"the subtitles don't line up with the dialogue at any shift"**: they're probably for another episode, or another cut. Check a few cues against the video.
- **"language auto-detect, since the audio track's tag is und"**: Whisper picks the language itself, which is usually right. If the transcript comes out in the wrong language, set `lang`.

### 4. Research

Write one cast-notes file per group and one synopsis per video, at the paths the manifest gives. The two recipes below are the whole method; the Jujutsu Kaisen files in `shows/jujutsu_kaisen/` are worked examples of both. Prepare's contact sheets exist so the cast notes can be checked against your own footage, so write the notes after prepare, not before.

Research the synopses first: their character lists tell you who the cast notes need. Subagents work well for both, one per character batch or episode, as long as they report the specifics (the wiki's wording, sheet numbers, what each sheet shows) rather than conclusions.

### 5. Finish

```sh
python scene_frames/show.py finish <show>
```

For each video this captions every contact sheet with Haiku through `claude -p` on your Claude Code sign-in (no API key), using the group's cast notes, and writes `shots.json`. It then reports the names the captions give and runs the story pass, which splits the shots into scenes with Opus and corrects names the dialogue contradicts (`story.json`). Captions are cached per sheet and keyed by the cast notes' text, so finish re-captions a video only when its group's notes change. finish refuses a video whose cast notes or synopsis don't exist yet, and does the rest.

Before captioning a whole show with new notes, try them on a few sheets:

```sh
python scene_frames/show.py try <show> --only <NAME> --sheets 12 40 41
```

It captions those sheets the way finish would, with the notes as they are now, prints each shot's names and caption, and caches nothing. Pick sheets where look-alikes appear, the ones you found while checking the notes. A sheet costs one Haiku call, about 15 s. If a character comes out misnamed, fix the notes and try again. A full film is ~180 sheets and a 24-minute episode ~45, so this is much cheaper than finding the problem after finish.

### 6. Check the report, fix, commit

finish's summary shows, under each video:

```
names in the captions of 2112 shots: Denji 484 (31 possibly), Reze 295 (12 possibly), Makima 133 (40 possibly), ...
not in the cast notes: Aki 1
not in the synopsis: Kishibe 5
133 scenes, 71 flagged caption names, 74 shots with corrected names -> out/index/.../story.json
```

`python scene_frames/show.py report <show>` prints the name report again at any time, with no Claude. Read it against what you know of each video:

- **Counts that don't fit the story.** A side character named in hundreds of shots, or a main character barely named, means the notes let the captioner confuse two people. In the Chainsaw Man film the first notes described the Angel Devil with white hair. In the film his hair is auburn, so Haiku called him Makima, the other red-haired character: Makima 133, Angel Devil 12, in a film where he's the bigger part.
- **"not in the cast notes".** The captioner named someone the notes don't describe. A real cameo is fine; add an entry if they matter to edits. A guess from another arc (2018 characters in 2006 episodes) means a look-alike's "Tell apart from" line or the setting line is missing.
- **"not in the synopsis".** Named, but the plot never mentions them. That's either a cameo, the opening or ending credits, or the next-episode preview, or a wrong name. A short name that slipped through also lands here; the cast notes' heading decides the spelling.
- **"shots with corrected names".** How many shots the story pass changed. Many of the same swap in one video (Makima to Angel Devil, 44 times) is the same notes problem, caught late.

To look at a shot, open its keyframe: `out/index/<NAME>/` plus the `keyframe.file` of that shot in `shots.json`, or find it on its contact sheet (shot N is on sheet ⌈N / 12⌉).

To fix a confusion, edit that group's cast notes (usually the two characters' "Tell apart from" lines, checked against the sheets), `try` the sheets where it happened, then rerun `finish --only "<group name>"`. That re-captions every video in the group, and only those. Check their reports again.

When the reports look right, commit the research, never media or indexes:

```sh
git add shows/<show>
git commit -m "shows/<show>: manifest, cast notes and synopses"
```

## A show that's already researched

Everything the index needs except the video is in `shows/<show>/`:

1. Put your copies of the videos (and subtitles, if any) at the paths in `shows/<show>/show.toml`, under `media/`. Rename your files to match the manifest rather than editing it, so it stays the same for everyone.
2. `python scene_frames/show.py prepare <show>`. Read the subtitle warnings: subtitles from another release are often offset.
3. `python scene_frames/show.py finish <show>`.

Your release can cut shots slightly differently, so shot numbers can differ from someone else's, but your index is your own. Pass `--only` with group or video names to index just the videos you have.

## Adding videos to a researched show

Add each video to its group in the manifest, write its synopsis, then prepare. If the new episodes bring characters the group's notes don't describe, add their entries to the notes, checked against the new episodes' sheets, and `try` them there. Changing a group's notes re-captions every video already in it when finish runs (about 2.5 minutes an episode), so check those reports again too. If the new videos jump to a new arc, start a new group with its own notes instead.

## show.py

```
python scene_frames/show.py prepare|try|finish|report SHOW [--only NAME ...] [--jobs N] [--sheets N ...]
                            [--media DIR] [--out DIR] [--dry-run]
```

| | |
|---|---|
| `SHOW` | A folder name in `shows/`, or a path to any folder with a `show.toml` |
| `prepare` | index.py with `--no-captions`: shots, stills, contact sheets, transcript, subtitles and the timing check. Lists the research still missing |
| `try` | Captions `--sheets` of one video (`--only NAME`) with its group's current notes and prints them. Caches nothing |
| `finish` | index.py with the group's cast notes, the name report, then story.py with the notes and the synopsis |
| `report` | The name report from each finished `shots.json` |
| `--only NAME ...` | Just these groups (by `name`) or videos (by file name without the extension) |
| `--jobs N` | Videos at once, default 2 |
| `--media DIR`, `--out DIR` | Where videos are and where indexes go; default `media/` and `out/index/` |
| `--dry-run` | Print the index.py and story.py commands instead of running them |

Every index is made with the same options, `--sample 5 --proxy 480 --faces anime`, plus `--subs` and `--lang` from the manifest, so every edit of a video shares one index. Indexing by hand with other options redoes detection for the whole video. The commands are in [pipeline.md](pipeline.md); `--dry-run` prints the exact ones.

It exits 1 when any video failed. The summary shows the end of a failed video's output, and the whole of it is in `out/index/<NAME>/index.log`. Common failures: a file the manifest names isn't there; `claude` isn't on PATH or isn't signed in (run `claude` once in a terminal to sign in); a caption reply stayed invalid after three tries, which a rerun usually gets past.

## Recipe: cast notes

The captioner sees 12 thumbnails at a time and nothing else: no story, no dialogue, no order. All it knows about who's who is the group's cast notes, so they decide whether names are right. They describe what's visible, never what happens; the story belongs in the synopsis.

**The file.** `shows/<show>/cast/<group>.md`. Only the `# ` heading (the title captions are given, such as `# Jujutsu Kaisen season 2, Hidden Inventory / Premature Death arc (MAPPA, 2023)`) and the `## Cast` section reach the captioner. Write nothing else in the file.

```markdown
# Jujutsu Kaisen season 2, Hidden Inventory / Premature Death arc (MAPPA, 2023)

## Cast

Lighting: some scenes are strongly tinted (red, blue, amber or sepia), which shifts hair and clothing colors. Under tinted light, match on shape instead: hairstyle, outfit cut and accessories.

Setting: 2006–2007. Gojo, Geto and Shoko are teenage students at Jujutsu High. Nobody from the 2018 story (Yuji, Nobara, Kenjaku) appears in these episodes.

### Satoru Gojo
- Hair: ...
- Eyes: ...
- Outfit: ...
- Accessories: ...
- Other forms: ...
- Tell apart from: ...
```

**Who goes in.** Everyone who appears on screen in the group's videos with a recognizable design: main and side characters, named curses, devils and creatures, and summoned beings. Unnamed groups that recur get one entry ("Time Vessel Association members"). Leave out anyone who is only talked about. The wiki's episode pages list who appears ("Characters in Order of Appearance"), and so does each synopsis. A character in several groups gets an entry in each file, written for how they look there.

**One entry per character**, headed `### Full Name` in the spelling the wiki uses. Captions use exactly these headings, and code maps short forms ("Gojo", "Rika") to them, so the heading is the name edits search for. A parenthetical after the name is a note, not part of the name: `### Kenjaku (in Suguru Geto's body)` names Kenjaku. Put every age and form of a character inside their one entry (Gojo as a baby, a child, a student and an adult). Give a separate entry only to a different being wearing their body, like Kenjaku in Geto's body, because that's a different person to name.

**Six bullets per entry**, in this order:

- **Hair:** shape first (spiky, bun, braid, bob, a single long bang, a topknot), then length, then color. Give each age or form its own sentence.
- **Eyes:** color and anything distinctive (ringed pupils, a mole under one eye, permanently half-lidded), and what usually covers them (sunglasses, a blindfold, goggles).
- **Outfit:** the usual one, cut and color, then every alternate with where it appears ("open navy aloha shirt with big yellow flowers (Okinawa, ep 3)").
- **Accessories:** anything signature: earrings, a choker, a scar, a halo, a weapon, a cigarette.
- **Other forms:** transformations, monster or armored forms, masks, other ages, and how each looks.
- **Tell apart from:** every character the captioner could mistake them for, each with the one visible difference ("Suguru Geto (same student uniform, but black hair in a bun, baggy pants and earrings)"). Write it from both sides: Gojo's entry names Geto, and Geto's names Gojo.

Lead with identifiers that survive any art style, such as hair shape, scars, signature accessories and silhouette, and put colors and outfits after them. Colors shift with lighting, and fan art, manga and AUs change both colors and clothes, so shape is what still works if this show's index ever takes other media.

**Describe the characters as drawn in this footage.** Wiki appearance sections often describe the manga, or another season. For the Chainsaw Man film they suggested white hair for the Angel Devil, whose hair in the film is auburn. Write what the video shows.

**Two notes above the entries.** A **lighting** line when scenes are strongly tinted (night, flashbacks, red light), saying what to match on instead of color. A **setting** line naming the era and who does *not* appear, when other parts of the show have characters a captioner might guess.

**Steps:**

1. **List the characters** from the synopses and the wiki's episode pages.
2. **Read each character's Appearance section** on the show's Fandom wiki. The wiki refuses direct page fetches, but its MediaWiki API answers through curl. List a page's sections, then fetch one:
   ```sh
   curl -s "https://WIKI.fandom.com/api.php?action=parse&page=PAGE&prop=sections&format=json"
   curl -s "https://WIKI.fandom.com/api.php?action=parse&page=PAGE&prop=wikitext&section=N&format=json"
   ```
   `WIKI` is the wiki's subdomain (`jujutsu-kaisen`, `chainsaw-man`) and `PAGE` the page title with underscores (`Satoru_Gojo`). Leave out `&section=N` for the whole page. To find a page's title, search: `https://WIKI.fandom.com/api.php?action=query&list=search&srsearch=TEXT&format=json`. Image captions and the infobox on the same page often say which outfit goes with which arc.
3. **Draft the entries** in the six-bullet form.
4. **Check every design against this footage.** Open contact sheets from prepare (`out/index/<NAME>/sheets/`) for scenes where the character appears. Each tile is labeled with its shot number and its start and end in seconds. To find a character's scenes, search the subtitle file (or `out/index/<NAME>/cache/transcript.json`) for their name or for a line from the synopsis, and open the sheet whose times cover it. The full-size stills are in `out/index/<NAME>/`, named by shot number and time. For every character, confirm the hair, eyes, outfits and forms as the footage draws them, and correct the draft where it differs. Record new outfits and forms you find. Don't skip this step: the Chainsaw Man notes skipped it, and about 45 captions called the Angel Devil Makima.
5. **Write the "Tell apart from" lines** from what you saw: which characters share a hair color, a uniform or a silhouette, and the one difference visible in a thumbnail.
6. **Add the lighting and setting notes.**
7. **`try` the notes** on sheets with look-alikes (step 5 of the procedure) and fix what comes out wrong.

## Recipe: synopsis

The story pass reads the synopsis with every shot's caption and lines, as text, and splits the shots into scenes. The synopsis tells it what happens and in what order, so it can label scenes and spot names the plot contradicts. It needs the plot in order, in plain words, with who does what. It needs no interpretation.

**The file.** `shows/<show>/synopses/<video>.md`, one per video, named for the episode rather than the file (`s2e01.md`, `jujutsu_kaisen_0.md`), since the episode is the same in every release.

```markdown
# Jujutsu Kaisen season 2, episode 1 (episode 25 overall): "Hidden Inventory" (懐玉, Kaigyoku)

Source: Jujutsu Kaisen Fandom wiki, https://jujutsu-kaisen.fandom.com/wiki/Episode_25 (Summary and Plot Details), read through the MediaWiki API.

Opening song: "Ao no Sumika" by Tatsuya Kitani. Ending song: "Akari" by Soushi Sakiyama.

## Summary
...

## Plot
...

## Characters in order of appearance
...
```

**Steps:**

1. **Find the episode's wiki page.** Fandom wikis usually title episode pages by overall number (`Episode_25`). The search API above finds it from the episode's title.
2. **Fetch its sections** through the API (the commands above): the summary, the plot details (`Plot Details`, `Plot` or `Synopsis`), and the characters in order of appearance. The infobox at the top of the page names the opening and ending songs.
3. **Clean the wikitext into plain markdown**: keep the visible text of links (`[[Satoru Gojo|Gojo]]` becomes `Gojo`), drop templates (`{{...}}`), references (`<ref>...</ref>`), file links and bold or italic quotes. Keep the plot unabridged and in its order. Don't summarize it further: the order and the who-does-what are what the story pass uses.
4. **Write the header**: the title with season, episode and overall numbers and the original title, a `Source:` line with the page's URL and how it was read, and the opening and ending songs. Naming the songs helps the story pass label the credits.
5. **Add `## Characters in order of appearance`** from the wiki's list. The name report checks caption names against the synopsis, so a cameo listed here doesn't show up as a stranger.

**Films**, and episodes the wiki has no plot details for: combine the film's page (premise, music) with the pages for what it adapts (the manga chapters' or arc's plot details), in the order the film tells it. Say so in the `Source:` line, including that the film's own additions and ordering may differ from what follows. `shows/jujutsu_kaisen/synopses/jujutsu_kaisen_0.md` and `shows/chainsaw_man/synopses/reze_arc.md` are examples.

If the wiki has a music table with timestamps, leave it out of the synopsis, but it's a good check of `story.json`'s timing afterwards: the opening and ending scenes should start within a few seconds of it.

## What an edit session uses

An edit session reads `out/index/<NAME>/shots.json` for stills, captions, motion, look and lines, and `story.json` for scenes and names. Each scene in `story.json` lists its shots with corrected names (`shots[].characters`), and keeps the caption's own beside them (`caption_characters`) when the story pass changed them. Those corrected names are the ones to trust. Fields are in [pipeline.md](pipeline.md#story).
