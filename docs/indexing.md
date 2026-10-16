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
| 3. Prepare | `python scene_frames/show.py prepare <show>` | No | ~13 min for two episodes run together |
| 4. Research | `shows/<show>/cast/*.md`, `shows/<show>/synopses/*.md` | You and subagents | Start it while prepare runs |
| 5. Finish | `python scene_frames/show.py finish <show>` | Haiku for captions, then Opus for the story | ~3 min per episode, ~11 per film |
| 6. Check, fix, commit | the summary from step 5 | | |

Prepare is CPU work on your machine: a pass over a small copy of the video to find the cuts, a pass over the full video for stills and motion, and Whisper alongside both. Research needs none of it at first, so start the research as soon as prepare starts (step 4 says what can begin right away). finish uses Claude twice. Haiku looks at each contact sheet and captions it, about 15 s a sheet with 10 at a time. Then the story pass reads text only, the synopsis with every caption and line, and places each shot in the plot, which takes a stronger model (Opus); its requests run one after another, since each starts where the last one's scenes ended.

### 1. Add the videos

Put the files in `media/<show>/`, in folders if you like (`media/jujutsu_kaisen/season_2/`). Use one lowercase folder name per show, such as `chainsaw_man`. Naming `shows/<show>/` the same keeps things easy to find, but only the manifest's paths matter. Give every video a different file name across all shows, since the name is also its index folder's name.

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

For each video this finds the shots, saves three stills per shot, draws numbered contact sheets of the keyframes (`out/index/<NAME>/sheets/sheet_0001.jpg`, 12 shots each), transcribes the dialogue with Whisper and reads the subtitles. It uses no Claude. The full-size stills (`out/index/<NAME>/scene_0001_….jpg`) start appearing once the cuts are found, about 4 minutes in; the contact sheets are drawn at the very end. Two videos run at a time (`--jobs N` changes it; past 3, each gets slower), and every stage is cached, so rerunning the command after a failure or Ctrl-C picks up where it stopped. Each video's full output goes to `out/index/<NAME>/index.log`.

The summary at the end lists, for each video, its dialogue language and where that came from, its subtitle cue count, the subtitle timing check, any warnings and its shot count. Then it lists the research still missing: a cast-notes file without `### ` entries, or an empty synopsis, counts as missing. Look at the warnings now:

- **"the subtitles start 2.3 s late against the dialogue"**: the file was timed for another release of the video, with a different intro or recap. Shift it with the `ffmpeg -itsoffset` command the warning gives, point `subs` at the shifted file, and rerun prepare.
- **"the subtitles don't line up with the dialogue at any shift"**: they're probably for another episode, or another cut. Check a few cues against the video.
- **"language auto-detect, since the audio track's tag is und"**: Whisper picks the language itself, which is usually right. If the transcript comes out in the wrong language, set `lang`.

### 4. Research

Write one cast-notes file per group and one synopsis per video, at the paths the manifest gives. The two recipes below are the whole method. `shows/jujutsu_kaisen/cast/s2_shibuya.md` and `s2_hidden_inventory.md` and the synopses beside them are worked examples.

In order, overlapping prepare:

1. **While prepare runs:** write the synopses (they need only the wiki), then fetch every character's Appearance section from the wiki and draft the cast notes from it. The synopses' character lists tell you who the notes need.
2. **Once the stills appear:** start checking the drafts against the footage.
3. **Once the contact sheets exist:** skim every sheet of every video once, for characters on screen that no list names (memories, flashbacks, imagined shots, the opening and ending songs, the next-episode preview), and finish checking designs.

Subagents work well for this, one per episode or batch of characters, as long as they report the specifics (the wiki's wording, sheet and shot numbers, what each still shows) rather than conclusions. A subagent runs your session's model unless you give it another. Fetching and cleaning wiki text is mechanical, so a smaller model (Sonnet) is plenty for synopses and wiki fetches. Checking designs against the footage is where a mistake costs a full re-caption, so keep that on your strongest model.

### 5. Finish

```sh
python scene_frames/show.py finish <show>
```

For each video this captions every contact sheet with Haiku through `claude -p` on your Claude Code sign-in (no API key), using the group's cast notes, and writes `shots.json`. It then reports the names the captions give and runs the story pass, which splits the shots into scenes with Opus and corrects names the dialogue contradicts (`story.json`). The story pass stays on Opus on purpose: on a film, Sonnet's name corrections came out noticeably worse, and those corrections are what edits trust. Captions are cached per sheet and keyed by the cast notes' text, so finish re-captions a video only when its group's notes change. The story pass reruns when a caption or the synopsis changes. Neither is deterministic: a rerun shifts some "possibly"s and moves a few scene splits. finish refuses a video whose cast notes or synopsis aren't ready, and does the rest.

Before captioning a whole show with new notes, try them on a few sheets:

```sh
python scene_frames/show.py try <show> --only <NAME> --sheets 12 40 41
```

It captions those sheets the way finish would, with the notes as they are now, prints each shot's names and caption, and caches nothing. Pick sheets where look-alikes appear, the ones you found while checking the notes. Each sheet is one Haiku call, and they all run at once: four to six sheets take about 30 s. If a character comes out misnamed, fix the notes and try again. A full film is ~180 sheets and a 24-minute episode ~45, so this is much cheaper than finding the problem after finish. Captions vary a little from run to run, mostly in which names get "possibly", so judge the notes by what most shots do, not by one.

### 6. Check the report, fix, commit

finish's summary shows, under each video, the name report, the story pass's result and what the story pass corrected. For the Reze film:

```
names in the captions of 2112 shots: Denji 478 (332 possibly), Reze 297 (195 possibly), Aki Hayakawa 110 (84
    possibly), Angel Devil 105 (54 possibly), Makima 86 (57 possibly), Power 63 (21 possibly), Beam 22 (22
    possibly), Typhoon Devil 9 (9 possibly), Master 7 (4 possibly), Kishibe 6 (6 possibly), Pochita 6 (1
    possibly)
84 scenes, 40 flagged caption names, 50 shots with corrected names -> out/index/.../story.json
story corrections: Angel Devil → Aki's red-haired Division 2 coworker 9, Reze → nobody 7, Kishibe → serial
    killer 4, Angel Devil → nobody 4, Makima → Power 4, ...
```

`python scene_frames/show.py report <show>` prints it all again at any time, with no Claude. Read it against what you know of each video:

- **Counts that don't fit the story.** A side character named in hundreds of shots, or a main character barely named, means the notes let the captioner confuse two people. In the Reze film the first notes described the Angel Devil with white hair. In the film his hair is auburn, so Haiku called him Makima, the other red-haired character: Makima 133, Angel Devil 12, in a film where he's the bigger part. With the hair fixed: Angel Devil 105, Makima 86.
- **"not in the cast notes".** The captioner named someone the notes don't describe. A real cameo is fine; add an entry if they matter to edits. A guess from another arc means a look-alike's "Tell apart from" line or the setting line is missing. Check each one: in season 2 episode 5, a 2006 episode, "not in the cast notes: Nobara Kugisaki 2, Yuji Itadori 2" turned out to be a real 2018 scene, so that one was fine.
- **"not in the synopsis".** Named, but the plot never mentions them. That's a memory or flashback, a cameo, the opening or ending song, or the next-episode preview, or else a wrong name. Add real appearances to the synopsis's character list with where they appear.
- **"story corrections".** Each swap the story pass made from the caption's name to who the scene's lines and the synopsis say it is, with how many shots. Many of the same swap is the notes' problem, caught late. A swap to someone with no entry, like "Aki's red-haired Division 2 coworker" above, means the notes are missing a look-alike: here a red-haired Division 2 hunter, with no halo or wings, whom the captioner took for the Angel Devil. Add the missing character, with "Tell apart from" lines on both sides. A swap to a character the footage contradicts means the notes are wrong (the hot-spring example above).
- **"possibly".** The captioner hedges on small, dark or side-on figures, and is often right anyway, so a high "possibly" count isn't by itself a problem.

To check a name, `python scene_frames/show.py report <show> --only <NAME> --name "Angel Devil"` lists every shot whose caption gives it, with its keyframe still and any story correction. Shot N is also on contact sheet ⌈N / 12⌉.

To fix a confusion, edit that group's cast notes (usually the two characters' "Tell apart from" lines, checked against the stills), `try` the sheets where it happened, then rerun `finish --only "<group name>"`. That re-captions every video in the group, and only those. Check their reports again.

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
                            [--name NAME] [--media DIR] [--out DIR] [--dry-run]
```

| | |
|---|---|
| `SHOW` | A folder name in `shows/`, or a path to any folder with a `show.toml` |
| `prepare` | index.py with `--no-captions`: shots, stills, contact sheets, transcript, subtitles and the timing check. Lists the research still missing |
| `try` | Captions `--sheets` of one video (`--only NAME`) with its group's current notes and prints them. Caches nothing |
| `finish` | index.py with the group's cast notes, the name report, then story.py with the notes and the synopsis |
| `report` | The name report and story corrections from each finished index. With `--name NAME`, every shot whose caption gives that name instead, with its keyframe and any story correction |
| `--only NAME ...` | Just these groups (by `name`) or videos (by file name without the extension) |
| `--jobs N` | Videos at once, default 2 |
| `--media DIR`, `--out DIR` | Where videos are and where indexes go; default `media/` and `out/index/` |
| `--dry-run` | Print the index.py and story.py commands instead of running them |

Every index is made with the same options, `--sample 5 --proxy 480 --faces anime`, plus `--subs` and `--lang` from the manifest, so every edit of a video shares one index. Indexing by hand with other options redoes detection for the whole video. The commands are in [pipeline.md](pipeline.md); `--dry-run` prints the exact ones.

It exits 1 when any video failed. The summary shows the end of a failed video's output, and the whole of it is in `out/index/<NAME>/index.log`. Common failures: a file the manifest names isn't there; `claude` isn't on PATH or isn't signed in (run `claude` once in a terminal to sign in); a caption reply stayed invalid after three tries, which a rerun usually gets past.

## Recipe: cast notes

The captioner sees 12 thumbnails at a time and nothing else: no story, no dialogue, no order. All it knows about who's who is the group's cast notes, so they decide whether names are right. They describe what's visible, never what happens; the story belongs in the synopsis.

The story pass reads the notes too, as evidence for who is who, and it trusts them. A wrong claim about a scene therefore does double damage. One draft said Hanami lies in a misty hot spring in a memory, but the full-size stills show Mahito there. Haiku captioned those shots correctly as Mahito, and the story pass then "corrected" them to Hanami, quoting the notes. Check every scene-specific claim on the full-size still, not the thumbnail.

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

**Who goes in.** Everyone who appears on screen in the group's videos with a recognizable design: main and side characters, named curses, devils and creatures, and summoned beings, including anyone seen only in a memory, an imagined shot, the opening or ending song or the next-episode preview. Unnamed groups that recur get one entry ("Time Vessel Association members"). Leave out anyone who is only talked about. The wiki's episode pages list who appears ("Characters in Order of Appearance"), and so does each synopsis, but those lists miss characters seen only in memories, imagined shots, credits and previews. That's why research step 3 skims every sheet. A character in several groups gets an entry in each file, written for how they look there. Some things aren't entries:

- **A named object** that fills many shots, such as the Prison Realm, goes in the entries of the characters who use it, never in its own entry, because the captioner would name it as a person.
- **Crowds of unnamed people**, such as Halloween civilians in costumes, go in the setting line, saying they are not any of the characters, so they're never named.
- **A character's devices, puppets, chibi gag versions and imagined versions** go in that character's "Other forms".

**One entry per character**, headed `### Name`. Captions use exactly these headings, and code maps short forms ("Gojo", "Rika") to them, so the heading is the name edits search for. Use the full name in the wiki's spelling (`### Satoru Gojo`). For a character the footage and fans call by another name, head the entry with that name (`### Mechamaru`, though the wiki's page is "Kokichi Muta").

A parenthetical after the heading's name is read two ways:

- **Written as names**, every word capitalized: other names the character goes by. `### Café owner (Master)` and `### Mechamaru (Kokichi Muta)` make captions that say "Master" or "Kokichi Muta" count as that entry. Separate several with `/`.
- **Written in lowercase:** a note, not part of the name. `### Kenjaku (in Suguru Geto's body)` names only Kenjaku.

Put every age and form of a character inside their one entry (Gojo as a baby, a child, a student and an adult). Give a separate entry only to a different being wearing their body, like Kenjaku in Geto's body, because that's a different person to name.

**Six bullets per entry**, in this order:

- **Hair:** shape first (spiky, bun, braid, bob, a single long bang, a topknot), then length, then color. Give each age or form its own sentence.
- **Eyes:** color and anything distinctive (ringed pupils, a mole under one eye, permanently half-lidded), and what usually covers them (sunglasses, a blindfold, goggles).
- **Outfit:** the usual one, cut and color, then every alternate with where it appears ("open navy aloha shirt with big yellow flowers (Okinawa, ep 3)").
- **Accessories:** anything signature: earrings, a choker, a scar, a halo, a weapon, a cigarette.
- **Other forms:** transformations, monster or armored forms, masks, other ages, and how each looks.
- **Tell apart from:** every character the captioner could mistake them for, each with the one visible difference ("Suguru Geto (same student uniform, but black hair in a bun, baggy pants and earrings)"). Write it from both sides: Gojo's entry names Geto, and Geto's names Gojo.

Lead with identifiers that survive any art style, such as hair shape, scars, signature accessories and silhouette, and put colors and outfits after them. Colors shift with lighting, and fan art, manga and AUs change both colors and clothes, so shape is what still works if this show's index ever takes other media.

**Describe the characters as drawn in this footage.** Wiki appearance sections often describe the manga, or another season. For the Chainsaw Man film they suggested white hair for the Angel Devil, whose hair in the film is auburn. Write what the video shows.

**Two notes above the entries.** A **lighting** line when scenes are strongly tinted (night, flashbacks, red light), saying what to match on instead of color. Include the opening and ending songs' looks and the preview's style (red silhouettes, white line art on black), since the captioner guesses at those. A **setting** line naming the era and who does *not* appear, when other parts of the show have characters a captioner might guess, and which characters appear only in memories, the opening or one episode.

**Steps:**

1. **List the characters** from the synopses and the wiki's episode pages.
2. **Read each character's Appearance section** on the show's Fandom wiki. The wiki refuses direct page fetches, but its MediaWiki API answers through curl. To fetch many characters' Appearance sections at once:
   ```sh
   for page in Satoru_Gojo Choso Ui_Ui; do
     curl -s "https://WIKI.fandom.com/api.php?action=parse&page=$page&prop=wikitext&format=json" |
       python3 -c 'import json, re, sys; text = json.load(sys.stdin)["parse"]["wikitext"]["*"]; m = re.search(r"^==\s*Appearance\s*==\s*$(.*?)(?=^==[^=])", text, re.S | re.M); print(m[1].strip() if m else "(no Appearance section)")' > "$page.txt"
   done
   ```
   `WIKI` is the wiki's subdomain (`jujutsu-kaisen`, `chainsaw-man`) and each page is a title with underscores. To list a page's sections, use `prop=sections` instead of `prop=wikitext`; to fetch one, add `&section=N`. To find a page's title, search: `https://WIKI.fandom.com/api.php?action=query&list=search&srsearch=TEXT&format=json`. Image captions and the infobox on the same page often say which outfit goes with which arc.
3. **Draft the entries** in the six-bullet form. Steps 1–3 need no footage, so do them while prepare runs.
4. **Check every design against this footage.** The full-size stills appear in `out/index/<NAME>/` a few minutes into prepare, three per shot (`scene_0123_45.67s.jpg`, plus `_first_` and `_last_` ones), and the contact sheets (`sheets/`) at the end. Each tile on a sheet is labeled with its shot number and its start and end in seconds. To find a character's scenes, search the subtitle file (or `out/index/<NAME>/cache/transcript.json`) for their name or for a line from the synopsis, and open the sheet whose times cover it. For every character, confirm the hair, eyes, outfits and forms as the footage draws them, and correct the draft where it differs. Record new outfits and forms you find. Wikis disagree with the footage often: in two Shibuya episodes the wiki was wrong about eight characters' hair or clothes. Don't skip this step: the Chainsaw Man notes skipped it, and about 45 captions called the Angel Devil Makima.
   - A sheet shows one keyframe per shot. A fast montage can pack several images into one shot, so in montages and memory sequences look at the `_first_` and `_last_` stills too.
   - Check any claim about a particular scene ("lying in a hot spring in a memory") on the full-size still, since the story pass will rely on it.
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
3. **Clean the wikitext into plain markdown**: keep the visible text of links (`[[Satoru Gojo|Gojo]]` becomes `Gojo`), drop templates (`{{...}}`), references (`<ref>...</ref>`), file links and bold or italic quotes. When a link's text names someone other than its page would suggest, keep both: `[[Kenjaku|Pseudo-Geto]]` becomes `Pseudo-Geto (Kenjaku)`, so neither the story pass nor the name report reads it as Suguru Geto. Keep the plot unabridged and in its order. Don't summarize it further: the order and the who-does-what are what the story pass uses.
4. **Write the header**: the title with season, episode and overall numbers and the original title, a `Source:` line with the page's URL and how it was read, and the opening and ending songs. Naming the songs helps the story pass label the credits.
5. **Add `## Characters in order of appearance`** from the wiki's list, using the cast notes' heading names, so one character never goes by two names in `story.json` (write `Mechamaru (Kokichi Muta, by proxy)`, not the wiki's `Kokichi Muta (Proxy)`). Keep the wiki's qualifiers, such as "(Mentioned)" and "(Flashback)". Add the characters research step 3 found on screen that the wiki's list misses, with where they appear ("Riko Amanai (Gojo's memories)", "Sukuna (opening song)"). The name report checks caption names against the synopsis, so a cameo listed here doesn't show up as a stranger.

**Films**, and episodes the wiki has no plot details for: combine the film's page (premise, music) with the pages for what it adapts (the manga chapters' or arc's plot details), in the order the film tells it. Say so in the `Source:` line, including that the film's own additions and ordering may differ from what follows. `shows/jujutsu_kaisen/synopses/jujutsu_kaisen_0.md` and `shows/chainsaw_man/synopses/reze_arc.md` are examples.

If the wiki has a music table with timestamps, leave it out of the synopsis, but it's a good check of `story.json`'s timing afterwards: the opening and ending scenes should start within a few seconds of it.

## From inside an edit

An edit session that needs a video that isn't indexed yet shouldn't stop the edit for it. Index it in the background and keep working with the indexes that exist.

- **The video is in its show's manifest already:** run `python scene_frames/show.py prepare <show> --only <NAME>`, then `finish`, in the background.
- **It isn't** (a new episode, or a show with no `shows/` folder): hand its indexing research to a subagent briefed with this doc, the video's path and the show. Ask it to follow the new-show or adding-videos procedure through finish, and to report the name report and story corrections. Use a subagent rather than a new session, so this works outside the desktop app too.

Don't write cast notes inline, in a hurry, to get on with the edit. That's how the Reze film's Angel Devil came out as Makima. When the subagent reports, read its name report before using the new index.

## What an edit session uses

An edit session reads `out/index/<NAME>/shots.json` for stills, captions, motion, look and lines, and `story.json` for scenes and names. Each scene in `story.json` lists its shots with corrected names (`shots[].characters`), and keeps the caption's own beside them (`caption_characters`) when the story pass changed them. Those corrected names are the ones to trust. Don't drop names marked unsure (`certain: false`): the captioner hedges on small, dark or side-on figures even when it's right. In one Shibuya episode, Takuma Ino was named in 24 shots, all unsure and all correct. Fields are in [pipeline.md](pipeline.md#story).
