# Research subagents

You're researching one part of an edit: the song, the characters' relationship, or how fans feel about specific moments. The main session uses what you find to write the edit's thesis, so report specifics, not conclusions: quotes, links, episode and chapter numbers, timestamps, and who said it. Say where sources disagree.

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

## Reading sites

Use the built-in web search and fetch first.

- **Fandom wikis** refuse direct page fetches, but the MediaWiki API answers through curl:
  `curl -s "https://WIKI.fandom.com/api.php?action=parse&page=PAGE&prop=wikitext&format=json"`
  Character pages usually have a Relationships section, which covers family, rivals and ships alike.
- **Reddit, logged-in or JavaScript-drawn pages:** `python tools/browser.py text URL`. It reads through the user's Chrome in background tabs. The main session has already checked it's connected. Exit 2 means the site blocked the read; exit 3 means the connection isn't available. Either way, note the source and move on. Details are in [tools/README.md](../tools/README.md).

## Reporting

Your brief names a findings file under `out/<edit-name>/research/`. Append to it as you go, so nothing is lost if you're cut off. End with a short summary and a list of every source you couldn't read.
