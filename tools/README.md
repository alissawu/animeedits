# fetch.py

Fetch a web page as markdown. Tries a plain request with a real browser user agent, falls back to headless Chromium if the page looks like a login wall or block page.

```
pip install httpx trafilatura playwright && playwright install chromium
python tools/fetch.py https://old.reddit.com/r/ChainsawMan/
```

For Reddit, old.reddit.com or a post URL ending in .json is the most reliable.

# browser.py

Read JavaScript-heavy or logged-in pages (Tumblr tags, Reddit, TikTok search) through a dedicated headless Chrome over CDP. The profile lives in `~/.animeedits-chrome`, separate from your normal Chrome (Chrome refuses remote debugging on that). It runs as a background-only process: no window, no Dock icon, and it never takes focus.

```
pip install playwright
python tools/browser.py login          # once: opens Reddit, Tumblr and TikTok sign-in tabs; sign in, then Cmd+Q
python tools/browser.py text URL       # starts headless Chrome if needed, prints title, final URL and page text
python tools/browser.py html URL
python tools/browser.py stop
```

A login wall, bot challenge or 401/403/429 exits with status 2 and a reason. It's reported, never worked around. Logged out, Reddit answers with a challenge and TikTok search shows no results; Tumblr tag pages work. The CDP port (9333) listens on localhost only, but any local program could drive this logged-in browser while it runs. Stop it when you're done.

# tiktok.py

Download a TikTok without watermark via the tikwm.com API (works with short /t/ links too).

    python tools/tiktok.py <url> [out.mp4]
