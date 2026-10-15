# fetch.py

Fetch a web page as markdown. Tries a plain request with a real browser user agent, falls back to headless Chromium if the page looks like a login wall or block page.

```
pip install httpx trafilatura playwright && playwright install chromium
python tools/fetch.py https://old.reddit.com/r/ChainsawMan/
```

For Reddit, old.reddit.com or a post URL ending in .json is the most reliable.

# browser.py

A fallback for pages the built-in web search and fetch can't read: Reddit, TikTok search, Tumblr tag pages and anything else behind a login or drawn by JavaScript. It reads them through Chrome over CDP without touching your screen.

**Default: your own Chrome, with its logins.** Turn it on once in Chrome at `chrome://inspect/#remote-debugging` ("Allow remote debugging for this browser instance"). Chrome asks you to Allow every new connection, so the first read starts a small helper that holds one connection open, and later reads reuse it. That's one Allow per session; the helper exits after 30 idle minutes or on `stop`. Each page opens in a background tab and is closed afterwards, so your active tab, windows and cursor never move. Tested: with `serve --check-focus`, every read reported the tabs you were looking at as still showing.

**`--profile`: a separate headless Chrome** with its own profile in `~/.animeedits-chrome` and only the logins you give it. It's a background-only process (no window, no Dock icon) with no Allow prompt. Use it when your Chrome isn't available.

```
pip install playwright
python tools/browser.py text URL [--profile]   # title, final URL and visible text
python tools/browser.py html URL [--profile]   # rendered HTML, e.g. to pull video links from TikTok search
python tools/browser.py stop                   # stop the helper (and the headless profile Chrome)
python tools/browser.py login [URL ...]        # profile only: sign in once in a window, then Cmd+Q
```

A login wall, bot challenge or 401/403/429 exits with status 2 and a reason. It's reported, never worked around. Exit 3 means no debuggable Chrome, or the Allow was never clicked. While it runs, the helper on localhost:9334 needs a token from `~/.animeedits-browser.json` (mode 600), so web pages can't drive it.

# tiktok.py

Download a TikTok without watermark via the tikwm.com API (works with short /t/ links too).

    python tools/tiktok.py <url> [out.mp4]
