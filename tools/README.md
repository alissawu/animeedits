Dependencies are in `tools/requirements.txt` and installed in the repo's `.venv` (setup is in CLAUDE.md). Browser automation uses patchright, a drop-in fork of Playwright.

# browser.py

A fallback for pages the built-in web search and fetch can't read: Reddit, TikTok search, Tumblr tag pages and anything else behind a login or drawn by JavaScript. It reads them through Chrome over CDP without touching your screen.

**Default: your own Chrome, with its logins.** Turn it on once in Chrome at `chrome://inspect/#remote-debugging` ("Allow remote debugging for this browser instance"). Chrome asks you to Allow every new connection, so the first read starts a small helper that holds one connection open, and later reads reuse it. That's one Allow per session; the helper exits after 30 idle minutes or on `stop`. Each page opens in a background tab and is closed afterwards, so your active tab, windows and cursor never move. Tested: with `serve --check-focus`, every read reported the tabs you were looking at as still showing.

**`--profile`: a separate headless Chrome** with its own profile in `~/.animeedits-chrome` and only the logins you give it. It's a background-only process (no window, no Dock icon) with no Allow prompt. Use it when your Chrome isn't available.

**Checking first:** `status` reports whether reads can use your Chrome without connecting to it, so it never triggers the Allow prompt. It reads the port Chrome writes to `DevToolsActivePort` in its data folder and checks something is listening there; no port or tab needs setting. It prints one of:
- `connected`: a helper is already holding a connection, so reads won't prompt.
- `ready`: remote debugging is on, so the first read will ask you to Allow.
- `off` (exit 3): Chrome isn't running or remote debugging is off.

```
python tools/browser.py status                 # connected / ready / off, never prompts
python tools/browser.py text URL [--profile]   # title, final URL and visible text
python tools/browser.py html URL [--profile]   # rendered HTML, e.g. to pull video links from TikTok search
python tools/browser.py stop                   # stop the helper (and the headless profile Chrome)
python tools/browser.py login [URL ...]        # profile only: sign in once in a window, then Cmd+Q
```

A login wall, bot challenge or 401/403/429 exits with status 2 and a reason. It's reported, never worked around. Exit 3 means no debuggable Chrome (from `status` too), or the Allow was never clicked. While it runs, the helper on localhost:9334 needs a token from `~/.animeedits-browser.json` (mode 600), so web pages can't drive it.

# tiktok.py

Download a TikTok without watermark via the tikwm.com API (works with short /t/ links too).

    python tools/tiktok.py <url> [out.mp4]
