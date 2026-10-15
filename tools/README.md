# fetch.py

Fetch a web page as markdown. Tries a plain request with a real browser user agent, falls back to headless Chromium if the page looks like a login wall or block page.

```
pip install httpx trafilatura playwright && playwright install chromium
python tools/fetch.py https://old.reddit.com/r/ChainsawMan/
```

For Reddit, old.reddit.com or a post URL ending in .json is the most reliable.

# tiktok.py

Download a TikTok without watermark via the tikwm.com API (works with short /t/ links too).

    python tools/tiktok.py <url> [out.mp4]
