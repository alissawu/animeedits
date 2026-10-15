# Dependencies: tools/requirements.txt. The browser fallback runs the installed Google Chrome headless.
import sys, httpx, trafilatura
from patchright.sync_api import sync_playwright

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"

def looks_blocked(html):
    h = html.lower()
    return len(h) < 2000 or any(s in h for s in ["log in to reddit", "blocked by network security", "captcha", "enable javascript"])

def via_http(url):
    r = httpx.get(url, headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}, follow_redirects=True, timeout=20)
    r.raise_for_status()
    return r.text

def via_browser(url):
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page(user_agent=UA)
        page.goto(url, wait_until="networkidle", timeout=45000)
        html = page.content()
        b.close()
        return html

def fetch(url):
    try:
        html = via_http(url)
        if looks_blocked(html):
            raise RuntimeError("blocked")
    except Exception:
        html = via_browser(url)
    return trafilatura.extract(html, output_format="markdown", include_links=True) or ""

if __name__ == "__main__":
    print(fetch(sys.argv[1]))
