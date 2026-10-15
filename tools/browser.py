"""Read pages through a dedicated, headless, logged-in Chrome over CDP.

The research profile lives outside your normal Chrome data (Chrome refuses remote
debugging on the default one), runs headless, and never opens or focuses a window.
Sign in to sites once with `login`, which is the only command that shows a window.

usage:
  python tools/browser.py login [URL ...]   # headed, once: sign in, then quit Chrome (Cmd+Q)
  python tools/browser.py start             # headless in the background
  python tools/browser.py text URL          # print the rendered page's title, final URL and text
  python tools/browser.py html URL          # print the rendered HTML
  python tools/browser.py stop

Pages that answer with a login wall, a bot challenge or 401/403/429 raise Blocked
(exit 2). They are reported, never worked around.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

CHROME = os.environ.get("ANIMEEDITS_CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PROFILE = Path(os.environ.get("ANIMEEDITS_CHROME_PROFILE", Path.home() / ".animeedits-chrome"))
PORT = int(os.environ.get("ANIMEEDITS_CDP_PORT", "9333"))
ENDPOINT = f"http://127.0.0.1:{PORT}"
LOGIN_PAGES = ["https://www.reddit.com/login", "https://www.tumblr.com/login", "https://www.tiktok.com/login"]
CHALLENGE_TITLES = ("just a moment", "attention required", "verify you are human", "are you a robot",
                    "prove your humanity", "captcha", "security check", "access denied")


class Blocked(RuntimeError):
    pass


def running() -> bool:
    try:
        with urllib.request.urlopen(f"{ENDPOINT}/json/version", timeout=1) as r:
            return bool(json.load(r).get("webSocketDebuggerUrl"))
    except OSError:
        return False


def chrome_args(headless: bool) -> list[str]:
    args = [CHROME, f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check"]
    if headless:
        args += ["--headless=new", f"--remote-debugging-port={PORT}", "--remote-debugging-address=127.0.0.1"]
    return args


def start() -> None:
    if running():
        print(f"already running at {ENDPOINT}")
        return
    PROFILE.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(chrome_args(headless=True) + ["about:blank"], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if running():
            print(f"headless research Chrome at {ENDPOINT}, profile {PROFILE}")
            return
        time.sleep(0.2)
    raise SystemExit("Chrome did not start; is the profile open in a login window? Quit it and retry.")


def stop() -> None:
    if not running():
        print("not running")
        return
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(ENDPOINT)
        browser.new_browser_cdp_session().send("Browser.close")
    print("stopped")


def login(urls: list[str]) -> None:
    if running():
        stop()
        time.sleep(1)
    PROFILE.mkdir(parents=True, exist_ok=True)
    print("Sign in to each tab, then quit this Chrome with Cmd+Q. Your cookies stay in", PROFILE)
    subprocess.run(chrome_args(headless=False) + (urls or LOGIN_PAGES), check=False)


def blocked_reason(status: int | None, url: str, title: str, html: str) -> str | None:
    if status in (401, 403, 429):
        return f"HTTP {status}"
    path = urlparse(url).path.lower()
    if path.startswith("/login") or path.startswith("/account/login"):
        return f"login wall at {url}"
    if any(s in title.lower() for s in CHALLENGE_TITLES):
        return f"challenge page: {title!r}"
    if "blocked by network security" in html.lower():
        return "blocked by network security"
    return None


def render(url: str) -> tuple[str, str, str, str]:
    """Return (final url, title, html, visible text) for a page rendered in the research profile."""
    if not running():
        start()
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(ENDPOINT)
        page = browser.contexts[0].new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            try:
                page.wait_for_load_state("networkidle", timeout=8_000)
            except PlaywrightTimeout:
                pass  # feeds and chat widgets never go idle; the content is there by now
            title, html = page.title(), page.content()
            reason = blocked_reason(response.status if response else None, page.url, title, html)
            if reason:
                raise Blocked(reason)
            return page.url, title, html, page.inner_text("body")
        finally:
            page.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start")
    sub.add_parser("stop")
    sub.add_parser("login").add_argument("urls", nargs="*")
    for name in ("text", "html"):
        sub.add_parser(name).add_argument("url")
    a = ap.parse_args()
    if a.cmd == "start":
        start()
    elif a.cmd == "stop":
        stop()
    elif a.cmd == "login":
        login(a.urls)
    else:
        try:
            final, title, html, text = render(a.url)
        except Blocked as e:
            print(f"blocked: {e}", file=sys.stderr)
            raise SystemExit(2) from None
        except PlaywrightError as e:
            print(f"error: {e}", file=sys.stderr)
            raise SystemExit(1) from None
        print(html if a.cmd == "html" else f"# {title}\n{final}\n\n{text}")


if __name__ == "__main__":
    main()
