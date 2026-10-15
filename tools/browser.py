"""Read rendered, logged-in pages over CDP without touching the user's screen.

Default: the user's own Chrome, with its logins, once they have enabled
chrome://inspect/#remote-debugging ("Allow remote debugging for this browser instance").
Chrome asks the user to Allow every new connection, so a small helper (`serve`) holds
one connection open and every read goes through it: one Allow per session. Each page
opens in a background tab, so the active tab, window and cursor never move, and the
tab is closed when done. The helper exits after 30 idle minutes or on `stop`.

--profile: a separate headless Chrome with its own profile (~/.animeedits-chrome)
and only the logins given to it. It's a background-only process with no window and
no Allow prompt. Use it when the user's Chrome isn't available.

usage:
  python tools/browser.py text URL [--profile]   # title, final URL and visible text
  python tools/browser.py html URL [--profile]   # rendered HTML
  python tools/browser.py stop                   # stop the helper (and the headless profile Chrome)
  python tools/browser.py login [URL ...]        # profile only: sign in once (opens a window), then Cmd+Q

`text` and `html` start the helper when it isn't running. For the user's Chrome that
shows an Allow prompt in Chrome, so tell the user before the first read.

Exit codes: 2 when a page is a login wall, bot challenge or 401/403/429 (reported,
never worked around); 3 when no debuggable Chrome is available or the user did not Allow.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from playwright.sync_api import Browser, Page, sync_playwright
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeout

USER_CHROME_DATA = Path(os.environ.get("ANIMEEDITS_USER_CHROME_DATA",
                                       Path.home() / "Library/Application Support/Google/Chrome"))
CHROME = os.environ.get("ANIMEEDITS_CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
PROFILE = Path(os.environ.get("ANIMEEDITS_CHROME_PROFILE", Path.home() / ".animeedits-chrome"))
PROFILE_PORT = int(os.environ.get("ANIMEEDITS_CDP_PORT", "9333"))
PROFILE_ENDPOINT = f"http://127.0.0.1:{PROFILE_PORT}"
HELPER_PORT = int(os.environ.get("ANIMEEDITS_HELPER_PORT", "9334"))
STATE = Path.home() / ".animeedits-browser.json"
LOG = Path.home() / ".animeedits-browser.log"
IDLE_SECONDS = 30 * 60
ALLOW_WAIT_SECONDS = 180
LOGIN_PAGES = ["https://www.reddit.com/login", "https://www.tiktok.com/login"]
CHALLENGE_TITLES = ("just a moment", "attention required", "verify you are human", "are you a robot",
                    "prove your humanity", "captcha", "security check", "access denied")
NO_CHROME = ("No Chrome with remote debugging is available. Ask the user to open "
             "chrome://inspect/#remote-debugging in their Chrome and check \"Allow remote debugging for "
             "this browser instance\", or rerun with --profile to use the separate headless research profile.")


class Blocked(RuntimeError):
    pass


# --- endpoints -------------------------------------------------------------------------

def user_chrome_endpoint() -> str | None:
    """WebSocket endpoint of the user's Chrome, if remote debugging is on and it's running."""
    try:
        port, path = (USER_CHROME_DATA / "DevToolsActivePort").read_text().split()[:2]
        with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
            pass  # the file outlives Chrome, so check something is listening
    except (OSError, ValueError):
        return None
    return f"ws://127.0.0.1:{port}{path}"


def profile_running() -> bool:
    try:
        with urllib.request.urlopen(f"{PROFILE_ENDPOINT}/json/version", timeout=1) as r:
            return bool(json.load(r).get("webSocketDebuggerUrl"))
    except OSError:
        return False


def chrome_args(headless: bool) -> list[str]:
    args = [CHROME, f"--user-data-dir={PROFILE}", "--no-first-run", "--no-default-browser-check"]
    if headless:
        args += ["--headless=new", f"--remote-debugging-port={PROFILE_PORT}", "--remote-debugging-address=127.0.0.1"]
    return args


def start_profile_chrome() -> None:
    if profile_running():
        return
    PROFILE.mkdir(parents=True, exist_ok=True)
    subprocess.Popen(chrome_args(headless=True) + ["about:blank"], stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        if profile_running():
            return
        time.sleep(0.2)
    raise SystemExit("Profile Chrome did not start; is it open in a login window? Quit it and retry.")


# --- reading a page --------------------------------------------------------------------

def blocked_reason(status: int | None, url: str, title: str, html: str) -> str | None:
    if status in (401, 403, 429):
        return f"HTTP {status}"
    path = urllib.parse.urlparse(url).path.lower()
    if path.startswith("/login") or path.startswith("/account/login"):
        return f"login wall at {url}"
    if any(s in title.lower() for s in CHALLENGE_TITLES):
        return f"challenge page: {title!r}"
    if "blocked by network security" in html.lower():
        return "blocked by network security"
    return None


def visible_tabs(browser: Browser, exclude: Page | None = None) -> set[str]:
    """Target IDs of tabs currently showing in a window (one per window)."""
    shown: set[str] = set()
    for page in browser.contexts[0].pages:
        if page is exclude:
            continue
        try:
            if page.evaluate("document.visibilityState") == "visible":
                shown.add(str(page.context.new_cdp_session(page).send("Target.getTargetInfo")["targetInfo"]["targetId"]))
        except PlaywrightError:
            pass  # discarded or crashed tabs can't run script
    return shown


def read_page(browser: Browser, url: str, background: bool, check_focus: bool) -> dict[str, Any]:
    context = browser.contexts[0]
    before = visible_tabs(browser) if check_focus else set()
    if background:
        with context.expect_page(timeout=15_000) as opened:
            browser.new_browser_cdp_session().send("Target.createTarget", {"url": "about:blank", "background": True})
        page = opened.value
    else:
        page = context.new_page()
    try:
        response = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        try:
            page.wait_for_load_state("networkidle", timeout=8_000)
        except PlaywrightTimeout:
            pass  # feeds and chat widgets never go idle; the content is there by now
        title, html = page.title(), page.content()
        result: dict[str, Any] = {"url": page.url, "title": title, "html": html, "text": page.inner_text("body")}
        if check_focus:
            result["focus_kept"] = before <= visible_tabs(browser, exclude=page)
        reason = blocked_reason(response.status if response else None, page.url, title, html)
        if reason:
            raise Blocked(reason)
        return result
    finally:
        page.close()


# --- the helper ------------------------------------------------------------------------

def serve(use_profile: bool, check_focus: bool) -> None:
    """Hold one CDP connection and answer page reads from local clients."""
    if use_profile:
        start_profile_chrome()
        endpoint: str | None = PROFILE_ENDPOINT
    else:
        endpoint = user_chrome_endpoint()
    if endpoint is None:
        raise SystemExit(NO_CHROME)
    token = secrets.token_urlsafe(24)
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(endpoint, timeout=ALLOW_WAIT_SECONDS * 1000)
        state = {"pid": os.getpid(), "port": HELPER_PORT, "token": token, "profile": use_profile}
        last_used = time.monotonic()
        quitting = False

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def reply(self, status: int, body: dict[str, Any]) -> None:
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                nonlocal last_used, quitting
                if not secrets.compare_digest(self.headers.get("X-Token", ""), token):
                    self.reply(403, {"error": "bad token"})
                    return
                route = urllib.parse.urlparse(self.path)
                if route.path == "/health":
                    self.reply(200, {"ok": True, "profile": use_profile})
                elif route.path == "/quit":
                    quitting = True
                    self.reply(200, {"ok": True})
                elif route.path == "/read":
                    last_used = time.monotonic()
                    url = urllib.parse.parse_qs(route.query).get("url", [""])[0]
                    try:
                        self.reply(200, read_page(browser, url, background=not use_profile, check_focus=check_focus))
                    except Blocked as e:
                        self.reply(409, {"blocked": str(e)})
                    except PlaywrightError as e:
                        self.reply(502, {"error": str(e)})
                else:
                    self.reply(404, {"error": "unknown route"})

        server = HTTPServer(("127.0.0.1", HELPER_PORT), Handler)
        server.timeout = 5
        STATE.write_text(json.dumps(state))
        STATE.chmod(0o600)
        try:
            while not quitting and browser.is_connected() and time.monotonic() - last_used < IDLE_SECONDS:
                server.handle_request()
        finally:
            server.server_close()
            STATE.unlink(missing_ok=True)
        # Leaving the with-block disconnects; the user's Chrome keeps running.


def helper_state() -> dict[str, Any] | None:
    try:
        state: dict[str, Any] = json.loads(STATE.read_text())
    except (OSError, ValueError):
        return None
    try:
        call(state, "/health", timeout=3)
    except OSError:
        return None
    return state


def call(state: dict[str, Any], path: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(f"http://127.0.0.1:{state['port']}{path}", headers={"X-Token": state["token"]})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body: dict[str, Any] = json.load(r)
    except urllib.error.HTTPError as e:
        body = json.load(e)
    return body


def ensure_helper(use_profile: bool) -> dict[str, Any]:
    state = helper_state()
    if state is not None:
        if state["profile"] != use_profile:
            raise SystemExit("A helper for the other browser is running; run `python tools/browser.py stop` first.")
        return state
    if not use_profile and user_chrome_endpoint() is None:
        print(NO_CHROME, file=sys.stderr)
        raise SystemExit(3)
    if not use_profile:
        print("Connecting to your Chrome: click Allow in Chrome if it asks.", file=sys.stderr)
    with LOG.open("a") as log:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve"] + (["--profile"] if use_profile else []),
                         stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    for _ in range(ALLOW_WAIT_SECONDS * 2):
        state = helper_state()
        if state is not None:
            return state
        time.sleep(0.5)
    print(f"Chrome never allowed the connection. See {LOG}.", file=sys.stderr)
    raise SystemExit(3)


def stop() -> None:
    state = helper_state()
    if state is not None:
        call(state, "/quit", timeout=10)
        print("helper stopped")
    if profile_running():
        with sync_playwright() as p:
            p.chromium.connect_over_cdp(PROFILE_ENDPOINT).new_browser_cdp_session().send("Browser.close")
        print("profile Chrome stopped")


def login(urls: list[str]) -> None:
    if profile_running():
        stop()
        time.sleep(1)
    PROFILE.mkdir(parents=True, exist_ok=True)
    print("Sign in to each tab, then quit this Chrome with Cmd+Q. Your cookies stay in", PROFILE)
    subprocess.run(chrome_args(headless=False) + (urls or LOGIN_PAGES), check=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stop")
    sub.add_parser("login").add_argument("urls", nargs="*")
    serve_cmd = sub.add_parser("serve", help="hold one connection open (started automatically by text/html)")
    serve_cmd.add_argument("--profile", action="store_true")
    serve_cmd.add_argument("--check-focus", action="store_true",
                           help="report whether the user's showing tabs stayed showing during each read")
    for name in ("text", "html"):
        cmd = sub.add_parser(name)
        cmd.add_argument("url")
        cmd.add_argument("--profile", action="store_true", help="use the separate headless research profile")
    a = ap.parse_args()
    if a.cmd == "stop":
        stop()
    elif a.cmd == "login":
        login(a.urls)
    elif a.cmd == "serve":
        serve(a.profile, a.check_focus)
    else:
        body = call(ensure_helper(a.profile), "/read?" + urllib.parse.urlencode({"url": a.url}), timeout=120)
        if "blocked" in body:
            print(f"blocked: {body['blocked']}", file=sys.stderr)
            raise SystemExit(2)
        if "error" in body:
            print(f"error: {body['error']}", file=sys.stderr)
            raise SystemExit(1)
        focus = "" if "focus_kept" not in body else f"\nfocus_kept: {body['focus_kept']}"
        print(body["html"] if a.cmd == "html" else f"# {body['title']}\n{body['url']}{focus}\n\n{body['text']}")


if __name__ == "__main__":
    main()
