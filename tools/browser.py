"""Read rendered, logged-in pages over CDP without touching the user's screen.

Default: the user's own Chrome, with its logins, once they have enabled
chrome://inspect/#remote-debugging ("Allow remote debugging for this browser instance").
Chrome asks the user to Allow every new connection, so a small helper (`serve`) holds
one connection open and every read goes through it: one Allow per session. The helper
takes its port before it connects, so when several reads start at once only one helper
ever connects; the rest exit without opening a connection (or a prompt). It answers
reads from several clients at once, each page in its own background tab, so the active
tab, window and cursor never move, and each tab is closed when done. The helper exits
after 30 idle minutes or on `stop`.

--profile: a separate headless Chrome with its own profile (~/.animeedits-chrome)
and only the logins given to it. It's a background-only process with no window and
no Allow prompt. Use it when the user's Chrome isn't available.

usage:
  python tools/browser.py status                 # can reads use the user's Chrome? never prompts
  python tools/browser.py text URL [--profile]   # title, final URL and visible text
  python tools/browser.py html URL [--profile]   # rendered HTML
  python tools/browser.py stop                   # stop the helper (and the headless profile Chrome)
  python tools/browser.py login [URL ...]        # profile only: sign in once (opens a window), then Cmd+Q

`text` and `html` start the helper when it isn't running. For the user's Chrome that
shows an Allow prompt in Chrome, so tell the user before the first read. `status` checks
without connecting and prints one of: `connected` (a helper is running; reads won't
prompt), `connecting` (a helper is waiting for the user to click Allow), `ready` (the
first read will prompt) or `off` (exit 3).

Exit codes: 2 when a page is a login wall, bot challenge or 401/403/429 (reported,
never worked around); 3 when no debuggable Chrome is available or the user did not Allow.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import errno
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from patchright.async_api import Browser, Page, Playwright, async_playwright
from patchright.async_api import Error as PlaywrightError
from patchright.async_api import TimeoutError as PlaywrightTimeout
from patchright.sync_api import sync_playwright

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
READ_SECONDS = 90  # one page, from opening its tab to closing it
MAX_TABS = 4  # background tabs open at once in the user's Chrome; further reads wait their turn
START_WAIT_SECONDS = 90  # for a helper to answer; one mid-read can be slow until it finishes
READ_WAIT_SECONDS = 15 * 60  # a client's read may wait for the Allow and for other reads' tabs
LOGIN_PAGES = ["https://www.reddit.com/login", "https://www.tiktok.com/login"]
CHALLENGE_TITLES = ("just a moment", "attention required", "verify you are human", "are you a robot",
                    "prove your humanity", "captcha", "security check", "access denied")
NO_CHROME = ("No Chrome with remote debugging is available. Ask the user to open "
             "chrome://inspect/#remote-debugging in their Chrome and check \"Allow remote debugging for "
             "this browser instance\", or rerun with --profile to use the separate headless research profile.")


class Blocked(RuntimeError):
    pass


class NoBrowser(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class Helper:
    """A running helper, as clients see it."""
    port: int
    token: str
    profile: bool
    connected: bool  # False while it waits for the user to click Allow


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
    raise NoBrowser("Profile Chrome did not start; is it open in a login window? Quit it and retry.")


async def connect(p: Playwright, use_profile: bool) -> Browser:
    """Connect to the profile Chrome (starting it if needed) or to the user's Chrome, which asks them to Allow."""
    if use_profile:
        await asyncio.to_thread(start_profile_chrome)
        endpoint: str | None = PROFILE_ENDPOINT
    else:
        endpoint = user_chrome_endpoint()
    if endpoint is None:
        raise NoBrowser(NO_CHROME)
    return await p.chromium.connect_over_cdp(endpoint, timeout=ALLOW_WAIT_SECONDS * 1000)


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


async def target_id(page: Page) -> str:
    session = await page.context.new_cdp_session(page)
    try:
        return str((await session.send("Target.getTargetInfo"))["targetInfo"]["targetId"])
    finally:
        await session.detach()


async def visible_tabs(browser: Browser, exclude: Page | None = None) -> set[str]:
    """Target IDs of tabs currently showing in a window (one per window)."""
    shown: set[str] = set()
    for page in browser.contexts[0].pages:
        if page is exclude:
            continue
        try:
            if await page.evaluate("document.visibilityState") == "visible":
                shown.add(await target_id(page))
        except PlaywrightError:
            pass  # discarded or crashed tabs can't run script
    return shown


async def open_background_tab(browser: Browser) -> Page:
    """A new blank tab behind the user's current one. It's found by its target ID, so a tab
    opened at the same moment, by the user or by another read, is never taken for it."""
    context = browser.contexts[0]
    checked = set(context.pages)
    session = await browser.new_browser_cdp_session()
    try:
        created = await session.send("Target.createTarget", {"url": "about:blank", "background": True})
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            for page in context.pages:
                if page in checked:
                    continue
                try:
                    if await target_id(page) == created["targetId"]:
                        return page
                    checked.add(page)
                except PlaywrightError:
                    pass  # closed while we asked, so it isn't ours
            await asyncio.sleep(0.05)
        await session.send("Target.closeTarget", {"targetId": created["targetId"]})
        raise TimeoutError("the new background tab never appeared")
    finally:
        with contextlib.suppress(PlaywrightError):
            await session.detach()


async def read_page(browser: Browser, url: str, background: bool, check_focus: bool) -> dict[str, Any]:
    before = await visible_tabs(browser) if check_focus else set()
    page = await open_background_tab(browser) if background else await browser.contexts[0].new_page()
    try:
        response = await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
        try:
            await page.wait_for_load_state("networkidle", timeout=8_000)
        except PlaywrightTimeout:
            pass  # feeds and chat widgets never go idle; the content is there by now
        title, html = await page.title(), await page.content()
        result: dict[str, Any] = {"url": page.url, "title": title, "html": html, "text": await page.inner_text("body")}
        if check_focus:
            result["focus_kept"] = before <= await visible_tabs(browser, exclude=page)
        reason = blocked_reason(response.status if response else None, page.url, title, html)
        if reason:
            raise Blocked(reason)
        return result
    finally:
        with contextlib.suppress(PlaywrightError):
            await page.close()


# --- the helper ------------------------------------------------------------------------

def note(message: str) -> None:
    """A timestamped line in the helper's log (its stderr)."""
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{os.getpid()}] {message}", file=sys.stderr, flush=True)


def write_state(token: str, use_profile: bool) -> None:
    fd = os.open(STATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)  # the mode above only applies to a new file
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "port": HELPER_PORT, "token": token, "profile": use_profile}, f)


def remove_state(token: str) -> None:
    """Remove the state file, unless a newer helper has already replaced it."""
    try:
        if json.loads(STATE.read_text()).get("token") == token:
            STATE.unlink()
    except (OSError, ValueError, AttributeError):
        pass


async def serve(use_profile: bool, check_focus: bool) -> None:
    """Hold one CDP connection and answer page reads from local clients, several at a time.

    Requests are answered on their own threads, so health checks are answered even while
    pages load; the pages themselves are read on this event loop. Exits with status 3 if it
    couldn't connect.
    """
    loop = asyncio.get_running_loop()
    token = secrets.token_urlsafe(24)
    browser: Browser | None = None
    unavailable = "Chrome closed the connection."
    settled = asyncio.Event()  # the connection attempt is over, either way
    stopping = asyncio.Event()
    tabs = asyncio.Semaphore(MAX_TABS)
    last_used = time.monotonic()

    async def read(url: str) -> tuple[int, dict[str, Any]]:
        nonlocal last_used
        await settled.wait()
        connected = browser
        if connected is None or not connected.is_connected():
            return 503, {"unavailable": unavailable}
        async with tabs:
            last_used = time.monotonic()
            try:
                page = await asyncio.wait_for(read_page(connected, url, background=not use_profile,
                                                        check_focus=check_focus), READ_SECONDS)
                return 200, page
            except Blocked as e:
                return 409, {"blocked": str(e)}
            except (PlaywrightError, TimeoutError) as e:
                return 502, {"error": str(e) or f"no page after {READ_SECONDS} s"}
            finally:
                last_used = time.monotonic()

    class Handler(BaseHTTPRequestHandler):
        timeout = 10  # a client that connects and sends nothing can't hold a thread

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
            if not secrets.compare_digest(self.headers.get("X-Token", "").encode(), token.encode()):
                self.reply(403, {"error": "bad token"})
                return
            route = urllib.parse.urlparse(self.path)
            if route.path == "/health":
                self.reply(200, {"ok": True, "profile": use_profile, "connected": browser is not None})
            elif route.path == "/quit":
                loop.call_soon_threadsafe(stopping.set)
                self.reply(200, {"ok": True})
            elif route.path == "/read":
                url = urllib.parse.parse_qs(route.query).get("url", [""])[0]
                if urllib.parse.urlparse(url).scheme not in ("http", "https"):
                    self.reply(400, {"error": f"not an http(s) URL: {url!r}"})
                    return
                self.reply(*asyncio.run_coroutine_threadsafe(read(url), loop).result())
            else:
                self.reply(404, {"error": "unknown route"})

    async def connect_and_wait(p: Playwright) -> None:
        nonlocal browser, unavailable, last_used
        try:
            browser = await connect(p, use_profile)
        except (PlaywrightError, NoBrowser) as e:
            unavailable = f"Could not connect to Chrome: {e}"
            note(unavailable)
            return
        finally:
            settled.set()
        note("connected")
        last_used = time.monotonic()
        browser.on("disconnected", lambda _: stopping.set())
        while not stopping.is_set() and time.monotonic() - last_used < IDLE_SECONDS:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stopping.wait(), 5)
        note("exiting: " + ("Chrome closed the connection" if not browser.is_connected()
                            else "stop requested" if stopping.is_set() else "idle"))

    # Take the port before connecting: a second helper started by a racing client finds
    # it taken and exits here, before Chrome can show it an Allow prompt.
    try:
        server = ThreadingHTTPServer(("127.0.0.1", HELPER_PORT), Handler)
    except OSError as e:
        if e.errno != errno.EADDRINUSE:
            raise
        note(f"port {HELPER_PORT} is taken: a helper is already running or starting, so not connecting")
        return
    server.daemon_threads = False  # so server_close waits for in-flight replies
    threading.Thread(target=server.serve_forever, daemon=True).start()
    write_state(token, use_profile)
    note(f"listening on port {HELPER_PORT}; connecting to {'the profile' if use_profile else 'your'} Chrome")
    try:
        async with async_playwright() as p:
            try:
                await connect_and_wait(p)
            finally:
                await asyncio.to_thread(server.shutdown)  # take no new requests,
                await asyncio.to_thread(server.server_close)  # and let in-flight reads finish and reply
    finally:
        remove_state(token)
    if browser is None:
        raise SystemExit(3)


def call(helper: Helper, path: str, timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(f"http://127.0.0.1:{helper.port}{path}", headers={"X-Token": helper.token})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
    except urllib.error.HTTPError as e:
        body = json.load(e)
    if not isinstance(body, dict):
        raise ValueError(f"unexpected reply from the helper: {body!r}")
    return body


def helper_state() -> Helper | None:
    """The running helper, if one answers."""
    try:
        state = json.loads(STATE.read_text())
        port, token, profile = state["port"], state["token"], state["profile"]
    except (OSError, ValueError, TypeError, KeyError):
        return None
    if not (isinstance(port, int) and isinstance(token, str) and isinstance(profile, bool)):
        return None
    helper = Helper(port, token, profile, connected=False)
    try:
        health = call(helper, "/health", timeout=3)
    except (OSError, ValueError):
        return None
    if health.get("ok") is not True:
        return None
    return dataclasses.replace(helper, connected=health.get("connected") is True)


def ensure_helper(use_profile: bool) -> Helper:
    """The running helper, starting one if none answers."""
    helper = helper_state()
    if helper is None:
        if not use_profile and user_chrome_endpoint() is None:
            print(NO_CHROME, file=sys.stderr)
            raise SystemExit(3)
        with LOG.open("a") as log:
            started = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "serve"] + (["--profile"] if use_profile else []),
                stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
        # If another client's helper got the port first, ours exits with status 0 without
        # connecting; keep waiting, for theirs.
        deadline = time.monotonic() + START_WAIT_SECONDS
        while (helper := helper_state()) is None:
            if started.poll() not in (None, 0):
                print(f"The browser helper exited. See {LOG}.", file=sys.stderr)
                raise SystemExit(3 if started.returncode == 3 else 1)
            if time.monotonic() > deadline:
                print(f"No browser helper answered within {START_WAIT_SECONDS} s. See {LOG}.", file=sys.stderr)
                raise SystemExit(1)
            time.sleep(0.2)
    if helper.profile != use_profile:
        raise SystemExit("A helper for the other browser is running; run `python tools/browser.py stop` first.")
    return helper


def status() -> None:
    """Report whether reads can use the user's Chrome, without connecting to it (so no Allow prompt)."""
    helper = helper_state()
    if helper is not None and not helper.profile:
        if helper.connected:
            print("connected: reads go through the user's Chrome with no Allow prompt.")
        else:
            print("connecting: a helper is waiting for the user to click Allow in Chrome; reads wait for it.")
        return
    if user_chrome_endpoint() is None:
        print(f"off: {NO_CHROME}")
        raise SystemExit(3)
    hint = " A --profile helper is running; run `python tools/browser.py stop` first." if helper is not None else ""
    print(f"ready: the first read will ask the user to click Allow in Chrome.{hint}")


def stop() -> None:
    helper = helper_state()
    if helper is not None:
        call(helper, "/quit", timeout=10)
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
    sub.add_parser("status")
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
    if a.cmd == "status":
        status()
    elif a.cmd == "stop":
        stop()
    elif a.cmd == "login":
        login(a.urls)
    elif a.cmd == "serve":
        asyncio.run(serve(a.profile, a.check_focus))
    else:
        helper = ensure_helper(a.profile)
        if not helper.connected and not helper.profile:
            print("Connecting to your Chrome: click Allow in Chrome if it asks.", file=sys.stderr)
        try:
            body = call(helper, "/read?" + urllib.parse.urlencode({"url": a.url}), timeout=READ_WAIT_SECONDS)
        except (OSError, ValueError) as e:
            print(f"error: the browser helper stopped answering ({e}). See {LOG}.", file=sys.stderr)
            raise SystemExit(1)
        if "unavailable" in body:
            print(f"unavailable: {body['unavailable']}", file=sys.stderr)
            raise SystemExit(3)
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
