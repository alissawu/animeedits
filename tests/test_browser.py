"""browser.py: `status` never connects to Chrome, and the helper opens one connection however it's used."""
import asyncio
import contextlib
import io
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
from typing import Any
import unittest
from unittest import mock
import urllib.parse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import browser

ENDPOINT = "ws://127.0.0.1:9222/devtools/browser/abc"


def running(profile: bool, connected: bool = True) -> browser.Helper:
    return browser.Helper(port=1, token="t", profile=profile, connected=connected)


def run_status(helper: browser.Helper | None, endpoint: str | None) -> tuple[str, object]:
    """Run `status` with the helper and Chrome faked; fail if it connects to Chrome or starts a helper."""
    out = io.StringIO()
    code: object = 0
    with (mock.patch.object(browser, "helper_state", return_value=helper),
          mock.patch.object(browser, "user_chrome_endpoint", return_value=endpoint),
          mock.patch.object(browser, "sync_playwright", side_effect=AssertionError("status connected to Chrome")),
          mock.patch.object(browser, "async_playwright", side_effect=AssertionError("status connected to Chrome")),
          mock.patch("subprocess.Popen", side_effect=AssertionError("status started a helper")),
          contextlib.redirect_stdout(out)):
        try:
            browser.status()
        except SystemExit as e:
            code = e.code
    return out.getvalue(), code


class StatusTest(unittest.TestCase):
    def test_connected_when_user_chrome_helper_runs(self) -> None:
        out, code = run_status(running(profile=False), ENDPOINT)
        self.assertTrue(out.startswith("connected:"), out)
        self.assertEqual(code, 0)

    def test_connecting_while_the_helper_waits_for_allow(self) -> None:
        out, code = run_status(running(profile=False, connected=False), ENDPOINT)
        self.assertTrue(out.startswith("connecting:"), out)
        self.assertEqual(code, 0)

    def test_ready_when_chrome_is_debuggable(self) -> None:
        out, code = run_status(None, ENDPOINT)
        self.assertTrue(out.startswith("ready:"), out)
        self.assertNotIn("stop", out)
        self.assertEqual(code, 0)

    def test_ready_says_to_stop_a_running_profile_helper(self) -> None:
        out, code = run_status(running(profile=True), ENDPOINT)
        self.assertTrue(out.startswith("ready:"), out)
        self.assertIn("browser.py stop", out)
        self.assertEqual(code, 0)

    def test_off_exits_3(self) -> None:
        for helper in (None, running(profile=True)):
            out, code = run_status(helper, None)
            self.assertTrue(out.startswith("off:"), out)
            self.assertIn("chrome://inspect/#remote-debugging", out)
            self.assertEqual(code, 3)


class UserChromeEndpointTest(unittest.TestCase):
    """The check `status` relies on: Chrome's DevToolsActivePort file plus a live listener."""

    def endpoint(self, port_file: str | None) -> str | None:
        with tempfile.TemporaryDirectory() as d:
            if port_file is not None:
                (Path(d) / "DevToolsActivePort").write_text(port_file)
            with mock.patch.object(browser, "USER_CHROME_DATA", Path(d)):
                return browser.user_chrome_endpoint()

    def test_no_file(self) -> None:
        self.assertIsNone(self.endpoint(None))

    def test_malformed_file(self) -> None:
        self.assertIsNone(self.endpoint("not-a-port"))

    def test_stale_file_with_nothing_listening(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.assertIsNone(self.endpoint(f"{port}\n/devtools/browser/abc\n"))

    def test_listening(self) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            port = s.getsockname()[1]
            self.assertEqual(self.endpoint(f"{port}\n/devtools/browser/abc\n"),
                             f"ws://127.0.0.1:{port}/devtools/browser/abc")


class FakeBrowser:
    def is_connected(self) -> bool:
        return True

    def on(self, event: str, handler: Any) -> None:
        pass


class HelperTest(unittest.TestCase):
    """The real helper server with Chrome faked. Each connection to the user's Chrome is an
    Allow prompt, so the helper must connect once however many clients race to start it,
    and stay reachable while pages load so no client mistakes it for dead and starts another."""

    def setUp(self) -> None:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.connects = 0
        self.allowed = threading.Event()  # the user clicked Allow
        self.allowed.set()
        self.refuse = False
        self.reading = threading.Semaphore(0)  # released once per read that has started
        self.finish_reads = threading.Event()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        for patch in (mock.patch.object(browser, "HELPER_PORT", port),
                      mock.patch.object(browser, "STATE", Path(d.name) / "state.json"),
                      mock.patch.object(browser, "async_playwright", contextlib.nullcontext),
                      mock.patch.object(browser, "connect", self.fake_connect),
                      mock.patch.object(browser, "read_page", self.fake_read_page),
                      mock.patch.object(browser, "note", lambda message: None)):
            patch.start()
            self.addCleanup(patch.stop)

    async def fake_connect(self, p: object, use_profile: bool) -> FakeBrowser:
        self.connects += 1
        await asyncio.to_thread(self.allowed.wait, 10)
        if self.refuse:
            raise browser.NoBrowser("the user clicked Cancel")
        return FakeBrowser()

    async def fake_read_page(self, b: object, url: str, background: bool, check_focus: bool) -> dict[str, Any]:
        self.reading.release()
        await asyncio.to_thread(self.finish_reads.wait, 10)
        return {"url": url, "title": "t", "html": "", "text": url}

    def start_helper(self) -> tuple[threading.Thread, list[object]]:
        """`serve` on a thread; the list gets its exit status."""
        status: list[object] = []

        def run() -> None:
            try:
                asyncio.run(browser.serve(use_profile=False, check_focus=False))
                status.append(0)
            except SystemExit as e:
                status.append(e.code)

        thread = threading.Thread(target=run)
        thread.start()
        self.addCleanup(self.stop_helper, thread)
        return thread, status

    def stop_helper(self, thread: threading.Thread) -> None:
        self.allowed.set()
        self.finish_reads.set()
        helper = browser.helper_state()
        if helper is not None:
            browser.call(helper, "/quit", timeout=5)
        thread.join(10)

    def wait_for_helper(self) -> browser.Helper:
        deadline = time.monotonic() + 5
        while (helper := browser.helper_state()) is None:
            self.assertLess(time.monotonic(), deadline, "the helper never answered")
            time.sleep(0.05)
        return helper

    def start_read(self, helper: browser.Helper, url: str) -> tuple[threading.Thread, list[dict[str, Any]]]:
        reply: list[dict[str, Any]] = []
        path = "/read?" + urllib.parse.urlencode({"url": url})
        thread = threading.Thread(target=lambda: reply.append(browser.call(helper, path, timeout=20)))
        thread.start()
        return thread, reply

    def test_racing_helpers_connect_once(self) -> None:
        self.allowed.clear()  # every racer starts while the first is still waiting for the Allow
        helpers = [self.start_helper() for _ in range(5)]
        losers = []
        deadline = time.monotonic() + 5
        while len(losers) < 4:
            self.assertLess(time.monotonic(), deadline, "the extra helpers never exited")
            losers = [status for thread, status in helpers if not thread.is_alive()]
            time.sleep(0.05)
        self.assertEqual(losers, [[0]] * 4)
        self.assertEqual(self.connects, 1)
        self.assertFalse(self.wait_for_helper().connected)

    def test_reads_wait_for_allow(self) -> None:
        self.allowed.clear()
        self.start_helper()
        helper = self.wait_for_helper()
        self.assertFalse(helper.connected)
        reader, reply = self.start_read(helper, "https://a.example/")
        time.sleep(0.3)
        self.assertTrue(reader.is_alive(), "the read didn't wait for the Allow")
        self.allowed.set()
        self.finish_reads.set()
        reader.join(10)
        self.assertEqual(reply, [{"url": "https://a.example/", "title": "t", "html": "", "text": "https://a.example/"}])
        self.assertTrue(self.wait_for_helper().connected)

    def test_health_answers_while_reads_run_together(self) -> None:
        thread, status = self.start_helper()
        helper = self.wait_for_helper()
        urls = ["https://a.example/", "https://b.example/", "https://c.example/"]
        readers = [self.start_read(helper, url) for url in urls]
        for _ in urls:
            self.assertTrue(self.reading.acquire(timeout=5), "the reads didn't run at the same time")
        started = time.monotonic()
        self.assertIsNotNone(browser.helper_state())
        self.assertLess(time.monotonic() - started, 1)
        self.finish_reads.set()
        for reader, reply in readers:
            reader.join(10)
        self.assertEqual(sorted(reply[0]["text"] for reader, reply in readers), urls)
        browser.call(helper, "/quit", timeout=5)
        thread.join(10)
        self.assertEqual(status, [0])
        self.assertFalse(browser.STATE.exists())
        self.assertEqual(self.connects, 1)

    def test_quit_lets_a_read_in_progress_finish(self) -> None:
        thread, status = self.start_helper()
        helper = self.wait_for_helper()
        reader, reply = self.start_read(helper, "https://a.example/")
        self.assertTrue(self.reading.acquire(timeout=5))
        browser.call(helper, "/quit", timeout=5)
        self.finish_reads.set()
        reader.join(10)
        thread.join(10)
        self.assertEqual(reply[0]["text"], "https://a.example/")
        self.assertEqual(status, [0])

    def test_refused_connection_answers_reads_and_exits_3(self) -> None:
        self.allowed.clear()
        self.refuse = True
        thread, status = self.start_helper()
        helper = self.wait_for_helper()
        reader, reply = self.start_read(helper, "https://a.example/")
        time.sleep(0.3)
        self.allowed.set()
        reader.join(10)
        thread.join(10)
        self.assertEqual(reply, [{"unavailable": "Could not connect to Chrome: the user clicked Cancel"}])
        self.assertEqual(status, [3])
        self.assertFalse(browser.STATE.exists())

    def test_rejects_non_http_urls(self) -> None:
        self.start_helper()
        helper = self.wait_for_helper()
        reader, reply = self.start_read(helper, "file:///etc/passwd")
        reader.join(10)
        self.assertIn("not an http(s) URL", reply[0]["error"])

    def test_rejects_a_bad_token(self) -> None:
        self.start_helper()
        helper = self.wait_for_helper()
        forged = browser.Helper(helper.port, "wrong", helper.profile, helper.connected)
        self.assertEqual(browser.call(forged, "/health", timeout=5), {"error": "bad token"})


if __name__ == "__main__":
    unittest.main()
