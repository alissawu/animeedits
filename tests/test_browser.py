"""browser.py status: reports whether reads can use the user's Chrome without connecting to it."""
import contextlib
import io
from pathlib import Path
import socket
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import browser

ENDPOINT = "ws://127.0.0.1:9222/devtools/browser/abc"


def run_status(helper: dict[str, Any] | None, endpoint: str | None) -> tuple[str, object]:
    """Run `status` with the helper and Chrome faked; fail if it connects to Chrome or starts a helper."""
    out = io.StringIO()
    code: object = 0
    with (mock.patch.object(browser, "helper_state", return_value=helper),
          mock.patch.object(browser, "user_chrome_endpoint", return_value=endpoint),
          mock.patch.object(browser, "sync_playwright", side_effect=AssertionError("status connected to Chrome")),
          mock.patch("subprocess.Popen", side_effect=AssertionError("status started a helper")),
          contextlib.redirect_stdout(out)):
        try:
            browser.status()
        except SystemExit as e:
            code = e.code
    return out.getvalue(), code


class StatusTest(unittest.TestCase):
    def test_connected_when_user_chrome_helper_runs(self) -> None:
        out, code = run_status({"profile": False}, ENDPOINT)
        self.assertTrue(out.startswith("connected:"), out)
        self.assertEqual(code, 0)

    def test_ready_when_chrome_is_debuggable(self) -> None:
        out, code = run_status(None, ENDPOINT)
        self.assertTrue(out.startswith("ready:"), out)
        self.assertNotIn("stop", out)
        self.assertEqual(code, 0)

    def test_ready_says_to_stop_a_running_profile_helper(self) -> None:
        out, code = run_status({"profile": True}, ENDPOINT)
        self.assertTrue(out.startswith("ready:"), out)
        self.assertIn("browser.py stop", out)
        self.assertEqual(code, 0)

    def test_off_exits_3(self) -> None:
        for helper in (None, {"profile": True}):
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


if __name__ == "__main__":
    unittest.main()
