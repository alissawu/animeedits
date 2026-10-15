"""show.py: manifests, the commands it runs, the name report and the summary, with fake index.py and story.py."""
import contextlib
import io
import json
import os
import sys
from pathlib import Path
import tempfile
import unittest
from typing import Any
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import show

# Stand-ins for index.py and story.py: they log their arguments and write what the real ones would. FAKE_FAIL names a
# video whose index.py fails; FAKE_WARN makes prepare warn about the subtitles.
FAKE_INDEX = """\
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
video, out = Path(args[0]), Path(args[1])
with open(LOG, "a") as f:
    f.write(json.dumps({"script": "index", "args": args}) + "\\n")
print("objc[123]: Class AVFFrameReceiver is implemented in both", file=sys.stderr)
print("dialogue from a:0 jpn (aac, default) (1 audio streams), language ja (the audio track's jpn tag)")
if os.environ.get("FAKE_FAIL") == video.stem:
    print("Traceback (most recent call last):")
    print("RuntimeError: ffmpeg failed: moov atom not found")
    sys.exit(1)
(out / "cache").mkdir(parents=True, exist_ok=True)
if "--no-captions" in args:
    if os.environ.get("FAKE_WARN"):
        print("warning: the subtitles start 2.0 s late against the dialogue")
    print(f"prepared 3 shots on 1 contact sheets in {out / 'sheets'}, without captions")
else:
    shots = [{"id": 1, "characters": [{"name": "Mira Sato", "certain": True}]},
             {"id": 2, "characters": [{"name": "Mira Sato", "certain": False}, {"name": "Bayer", "certain": True}]},
             {"id": 3, "characters": []}]
    (out / "shots.json").write_text(json.dumps(shots))
    print(f"3 shots -> {out / 'shots.json'}")
"""
FAKE_STORY = """\
import json, sys
from pathlib import Path
with open(LOG, "a") as f:
    f.write(json.dumps({"script": "story", "args": sys.argv[1:]}) + "\\n")
out = Path(sys.argv[1])
(out / "story.json").write_text("[]")
print(f"2 scenes, 1 flagged caption names, 1 shots with corrected names -> {out / 'story.json'}")
"""

MANIFEST = """\
title = "Example Show"

[[groups]]
name = "Season 1"
cast = "cast/s1.md"

[[groups.videos]]
video = "example/S01E01.mkv"
subs = "example/S01E01.en.srt"
synopsis = "synopses/s1e01.md"

[[groups.videos]]
video = "example/S01E02.mkv"
synopsis = "synopses/s1e02.md"
lang = "ja"

[[groups]]
name = "The Movie"
cast = "cast/movie.md"

[[groups.videos]]
video = "example/Movie.mp4"
synopsis = "synopses/movie.md"
"""

CAST = "# Example Show, season 1\n\n## Cast\n### Mira Sato\n- Hair: purple bob\n### Tomo Fushiguro\n### Rin Fushiguro\n"


class ShowTestCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.show_dir, self.media, self.out = self.root / "shows" / "example", self.root / "media", self.root / "index"
        (self.show_dir / "cast").mkdir(parents=True)
        (self.show_dir / "synopses").mkdir()
        (self.show_dir / "show.toml").write_text(MANIFEST)
        (self.show_dir / "cast" / "s1.md").write_text(CAST)
        (self.show_dir / "synopses" / "s1e01.md").write_text("Mira meets Tomo.\n")
        (self.media / "example").mkdir(parents=True)
        for name in ("S01E01.mkv", "S01E01.en.srt", "S01E02.mkv", "Movie.mp4"):
            (self.media / "example" / name).write_text("")
        self.log = self.root / "calls.jsonl"
        scripts = {}
        for name, body in (("index.py", FAKE_INDEX), ("story.py", FAKE_STORY)):
            scripts[name] = self.root / name
            scripts[name].write_text(f"LOG = {str(self.log)!r}\n{body}")
        patches = [mock.patch.object(show, "INDEX_SCRIPT", scripts["index.py"]),
                   mock.patch.object(show, "STORY_SCRIPT", scripts["story.py"])]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def load(self) -> show.Show:
        return show.load_show(self.show_dir, self.media)

    def run_show(self, command: str, *args: str, **env: str) -> tuple[int, str]:
        argv = ["show.py", command, str(self.show_dir), *args, "--media", str(self.media), "--out", str(self.out)]
        stdout = io.StringIO()
        code = 0
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, env), \
                contextlib.redirect_stdout(stdout):
            try:
                show.main()
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
                if isinstance(e.code, str):
                    stdout.write(e.code)
        return code, stdout.getvalue()

    def calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []


class ManifestTests(ShowTestCase):
    def test_load(self) -> None:
        loaded = self.load()
        self.assertEqual(loaded.title, "Example Show")
        self.assertEqual([g.name for g in loaded.groups], ["Season 1", "The Movie"])
        first, second, movie = loaded.videos
        self.assertEqual((first.name, first.group.name), ("S01E01", "Season 1"))
        self.assertEqual(first.video, self.media / "example" / "S01E01.mkv")
        self.assertEqual(first.subs, self.media / "example" / "S01E01.en.srt")
        self.assertEqual(first.synopsis, self.show_dir / "synopses" / "s1e01.md")
        self.assertEqual(first.group.cast, self.show_dir / "cast" / "s1.md")
        self.assertIsNone(first.lang)
        self.assertEqual((second.subs, second.lang), (None, "ja"))
        self.assertEqual(movie.group.cast, self.show_dir / "cast" / "movie.md")

    def test_every_problem_is_listed(self) -> None:
        (self.show_dir / "show.toml").write_text("""\
titel = "typo"

[[groups]]
name = "Season 1"
cast = "/abs/s1.md"

[[groups.videos]]
video = "example/S01E01.mkv"
synopsis = "../elsewhere.md"
extra = 1

[[groups.videos]]
video = "other/S01E01.mp4"
synopsis = "synopses/again.md"

[[groups]]
name = "Season 1"
cast = "cast/s1.md"
videos = []
""")
        with self.assertRaises(show.ManifestError) as raised:
            self.load()
        message = str(raised.exception)
        for problem in ("unknown key 'titel'", "missing title", "cast '/abs/s1.md' must be relative",
                        "synopsis '../elsewhere.md' must be relative", "unknown key 'extra'",
                        "another group has this name", "needs at least one [[groups.videos]]"):
            self.assertIn(problem, message)
        self.assertEqual(len(message.splitlines()), 7)

    def test_duplicate_video_names(self) -> None:
        (self.show_dir / "show.toml").write_text(MANIFEST.replace("example/Movie.mp4", "movies/S01E01.mp4"))
        with self.assertRaisesRegex(show.ManifestError, "another video is also named S01E01"):
            self.load()

    def test_not_toml(self) -> None:
        (self.show_dir / "show.toml").write_text("title = \n")
        with self.assertRaisesRegex(show.ManifestError, "not valid TOML"):
            self.load()
        (self.show_dir / "show.toml").unlink()
        with self.assertRaisesRegex(show.ManifestError, "can't read"):
            self.load()

    def test_select(self) -> None:
        loaded = self.load()
        self.assertEqual([v.name for v in show.select(loaded, [])], ["S01E01", "S01E02", "Movie"])
        self.assertEqual([v.name for v in show.select(loaded, ["The Movie", "S01E01"])], ["S01E01", "Movie"])
        with self.assertRaisesRegex(show.ManifestError, "--only S09: not a group or video.*Season 1; The Movie"):
            show.select(loaded, ["S09"])

    def test_commands(self) -> None:
        first, second, _ = self.load().videos
        out = self.out / first.name
        prepare = show.index_command(first, out, "prepare")
        self.assertEqual(prepare[1:], [str(show.INDEX_SCRIPT), str(first.video), str(out), "--sample", "5", "--proxy",
                                       "480", "--faces", "anime", "--subs", str(first.subs), "--no-captions"])
        finish = show.index_command(second, self.out / second.name, "finish")
        self.assertEqual(finish[-4:], ["--lang", "ja", "--context", str(second.group.cast)])
        self.assertNotIn("--no-captions", finish)
        self.assertEqual(show.story_command(first, out)[2:], [str(out), "--context", str(first.group.cast),
                                                              "--synopsis", str(first.synopsis)])


class NameReportTests(unittest.TestCase):
    NAMES = ["Mira Sato", "Tomo Fushiguro", "Rin Fushiguro", "Angel Devil", "Typhoon Devil"]

    def test_mentioned(self) -> None:
        synopsis = show.name_key("Tomo finds the Angel Devil. Mrs. Fushiguro waits; Mira-san's sister calls.")
        for name, expected in {"Tomo Fushiguro": True, "Rin Fushiguro": False, "Angel Devil": True,
                               "Typhoon Devil": False, "Mira Sato": True, "Bayer": False, "Mira": True}.items():
            with self.subTest(name):
                self.assertEqual(show.mentioned(name, self.NAMES, synopsis), expected)

    def test_report(self) -> None:
        shots: list[Any] = [
            {"characters": [{"name": "Mira Sato", "certain": True}, {"name": "Bayer", "certain": False}]},
            {"characters": [{"name": "Mira Sato", "certain": False}]},
            {"characters": [{"name": "Rin Fushiguro", "certain": True}]},
            {"characters": []}]
        report = show.name_report(shots, self.NAMES, "Mira and Bayer talk.")
        self.assertEqual(report, [
            {"name": "Mira Sato", "shots": 2, "unsure": 1, "in_cast": True, "in_synopsis": True},
            {"name": "Bayer", "shots": 1, "unsure": 1, "in_cast": False, "in_synopsis": True},
            {"name": "Rin Fushiguro", "shots": 1, "unsure": 0, "in_cast": True, "in_synopsis": False},
        ])
        self.assertEqual(show.report_lines(report, len(shots)), [
            "names in the captions of 4 shots: Mira Sato 2 (1 possibly), Bayer 1 (1 possibly), Rin Fushiguro 1",
            "not in the cast notes: Bayer 1 (1 possibly)",
            "not in the synopsis: Rin Fushiguro 1",
        ])
        self.assertEqual(show.report_lines([], 3), ["names in the captions of 3 shots: none"])


class CommandTests(ShowTestCase):
    def test_prepare(self) -> None:
        code, printed = self.run_show("prepare", FAKE_WARN="1")
        self.assertEqual(code, 0, printed)
        self.assertEqual(sorted(c["args"][0] for c in self.calls()),
                         sorted(str(v.video) for v in self.load().videos))
        self.assertTrue(all(c["script"] == "index" and "--no-captions" in c["args"] for c in self.calls()))
        self.assertIn("prepare example: 3 of 3 videos done", printed)
        self.assertIn("ok  S01E01  (Season 1,", printed)
        self.assertIn("  warning: the subtitles start 2.0 s late against the dialogue\n"
                      "  prepared 3 shots on 1 contact sheets", printed)
        self.assertNotIn("objc[", printed.split("prepare example:")[1], "library noise stays in the log")
        self.assertIn("research to write before finish (docs/indexing.md#research):\n"
                      "  cast notes for The Movie: cast/movie.md\n"
                      "  synopsis of S01E02: synopses/s1e02.md\n"
                      "  synopsis of Movie: synopses/movie.md", printed)
        log = (self.out / "S01E01" / "index.log").read_text()
        self.assertIn("--no-captions", log.splitlines()[1])
        self.assertIn("objc[123]", log, "the log keeps everything")

    def test_prepare_reports_missing_files_and_failures(self) -> None:
        (self.media / "example" / "S01E01.en.srt").unlink()
        code, printed = self.run_show("prepare", "--only", "Season 1", FAKE_FAIL="S01E02")
        self.assertEqual(code, 1)
        self.assertEqual([c["args"][0] for c in self.calls()], [str(self.media / "example" / "S01E02.mkv")],
                         "a video missing a file isn't run")
        self.assertIn(f"FAILED  S01E01  (Season 1, 0m00s)\n  no subtitle file at {self.media}/example/S01E01.en.srt",
                      printed)
        self.assertIn("FAILED  S01E02", printed)
        self.assertIn("index.py exited with 1; the end of its output (all of it is in", printed)
        self.assertIn("    RuntimeError: ffmpeg failed: moov atom not found", printed)
        self.assertIn("0 of 2 videos done, 2 failed", printed)

    def test_finish(self) -> None:
        code, printed = self.run_show("finish")
        self.assertEqual(code, 1, "two videos have no research yet")
        self.assertEqual([(c["script"], Path(c["args"][1 if c["script"] == "index" else 0]).name) for c in self.calls()],
                         [("index", "S01E01"), ("story", "S01E01")])
        index_args, story_args = self.calls()[0]["args"], self.calls()[1]["args"]
        self.assertEqual(index_args[index_args.index("--context") + 1], str(self.show_dir / "cast" / "s1.md"))
        self.assertEqual(story_args[story_args.index("--synopsis") + 1], str(self.show_dir / "synopses" / "s1e01.md"))
        self.assertIn("ok  S01E01  (Season 1,", printed)
        self.assertIn("  3 shots -> ", printed)
        self.assertIn("  names in the captions of 3 shots: Mira Sato 2 (1 possibly), Bayer 1\n"
                      "  not in the cast notes: Bayer 1\n"
                      "  not in the synopsis: Bayer 1\n"
                      "  2 scenes, 1 flagged caption names, 1 shots with corrected names", printed)
        self.assertIn(f"FAILED  S01E02  (Season 1, 0m00s)\n  no synopsis at {self.show_dir}/synopses/s1e02.md", printed)
        self.assertIn(f"  no cast notes at {self.show_dir}/cast/movie.md\n  no synopsis at", printed)

        code, printed = self.run_show("report", "--only", "S01E01", "S01E02")
        self.assertEqual(code, 0)
        self.assertIn("S01E01  (Season 1)\n  names in the captions of 3 shots", printed)
        self.assertIn("S01E02  (Season 1)\n  no name report:", printed)

    def test_finish_stops_at_a_failed_index(self) -> None:
        code, printed = self.run_show("finish", "--only", "S01E01", FAKE_FAIL="S01E01")
        self.assertEqual(code, 1)
        self.assertEqual([c["script"] for c in self.calls()], ["index"], "no story without an index")
        self.assertNotIn("names in the captions", printed)

    def test_dry_run(self) -> None:
        code, printed = self.run_show("finish", "--dry-run", "--only", "S01E01")
        self.assertEqual(code, 0)
        self.assertEqual(self.calls(), [])
        index_line, story_line = printed.strip().splitlines()
        self.assertIn("--sample 5 --proxy 480 --faces anime --subs", index_line)
        self.assertIn(f"--context {self.show_dir}/cast/s1.md", index_line)
        self.assertIn(f"--synopsis {self.show_dir}/synopses/s1e01.md", story_line)

    def test_needs_media(self) -> None:
        argv = ["show.py", "prepare", str(self.show_dir), "--media", str(self.root / "none")]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()) as stderr:
            show.main()
        self.assertIn("no media folder at", stderr.getvalue())

    def test_try(self) -> None:
        out = self.out / "S01E01"
        (out / "cache").mkdir(parents=True)
        (out / "sheets").mkdir()
        rows = [{"i": i, "start": float(i), "end": i + 1.0} for i in range(1, show.SHEET_SIZE + 3)]
        (out / "cache" / "detection.json").write_text(json.dumps({"signature": {}, "data": rows}))
        (out / "sheets" / "sheet_0002.jpg").write_bytes(b"image")
        asked: list[tuple[list[int], str, str]] = []

        def caption(group: list[Any], sheet: Path, model: str, title: str, cast: str) -> dict[str, Any]:
            asked.append(([s["i"] for s in group], sheet.name, model))
            self.assertIn("### Mira Sato", cast)
            return {str(s["i"]): {"caption": f"Shot {s['i']}.", "characters": [{"name": "Mira", "certain": False}]}
                    for s in group}

        with mock.patch.object(show, "caption_sheet_cli", caption), mock.patch.object(show.shutil, "which",
                                                                                      return_value="claude"):
            code, printed = self.run_show("try", "--only", "S01E01", "--sheets", "2")
            self.assertEqual(code, 0, printed)
            self.assertEqual(asked, [([13, 14], "sheet_0002.jpg", show.CAPTION_MODEL)])
            self.assertIn("  0013 13.0-14.0s [possibly Mira Sato] Shot 13.", printed)
            code, printed = self.run_show("try", "--only", "S01E01", "--sheets", "3")
            self.assertIn("has sheets 1-2, not 3", printed)
            code, printed = self.run_show("try", "--only", "S01E01", "--sheets", "1")
            self.assertIn("run prepare first", printed)
            with contextlib.redirect_stderr(io.StringIO()):
                code, _ = self.run_show("try", "--sheets", "1")
            self.assertEqual(code, 2, "try takes one video")


if __name__ == "__main__":
    unittest.main()
