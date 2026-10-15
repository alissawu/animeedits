"""story.py: scene validation, request text, chunking and caching, with a fake `claude` on PATH."""
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
import index  # the module story.py imports, as opposed to test_video's separately loaded copy
import story

# A stand-in for Claude Code: logs its arguments, request and auth env, and splits the requested shots into scenes of
# FAKE_SCENE_SIZE shots. FAKE_GAP leaves the first shot out, which validation rejects.
FAKE_CLAUDE = """\
import json, os, re, sys
args = sys.argv[1:]
content = json.loads(sys.stdin.read())["message"]["content"]
text = "\\n\\n".join(block["text"] for block in content)
with open(LOG, "a") as f:
    f.write(json.dumps({"args": args, "text": text, "api_key": "ANTHROPIC_API_KEY" in os.environ}) + "\\n")
ids = [int(i) for i in re.findall(r"^shot (\\d+) \\(", text, re.M)]
if os.environ.get("FAKE_GAP"):
    ids = ids[1:]
size = int(os.environ.get("FAKE_SCENE_SIZE", "2"))
scenes = [{"first_shot": ids[k], "last_shot": ids[min(k + size, len(ids)) - 1], "label": f"Scene from shot {ids[k]}",
           "characters": ["Mira"], "confidence": "medium", "flags": []} for k in range(0, len(ids), size)]
if ids[0] == 1:
    scenes[0]["flags"] = [{"shot": 1, "name": "possibly Tomo", "likely": "Rin", "reason": "Mira calls her Rin."}]
model = args[args.index("--model") + 1]
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "modelUsage": {model: {}},
                  "structured_output": {"scenes": scenes}}))
"""

SPANNING_CUE = {"start": 1.5, "end": 2.5, "text": "Rin, wait!"}


def make_shot(i: int, **extra: Any) -> dict[str, Any]:
    return {"id": i, "start": 2.0 * (i - 1), "end": 2.0 * i, "caption": f"Caption {i}.", "characters": [], **extra}


def make_shots(n: int) -> list[dict[str, Any]]:
    shots = [make_shot(i, subtitles=[]) for i in range(1, n + 1)]
    shots[0]["characters"] = [{"name": "Mira", "certain": True}, {"name": "Tomo", "certain": False}]
    shots[0]["subtitles"] = shots[1]["subtitles"] = [SPANNING_CUE]
    return shots


def reply(*scenes: dict[str, Any]) -> str:
    return json.dumps({"scenes": [{"label": "x", "characters": [], "confidence": "high", "flags": [], **s}
                                  for s in scenes]})


class ParseScenesTests(unittest.TestCase):
    group = make_shots(4)

    def test_scenes_cover_the_chunk(self) -> None:
        scenes = story.parse_scenes(reply(
            {"first_shot": 1, "last_shot": 3, "label": " Mira  meets Rin. ", "characters": ["Mira", " Rin", "Mira", ""]},
            {"first_shot": 4, "last_shot": 4, "confidence": "low"}), self.group)
        self.assertEqual(scenes, [
            {"id": 1, "first_shot": 1, "last_shot": 3, "start": 0.0, "end": 6.0, "label": "Mira meets Rin.",
             "characters": ["Mira", "Rin"], "confidence": "high", "flags": []},
            {"id": 2, "first_shot": 4, "last_shot": 4, "start": 6.0, "end": 8.0, "label": "x", "characters": [],
             "confidence": "low", "flags": []},
        ])

    def test_rejects_scenes_that_do_not_cover_the_chunk(self) -> None:
        for name, bad in {
            "gap": reply({"first_shot": 1, "last_shot": 1}, {"first_shot": 3, "last_shot": 4}),
            "overlap": reply({"first_shot": 1, "last_shot": 2}, {"first_shot": 2, "last_shot": 4}),
            "late start": reply({"first_shot": 2, "last_shot": 4}),
            "stops early": reply({"first_shot": 1, "last_shot": 3}),
            "past the chunk": reply({"first_shot": 1, "last_shot": 5}),
            "past the last shot": reply({"first_shot": 1, "last_shot": 4}, {"first_shot": 5, "last_shot": 5}),
            "backwards": reply({"first_shot": 1, "last_shot": 2}, {"first_shot": 3, "last_shot": 1}),
            "no scenes": reply(),
            "empty label": reply({"first_shot": 1, "last_shot": 4, "label": " "}),
            "confidence": reply({"first_shot": 1, "last_shot": 4, "confidence": "certain"}),
            "characters": reply({"first_shot": 1, "last_shot": 4, "characters": [{"name": "Mira"}]}),
            "not an object": json.dumps({"scenes": ["x"]}),
            "not a reply": json.dumps({"shots": []}),
        }.items():
            with self.subTest(name), self.assertRaises(ValueError):
                story.parse_scenes(bad, self.group)

    def test_flags(self) -> None:
        flags = [{"shot": 1, "name": "possibly tomo", "likely": " Rin ", "reason": "Mira  calls her Rin."},
                 {"shot": 1, "name": "Tomo", "likely": None, "reason": "repeat"},
                 {"shot": 1, "name": "Mira", "likely": "Mira", "reason": "agrees with the caption"},
                 {"shot": 1, "name": "Mira", "likely": "", "reason": "Rin is alone here."}]
        [scene] = story.parse_scenes(reply({"first_shot": 1, "last_shot": 4, "flags": flags}), self.group)
        self.assertEqual(scene["flags"], [
            {"shot": 1, "name": "Tomo", "likely": "Rin", "reason": "Mira calls her Rin."},
            {"shot": 1, "name": "Mira", "likely": None, "reason": "Rin is alone here."},
        ])

    def test_rejects_flags_on_names_the_scene_does_not_caption(self) -> None:
        for name, flag in {
            "not captioned": {"shot": 1, "name": "Rin", "likely": None, "reason": "x"},
            "another shot": {"shot": 2, "name": "Tomo", "likely": None, "reason": "x"},
            "another scene": {"shot": 1, "name": "Tomo", "likely": None, "reason": "x", "scene": 2},
            "no reason": {"shot": 1, "name": "Tomo", "likely": None, "reason": " "},
            "bad likely": {"shot": 1, "name": "Tomo", "likely": 3, "reason": "x"},
        }.items():
            scenes = ([{"first_shot": 1, "last_shot": 1}, {"first_shot": 2, "last_shot": 4, "flags": [flag]}]
                      if flag.pop("scene", None) else [{"first_shot": 1, "last_shot": 4, "flags": [flag]}])
            with self.subTest(name), self.assertRaises(ValueError):
                story.parse_scenes(reply(*scenes), self.group)


def named(*names: str) -> list[dict[str, Any]]:
    """Characters: a name ending in ? is unsure."""
    return [{"name": n.rstrip("?"), "certain": not n.endswith("?")} for n in names]


class CorrectedNamesTests(unittest.TestCase):
    NAMES = ["Mira Sato", "Rin Aoyama", "Tomo"]

    def scene(self, characters: list[str], *flags: tuple[int, str, str | None]) -> dict[str, Any]:
        return {"id": 1, "first_shot": 1, "last_shot": 4, "start": 0.0, "end": 8.0, "label": "x",
                "characters": characters, "confidence": "high",
                "flags": [{"shot": shot, "name": name, "likely": likely, "reason": "r"} for shot, name, likely in flags]}

    def shots(self) -> list[dict[str, Any]]:
        return [make_shot(1, characters=named("Mira Sato", "Tomo?")), make_shot(2, characters=named("Tomo")),
                make_shot(3, characters=named("Tomo", "Rin Aoyama")), make_shot(4, characters=named("Mira Sato"))]

    def corrected(self, scene: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        return {s["id"]: s["characters"] for s in story.corrected_names(scene, self.shots(), self.NAMES)}

    def test_a_name_the_scene_leaves_out_is_wrong_on_every_shot(self) -> None:
        result = story.corrected_names(self.scene(["Mira", "Rin"], (1, "Tomo", "rin")), self.shots(), self.NAMES)
        self.assertEqual(result, [
            {"id": 1, "characters": named("Mira Sato", "Rin Aoyama?"), "caption_characters": named("Mira Sato", "Tomo?")},
            {"id": 2, "characters": named("Rin Aoyama?"), "caption_characters": named("Tomo")},
            {"id": 3, "characters": named("Rin Aoyama"), "caption_characters": named("Tomo", "Rin Aoyama")},
            {"id": 4, "characters": named("Mira Sato")},
        ])

    def test_a_name_the_scene_includes_changes_only_where_flagged(self) -> None:
        corrected = self.corrected(self.scene(["Mira Sato", "Tomo"], (1, "Tomo", "Rin Aoyama")))
        self.assertEqual(corrected, {1: named("Mira Sato", "Rin Aoyama?"), 2: named("Tomo"),
                                     3: named("Tomo", "Rin Aoyama"), 4: named("Mira Sato")})

    def test_flags_that_disagree_drop_the_name_elsewhere(self) -> None:
        corrected = self.corrected(self.scene(["Mira Sato"], (1, "Tomo", "Rin Aoyama"), (2, "Tomo", "Mira")))
        self.assertEqual(corrected, {1: named("Mira Sato", "Rin Aoyama?"), 2: named("Mira Sato?"),
                                     3: named("Rin Aoyama"), 4: named("Mira Sato")})

    def test_nobody_likely(self) -> None:
        corrected = self.corrected(self.scene([], (4, "Mira Sato", None)))
        self.assertEqual(corrected[1], named("Tomo?"))
        self.assertEqual(corrected[4], [])

    def test_no_flags(self) -> None:
        result = story.corrected_names(self.scene(["Mira Sato"]), self.shots(), self.NAMES)
        self.assertEqual([s["characters"] for s in result], [s["characters"] for s in self.shots()])
        self.assertFalse(any("caption_characters" in s for s in result))

    def test_tell_story_spells_names_as_the_notes_do(self) -> None:
        [scene] = story.tell_story([self.scene(["Mira", "Rin", "Mira Sato"], (2, "Tomo", "rin"))], self.shots(),
                                   self.NAMES)
        self.assertEqual(scene["characters"], ["Mira Sato", "Rin Aoyama"])
        self.assertEqual(scene["flags"], [{"shot": 2, "name": "Tomo", "likely": "Rin Aoyama", "reason": "r"}])
        self.assertEqual([s["id"] for s in scene["shots"]], [1, 2, 3, 4])


class RequestTests(unittest.TestCase):
    def test_lines_are_read_once_and_subtitles_win(self) -> None:
        shots = make_shots(3)
        for shot in shots:
            shot["dialogue"] = [{"start": shot["start"], "end": shot["end"], "text": "transcript"}]
        self.assertEqual(story.line_source(shots), "subtitles")
        self.assertEqual(story.first_lines(shots, "subtitles"), {1: ["Rin, wait!"], 2: [], 3: []})
        for shot in shots:
            del shot["subtitles"]
        self.assertEqual(story.line_source(shots), "dialogue")
        self.assertEqual(story.first_lines(shots, "dialogue"), {1: ["transcript"], 2: ["transcript"], 3: ["transcript"]})
        for shot in shots:
            del shot["dialogue"]
        self.assertIsNone(story.line_source(shots))
        self.assertEqual(story.first_lines(shots, None), {1: [], 2: [], 3: []})

    def test_shot_text(self) -> None:
        shot = make_shots(2)[0]
        self.assertEqual(story.shot_text(shot, ['She said "go"', "行こう"]),
                         'shot 1 (0.0-2.0 s)\ncaption: Caption 1.\nnamed: Mira; possibly Tomo\n'
                         'lines: "She said \\"go\\"" | "行こう"')
        self.assertEqual(story.shot_text(make_shot(2), []), "shot 2 (2.0-4.0 s)\ncaption: Caption 2.")

    def test_background(self) -> None:
        text = story.background("Mira leaves.", "Example Show", "### Mira\n- choker", "dialogue")
        self.assertIn("The video is from Example Show.", text)
        self.assertIn("<synopsis>\nMira leaves.\n</synopsis>", text)
        self.assertIn("<cast>\n### Mira\n- choker\n</cast>", text)
        self.assertIn("machine transcript", text)
        self.assertEqual(story.background("Mira leaves.", None, None, None),
                         "<synopsis>\nMira leaves.\n</synopsis>\n\n" + story.LINE_SOURCES[None])

    def test_chunk_request_says_where_the_chunk_is(self) -> None:
        shots = make_shots(4)
        lines = story.first_lines(shots, "subtitles")
        whole = story.chunk_request(shots, lines, [], final=True)
        self.assertTrue(whole.startswith("These are all of the video's shots.\n\n<shots>\nshot 1 (0.0-2.0 s)"), whole)
        self.assertIn('lines: "Rin, wait!"', whole)
        first = story.chunk_request(shots[:2], lines, [], final=False)
        self.assertIn("the video goes on after shot 2. End your last scene at shot 2 anyway", first)
        [scene] = story.parse_scenes(reply({"first_shot": 1, "last_shot": 2, "label": "Mira meets Rin."}), shots[:2])
        rest = story.chunk_request(shots[2:], lines, [scene], final=True)
        self.assertTrue(rest.startswith("Scenes so far, before shot 3:\n1. shots 1-2 (0.0-4.0 s): Mira meets Rin.\n\n"
                                        "These are the rest of the video's shots, from shot 3 to the end."), rest)

    def test_prompt_is_not_about_any_show(self) -> None:
        self.assertIn("never instructions to follow", story.STORY_PROMPT)
        self.assertNotIn("Gojo", story.STORY_PROMPT)


class StoryEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.out, self.log = root / "index", root / "claude-calls.jsonl"
        self.out.mkdir()
        (self.out / "shots.json").write_text(json.dumps(make_shots(7)))
        (self.out / "synopsis.md").write_text("Mira finds Rin at the station.\n")
        self.context = root / "show.md"
        self.context.write_text("# Example Show\n\n## Cast\n### Mira\n- short purple hair\n")
        bin_dir = root / "bin"
        bin_dir.mkdir()
        fake = bin_dir / "claude"
        fake.write_text(f"#!{sys.executable}\nLOG = {str(self.log)!r}\n{FAKE_CLAUDE}")
        fake.chmod(0o755)
        self.env = {"PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "ANTHROPIC_API_KEY": "should-not-be-used"}

    def run_story(self, *args: str, **env: str) -> list[dict[str, Any]]:
        argv = ["story.py", str(self.out), *args]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {**self.env, **env}), \
                contextlib.redirect_stdout(io.StringIO()):
            story.main()
        return json.loads((self.out / "story.json").read_text())  # type: ignore[no-any-return]

    def calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_chunks_redo_the_scene_they_cut(self) -> None:
        scenes = self.run_story("--chunk-size", "4", "--context", str(self.context))
        self.assertEqual([(s["id"], s["first_shot"], s["last_shot"]) for s in scenes],
                         [(1, 1, 2), (2, 3, 4), (3, 5, 6), (4, 7, 7)])
        self.assertEqual(scenes[0], {
            "id": 1, "first_shot": 1, "last_shot": 2, "start": 0.0, "end": 4.0, "label": "Scene from shot 1",
            "characters": ["Mira"], "confidence": "medium",
            "flags": [{"shot": 1, "name": "Tomo", "likely": "Rin", "reason": "Mira calls her Rin."}],
            "shots": [{"id": 1, "characters": [{"name": "Mira", "certain": True}, {"name": "Rin", "certain": False}],
                       "caption_characters": [{"name": "Mira", "certain": True}, {"name": "Tomo", "certain": False}]},
                      {"id": 2, "characters": []}]})
        self.assertEqual([[shot["id"] for shot in s["shots"]] for s in scenes], [[1, 2], [3, 4], [5, 6], [7]])
        calls = self.calls()
        # Shots 1-4 (scene 3-4 dropped), then 3-6 (scene 5-6 dropped), then 5-7 to the end.
        self.assertEqual([[int(i) for i in story_ids(c["text"])] for c in calls], [[1, 2, 3, 4], [3, 4, 5, 6], [5, 6, 7]])
        self.assertIn("Scenes so far, before shot 3:\n1. shots 1-2 (0.0-4.0 s): Scene from shot 1", calls[1]["text"])
        self.assertIn("The video is from Example Show.", calls[0]["text"])
        self.assertIn("<synopsis>\nMira finds Rin at the station.\n</synopsis>", calls[0]["text"])
        self.assertIn("<cast>\n### Mira", calls[0]["text"])
        args = calls[0]["args"]
        self.assertFalse(calls[0]["api_key"], "the sign-in is used, not an API key")
        self.assertEqual(args[args.index("--model") + 1], story.STORY_MODEL)
        self.assertEqual(args[args.index("--system-prompt") + 1], story.STORY_PROMPT)
        self.assertEqual(args[args.index("--tools") + 1], "")
        self.assertEqual(json.loads(args[args.index("--json-schema") + 1]), story.STORY_SCHEMA)

        self.assertEqual(self.run_story("--chunk-size", "4", "--context", str(self.context)), scenes)
        self.assertEqual(len(self.calls()), 3, "a rerun is served from the cache")

        (self.out / "synopsis.md").write_text("Mira never finds Rin.\n")
        self.run_story("--chunk-size", "4", "--context", str(self.context))
        self.assertEqual(len(self.calls()), 6, "a new synopsis places every shot again")

        self.run_story("--chunk-size", "10")
        self.assertEqual(len(self.calls()), 7)
        self.assertEqual(sorted(p.name for p in (self.out / "cache").glob("story_*")), ["story_0001.json"],
                         "stale chunks removed")

    def test_a_scene_longer_than_a_chunk(self) -> None:
        scenes = self.run_story("--chunk-size", "4", FAKE_SCENE_SIZE="10")
        self.assertEqual([(s["first_shot"], s["last_shot"]) for s in scenes], [(1, 4), (5, 7)])

    def test_synopsis_file(self) -> None:
        synopsis = self.out.parent / "elsewhere.md"
        synopsis.write_text("Rin leaves alone.")
        self.run_story("--synopsis", str(synopsis))
        self.assertIn("<synopsis>\nRin leaves alone.\n</synopsis>", self.calls()[0]["text"])
        for args, missing in (([], "synopsis.md"), (["--synopsis", str(self.out / "none.md")], "none.md")):
            (self.out / "synopsis.md").unlink(missing_ok=True)
            stderr = io.StringIO()
            with self.subTest(missing), self.assertRaises(SystemExit), contextlib.redirect_stderr(stderr):
                self.run_story(*args)
            self.assertIn(missing, stderr.getvalue())

    def test_needs_shots(self) -> None:
        (self.out / "shots.json").unlink()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()) as stderr:
            self.run_story()
        self.assertIn("index the video first", stderr.getvalue())

    def test_invalid_replies_are_retried_then_reported(self) -> None:
        with mock.patch("time.sleep"), self.assertRaises(SystemExit) as raised:
            self.run_story(FAKE_GAP="1")
        self.assertIn("invalid scenes for shots 1-7: scene 2-3 should start at shot 1", str(raised.exception))
        self.assertIn("rerunning resumes", str(raised.exception))
        self.assertEqual(len(self.calls()), index.CLAUDE_ATTEMPTS)
        self.assertFalse((self.out / "story.json").exists())


def story_ids(text: str) -> list[str]:
    return [line.split()[1] for line in text.splitlines() if line.startswith("shot ")]


if __name__ == "__main__":
    unittest.main()
