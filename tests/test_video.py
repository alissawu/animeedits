"""detect.py and index.py on a synthetic video with known cuts, a white flash and known colors."""
import contextlib
import importlib.util
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
from detect import extract_frames
spec = importlib.util.spec_from_file_location("shot_index", Path(__file__).resolve().parents[1] / "scene_frames/index.py")
assert spec and spec.loader
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)

FPS = 24
SHOTS = [(0, 36), (36, 72), (72, 108)]  # hard cuts at 36 and 72
FLASH = (72, 75)  # three white frames open shot 3
BLUE, RED, GREEN = "1a2a6c", "c03030", "30a050"
VIDEO = (
    f"color=c=0x{BLUE}:s=320x180:r={FPS}:d=1.5[bg];"
    f"color=c=white:s=80x80:r={FPS}:d=1.5,drawbox=x=10:y=10:w=30:h=40:c=black:t=fill,"
    "drawbox=x=50:y=45:w=20:h=25:c=0xff8800:t=fill[box];"
    "[bg][box]overlay=x='if(lt(n,18),40,if(lt(n,26),40+20*(n-17),200))':y=50[s1];"
    f"color=c=0x{RED}:s=320x180:r={FPS}:d=1.5[s2];"
    f"color=c=white:s=320x180:r={FPS}:d=0.125[flash];"
    f"color=c=0x{GREEN}:s=320x180:r={FPS}:d=1.375[s3];"
    "[s1][s2][flash][s3]concat=n=4:v=1:a=0,format=yuv420p[v]"
)
SUBTITLES = ("1\n00:00:00,500 --> 00:00:01,200\n<i>First line</i>\n\n"
             "2\n00:00:02,000 --> 00:00:03,400\nSpans the cut\n")


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[k:k + 2], 16) for k in (0, 2, 4)], dtype=float)


def mean_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    assert img is not None, path
    return img.reshape(-1, 3).mean(axis=0)[::-1]


def hue_distance(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def box_left(img: np.ndarray) -> int:
    """Left edge of shot 1's white box."""
    return int(np.nonzero((img.min(axis=2) > 200).any(axis=0))[0].min())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg and ffprobe")
class SyntheticVideoTests(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]
    root: Path
    mp4: Path
    mkv: Path
    shots: list[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.mp4 = cls.root / "cuts.mp4"
        ffmpeg("-filter_complex", VIDEO, "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12",
               str(cls.mp4))
        # The same picture with two audio tracks (silent "jpn", tone "eng") and an SRT track.
        srt = cls.root / "subs.srt"
        srt.write_text(SUBTITLES)
        cls.mkv = cls.root / "streams.mkv"
        ffmpeg("-filter_complex", VIDEO, "-f", "lavfi", "-t", "4.5", "-i", "anullsrc=r=48000:cl=mono",
               "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4.5", "-i", str(srt),
               "-map", "[v]", "-map", "0:a", "-map", "1:a", "-map", "2:s",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-c:a", "aac", "-c:s", "srt",
               "-metadata:s:a:0", "language=jpn", "-metadata:s:a:1", "language=eng",
               "-metadata:s:a:1", "title=English", "-metadata:s:s:0", "language=eng", str(cls.mkv))
        cls.shots = extract_frames(cls.mp4, cls.root / "detect")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_cuts(self) -> None:
        self.assertEqual([(round(s["start"] * FPS), round(s["end"] * FPS)) for s in self.shots], SHOTS)
        written = json.loads((self.root / "detect" / "scenes.json").read_text())
        self.assertEqual(written, json.loads(json.dumps(self.shots)))

    def test_edge_frames(self) -> None:
        out = self.root / "detect"
        first_frames = [FLASH[1] if a == FLASH[0] else a for a, _ in SHOTS]
        for shot, (_, b), first in zip(self.shots, SHOTS, first_frames):
            self.assertAlmostEqual(shot["first"]["time"], first / FPS, places=3)
            self.assertAlmostEqual(shot["last"]["time"], (b - 1) / FPS, places=3)
            for still in (shot["file"], shot["first"]["file"], shot["last"]["file"]):
                self.assertTrue((out / still).is_file(), still)
        colors = [(BLUE, BLUE), (RED, RED), (GREEN, GREEN)]
        for shot, (first, last) in zip(self.shots[1:], colors[1:]):
            np.testing.assert_allclose(mean_rgb(out / shot["first"]["file"]), rgb(first), atol=8)
            np.testing.assert_allclose(mean_rgb(out / shot["last"]["file"]), rgb(last), atol=8)
        # Shot 1's box starts at the left and ends at the right.
        first, last = (cv2.imread(str(out / self.shots[0][edge]["file"])) for edge in ("first", "last"))
        assert first is not None and last is not None
        self.assertEqual((box_left(first), box_left(last)), (40, 200))

    def test_flashes(self) -> None:
        self.assertEqual([s["flashes"] for s in self.shots],
                         [[], [], [{"time": FLASH[0] / FPS, "frames": FLASH[1] - FLASH[0]}]])
        curve = self.shots[2]["motion"]["curve"]
        self.assertEqual(curve[:FLASH[1] - FLASH[0] + 1], [None] * 4, "no flow into, during or out of the flash")
        self.assertIsNotNone(curve[FLASH[1] - FLASH[0] + 1])

    def test_motion(self) -> None:
        for shot, (a, b) in zip(self.shots, SHOTS):
            motion = shot["motion"]
            self.assertEqual(motion["fps"], FPS)
            self.assertEqual(len(motion["curve"]), b - a)
            self.assertIsNone(motion["curve"][0])
        cap = cv2.VideoCapture(str(self.mp4))
        lefts = [box_left(cap.read()[1]) for _ in range(SHOTS[0][1])]
        cap.release()
        moves = [k for k in range(1, len(lefts)) if lefts[k] != lefts[k - 1]]
        self.assertEqual(len(moves), 8)
        box = self.shots[0]["motion"]
        self.assertEqual(len(box["peaks"]), 1)
        self.assertIn(round(box["peaks"][0] * FPS), moves)
        for k, speed in enumerate(box["curve"][1:], 1):
            if k in moves:
                self.assertGreater(speed, 0.1, k)
            else:
                self.assertLess(speed, 0.01, k)
        self.assertEqual([s["motion"]["peaks"] for s in self.shots[1:]], [[], []])

    def test_look(self) -> None:
        for shot, color in zip(self.shots[1:], (RED, GREEN)):
            look = shot["look"]
            r, g, b = rgb(color) / 255
            hue, saturation, _ = (float(x) for x in cv2.cvtColor(np.float32([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0])
            self.assertLess(hue_distance(look["hue"], hue), 4)
            self.assertAlmostEqual(look["saturation"], saturation, delta=0.05)
            self.assertAlmostEqual(look["luma"], 0.299 * r + 0.587 * g + 0.114 * b, delta=0.03)
            top = look["palette"][0]
            self.assertGreater(top["share"], 0.95)
            np.testing.assert_allclose(rgb(top["hex"][1:]), rgb(color), atol=10)
        self.assertLess(hue_distance(self.shots[0]["look"]["hue"], 228), 15, "blue background")

    def test_probe_and_decode_streams(self) -> None:
        audio = index.probe_streams(self.mkv, "audio")
        self.assertEqual([(s["index"], s["language"], s["title"]) for s in audio],
                         [(0, "jpn", None), (1, "eng", "English")])
        subs = index.probe_streams(self.mkv, "subtitle")
        self.assertEqual([(s["index"], s["codec"], s["language"]) for s in subs], [(0, "subrip", "eng")])
        silent, tone = index.decode_audio(self.mkv, 0), index.decode_audio(self.mkv, 1)
        self.assertAlmostEqual(len(tone) / index.SAMPLE_RATE, 4.5, delta=0.1)
        self.assertLess(float(np.sqrt(np.mean(silent ** 2))), 1e-3)
        self.assertAlmostEqual(float(np.sqrt(np.mean(tone ** 2))), 0.125 / math.sqrt(2), delta=0.01)
        self.assertEqual(index.read_subtitles(self.mkv, subs[0]), [
            {"start": 0.5, "end": 1.2, "text": "First line"},
            {"start": 2.0, "end": 3.4, "text": "Spans the cut"},
        ])
        with self.assertRaises(SystemExit):
            index.pick_stream(audio, 2, "a", "--audio-stream")

    def test_index_end_to_end(self) -> None:
        """index.py with a fake Claude client: captions, subtitles, caching and shots.json."""
        out = self.root / "index"
        calls: list[dict[str, Any]] = []

        def create(**kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            ids = [int(i) for i in kwargs["messages"][0]["content"][1]["text"].split(":")[1].split(",")]
            text = json.dumps({"shots": [{"id": i, "caption": f"shot {i}", "characters": []} for i in ids]})
            return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")

        class Anthropic:
            def __init__(self, **_: Any) -> None:
                self.messages = SimpleNamespace(create=create)

            def __enter__(self) -> "Anthropic":
                return self

            def __exit__(self, *_: object) -> None:
                pass

        class CredentialsError(Exception):
            pass

        fake = ModuleType("anthropic")
        fake.Anthropic = Anthropic  # type: ignore[attr-defined]
        fake.CredentialsError = CredentialsError  # type: ignore[attr-defined]

        def run(*args: str) -> list[Any]:
            argv = ["index.py", str(self.mkv), str(out), "--no-transcribe", "--sheet-size", "2", *args]
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"anthropic": fake}), \
                    contextlib.redirect_stdout(io.StringIO()):
                index.main()
            return json.loads((out / "shots.json").read_text())  # type: ignore[no-any-return]

        def no_credentials(**_: Any) -> None:
            raise CredentialsError("no profile")

        with mock.patch.object(fake, "Anthropic", no_credentials), self.assertRaises(SystemExit) as raised:
            run()
        self.assertIn("ant auth login", str(raised.exception))
        self.assertEqual(calls, [])

        shots = run("--subs-stream", "0")
        self.assertEqual(len(calls), math.ceil(len(SHOTS) / 2))
        self.assertEqual(sorted(p.name for p in (out / "sheets").iterdir()), ["sheet_0001.jpg", "sheet_0002.jpg"])
        self.assertEqual([s["caption"] for s in shots], ["shot 1", "shot 2", "shot 3"])
        self.assertEqual(set(shots[0]), {"id", "start", "end", "keyframe", "first_frame", "last_frame", "caption",
                                         "characters", "motion", "flashes", "look", "subtitles"})
        self.assertEqual([[line["text"] for line in s["subtitles"]] for s in shots],
                         [["First line"], ["Spans the cut"], ["Spans the cut"]])
        for shot in shots:
            for still in (shot["keyframe"], shot["first_frame"], shot["last_frame"]):
                self.assertTrue((out / still["file"]).is_file())

        self.assertEqual(run("--subs-stream", "0"), shots)
        self.assertEqual(len(calls), 2, "a rerun is served from the caches")

        run("--title", "Test Pattern")
        self.assertEqual(len(calls), 4, "a new title means new captions")
        self.assertIn("The video is from Test Pattern.", calls[-1]["system"][0]["text"])

        context = self.root / "story.md"
        cast = "## Cast\n- Box: white square, orange and black patches\n"
        context.write_text(f"# Test Pattern\n\n{cast}")
        run("--context", str(context))
        self.assertEqual(len(calls), 6, "the title comes from the # heading; the new cast means new captions")
        self.assertIn("<cast>\n- Box: white square", calls[-1]["system"][0]["text"])

        context.write_text(f"# Test Pattern\n\n{cast}\n## Scenes\n- Three color fields.\n")
        run("--context", str(context))
        self.assertEqual(len(calls), 6, "only the cast is in the prompt, so other sections don't invalidate it")
        self.assertNotIn("Three color fields", calls[-1]["system"][0]["text"])

        run("--context", str(context), "--sheet-size", "3")
        self.assertEqual(len(calls), 7)
        self.assertEqual([p.name for p in (out / "sheets").iterdir()], ["sheet_0001.jpg"], "stale sheets removed")

    def test_index_needs_audio_to_transcribe(self) -> None:
        argv = ["index.py", str(self.mp4), str(self.root / "no-audio")]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as raised:
            index.main()
        self.assertIn("--no-transcribe", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
