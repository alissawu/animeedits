"""render.py and edl.py on tiny synthetic sources. Needs ffmpeg on PATH."""
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import edl as edl_module
from edl import EdlError, Output, parse_edl
import render
import separate
from separate import SeparationError, VoiceSeparator

FPS = 20
W, H = 72, 128  # 9:16 output from 64x36 (16:9) sources
SR = render.SAMPLE_RATE

# Output frames (20 fps): red 0-19 | white flash 20-22, green 23-39 | green->blue crossfade 40-43,
# blue 44-59 | black 60-63 | black->red crossfade 64-67, red 68-79, fade to black 74-79.
EDL: dict[str, Any] = {
    "title": "Synthetic",
    "output": {"width": W, "height": H, "fps": FPS},
    "song": {"source": "song.wav", "src_in": 0.5, "src_out": 4.5},
    "clips": [
        {"source": "red.mp4", "src_in": 0.0, "src_out": 1.0, "timeline_in": 0.0,
         "sync": "free", "why": "Open on red."},
        {"source": "green.mp4", "src_in": 1.0, "src_out": 2.0, "timeline_in": 1.0,
         "sync": "drop", "why": "White flash into green on the drop.",
         "transition": {"type": "flash", "color": "white", "frames": 3}},
        {"source": "blue.mp4", "src_in": 0.5, "src_out": 1.5, "timeline_in": 2.0,
         "sync": "beat", "why": "Dissolve to blue.", "transition": {"type": "crossfade", "frames": 4}},
        {"source": "red.mp4", "src_in": 2.0, "src_out": 2.8, "timeline_in": 3.2,
         "sync": "phrase_end", "why": "Back to red out of black.", "transition": {"type": "crossfade", "frames": 4}},
    ],
    "audio": [{"source": "dialogue.mp4", "src_in": 1.0, "src_out": 2.0, "timeline_in": 1.5,
               "duck_db": -12, "why": "The 1 kHz line."}],
    "overlays": [{"source": "dot.png", "timeline_in": 0.5, "timeline_out": 0.75, "width": 0.25}],
    "captions": [{"text": "Hello there world", "timeline_in": 0.25, "timeline_out": 2.95,
                  "style": {"size": 0.12, "y": 0.85}, "reveal": {"unit": "word", "seconds": 0.5}}],
    "ending": {"fade_to_black": 0.3},
}

# 24 fps ramp source whose luma encodes the source frame number: Y = 16 + 4 * frame.
RAMP_EDL: dict[str, Any] = {
    "output": {"width": 32, "height": 32, "fps": 24},
    "song": {"source": "song.wav", "src_in": 0.0, "src_out": 3.0, "speed": 2.0},
    "clips": [
        {"source": "ramp.mp4", "src_in": 0.5, "src_out": 1.0, "timeline_in": 0.0, "sync": "free", "why": "forward"},
        {"source": "ramp.mp4", "src_in": 0.5, "src_out": 1.0, "timeline_in": 0.5, "sync": "free", "why": "reversed",
         "reverse": True},
        {"source": "ramp.mp4", "src_in": 0.0, "src_out": 1.0, "timeline_in": 1.0, "sync": "motion", "why": "double speed",
         "speed": 2.0},
    ],
}

# Retiming and keyframed transforms at 24 fps, 90x160. ramp.mp4 frames read back as source frame numbers.
# 0-23 ramp (0.5x -> 2x linear, -> 0.5x in_out) | 24-29 freeze | 30-47 play, hold, play |
# 48-59 half speed blended | 60-71 half speed motion-interpolated | 72-95 zoom punch 2x -> 1x with decaying shake
RETIME_EDL: dict[str, Any] = {
    "output": {"width": 90, "height": 160, "fps": 24},
    "song": {"source": "song.wav", "src_in": 0.0, "src_out": 4.0},
    "clips": [
        {"source": "ramp.mp4", "src_in": 0.25, "timeline_in": 0.0, "timeline_out": 1.0, "sync": "beat", "why": "ramp",
         "keyframes": [{"t": 0.0, "speed": 0.5}, {"t": 0.5, "speed": 2.0},
                       {"t": 1.0, "speed": 0.5, "ease": "in_out"}]},
        {"source": "ramp.mp4", "src_in": 0.5, "timeline_in": 1.0, "timeline_out": 1.25, "sync": "drop",
         "why": "freeze", "speed": 0},
        {"source": "ramp.mp4", "src_in": 0.0, "timeline_in": 1.25, "timeline_out": 2.0, "sync": "word", "why": "hold",
         "keyframes": [{"t": 0.0, "speed": 1}, {"t": 0.25, "speed": 0, "ease": "hold"},
                       {"t": 0.5, "speed": 1, "ease": "hold"}]},
        {"source": "ramp.mp4", "src_in": 0.0, "timeline_in": 2.0, "timeline_out": 2.5, "sync": "free",
         "why": "blend", "speed": 0.5, "slowmo": "blend"},
        {"source": "ramp.mp4", "src_in": 0.0, "timeline_in": 2.5, "timeline_out": 3.0,
         "speed": 0.5, "slowmo": "motion"},
        {"source": "box.mp4", "src_in": 0.0, "timeline_in": 3.0, "timeline_out": 4.0, "sync": "motion",
         "why": "punch", "shake_hz": 10,
         "keyframes": [{"t": 0.0, "zoom": 2.0, "shake": 0.04}, {"t": 0.5, "zoom": 1.0, "shake": 0.0, "ease": "out"}]},
    ],
}


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


def make_sources(root: Path) -> None:
    for name, color in (("red", "red"), ("green", "lime"), ("blue", "blue")):
        ffmpeg("-f", "lavfi", "-i", f"color=c={color}:s=64x36:r=24:d=4", "-c:v", "libx264", "-preset", "ultrafast",
               "-crf", "18", "-pix_fmt", "yuv420p", str(root / f"{name}.mp4"))
    ffmpeg("-f", "lavfi", "-i", "color=c=lime:s=64x36:r=24:d=3", "-f", "lavfi", "-i", "sine=f=1000:d=3:sample_rate=48000",
           "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
           str(root / "dialogue.mp4"))
    ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=6:sample_rate=44100", str(root / "song.wav"))
    ffmpeg("-f", "lavfi", "-i", "color=c=yellow:s=16x16", "-frames:v", "1", str(root / "dot.png"))
    ffmpeg("-f", "lavfi", "-i", "color=c=black:s=64x36:r=24:d=2,geq=lum='16+4*N':cb=128:cr=128",
           "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p", str(root / "ramp.mp4"))
    ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=320x180:r=24:d=2,drawbox=x=140:y=70:w=40:h=40:color=red:t=fill",
           "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", str(root / "box.mp4"))


def decode_rgb(video: Path, width: int, height: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3).astype(int)


def decode_luma(video: Path, width: int, height: int) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, width * height * 3 // 2)
    return frames[:, :width * height].reshape(-1, height, width).astype(int)


def decode_audio(video: Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32)


def tone(audio: np.ndarray, freq: float, start: float, end: float) -> float:
    """Amplitude of a sine at `freq` in [start, end) seconds."""
    chunk = audio[round(start * SR):round(end * SR)]
    t = np.arange(len(chunk)) / SR
    return float(2 * abs(np.dot(chunk, np.exp(-2j * np.pi * freq * t))) / len(chunk))


def write_edl(root: Path, name: str, edl: dict[str, Any]) -> Path:
    path = root / name
    path.write_text(json.dumps(edl, indent=2))
    return path


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class RenderTests(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]
    root: Path
    summary: str
    frames: np.ndarray
    audio: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="render-test-")
        cls.root = Path(cls.tmp.name)
        make_sources(cls.root)
        cls.summary = render.render(write_edl(cls.root, "edit.json", EDL), cls.root / "out" / "edit.mp4", jobs=2)
        cls.frames = decode_rgb(cls.root / "out" / "edit.mp4", W, H)
        cls.audio = decode_audio(cls.root / "out" / "edit.mp4")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def color(self, frame: int) -> np.ndarray:
        """Mean color of an area no caption or overlay touches."""
        return self.frames[frame, 8:40, 8:64].reshape(-1, 3).mean(0)

    def assertColor(self, frame: int, rgb: tuple[float, float, float], tol: float = 10) -> None:
        got = self.color(frame)
        self.assertTrue(np.all(abs(got - rgb) <= tol), f"frame {frame}: got {got.round()}, expected {rgb}")

    def test_length_matches_song_window(self) -> None:
        self.assertEqual(len(self.frames), 80)
        self.assertLess(abs(len(self.audio) - 4.0 * SR), 1024)

    def test_hard_cut_and_flash_land_on_exact_frames(self) -> None:
        red, white, green = (250, 0, 0), (253, 253, 253), (0, 252, 0)
        for frame in range(20):
            self.assertColor(frame, red)
        for frame in (20, 21, 22):
            self.assertColor(frame, white)
        for frame in range(23, 40):
            self.assertColor(frame, green)

    def test_crossfade_blends_over_its_frames(self) -> None:
        self.assertColor(39, (0, 252, 0))
        for k, frame in enumerate(range(40, 44)):
            w = (k + 1) / 5
            self.assertColor(frame, (0, 252 * (1 - w), 253 * w))
        self.assertColor(44, (0, 0, 253))

    def test_gap_fade_from_black_and_ending(self) -> None:
        self.assertColor(59, (0, 0, 253))
        for frame in range(60, 64):
            self.assertColor(frame, (0, 0, 0))
        for k, frame in enumerate(range(64, 68)):  # fade up from black
            self.assertColor(frame, (250 * (k + 1) / 5, 0, 0))
        for k, frame in enumerate(range(74, 80)):  # fade to black, last frame black
            self.assertColor(frame, (250 * (5 - k) / 6, 0, 0))

    def test_overlay_shows_only_in_its_range(self) -> None:
        centre = self.frames[:, 58:70, 30:42].reshape(len(self.frames), -1, 3).mean(1)
        yellow = [bool(np.all(abs(c - (250, 250, 0)) < 20)) for c in centre]
        self.assertEqual([f for f, y in enumerate(yellow) if y], list(range(10, 15)))

    def test_caption_types_on_word_by_word(self) -> None:
        band = self.frames[:, 88:128]
        background = self.frames[:, 8:40, 8:64].reshape(len(self.frames), -1, 3).mean(1)
        ink = [int((abs(band[f] - background[f]).sum(axis=2) > 120).sum()) for f in range(len(self.frames))]
        self.assertEqual(ink[4], 0)
        self.assertEqual(len({ink[f] for f in range(5, 10)}), 1)  # word 1 from frame 5
        self.assertGreater(ink[10], ink[9])  # word 2 at frame 10
        self.assertGreater(ink[15], ink[14])  # word 3 at frame 15
        self.assertEqual(ink[15], ink[19])
        self.assertGreater(ink[58], 0)
        self.assertEqual(ink[59], 0)  # gone after timeline_out

    def test_dialogue_layer_ducks_the_song(self) -> None:
        self.assertLess(tone(self.audio, 1000, 0.5, 1.4), 0.005)
        self.assertGreater(tone(self.audio, 1000, 1.6, 2.4), 0.05)
        self.assertLess(tone(self.audio, 1000, 2.6, 3.5), 0.005)
        ratio = tone(self.audio, 440, 1.6, 2.4) / tone(self.audio, 440, 0.5, 1.4)
        self.assertAlmostEqual(ratio, 10 ** (-12 / 20), delta=0.04)

    def test_writes_proof_and_table(self) -> None:
        out = self.root / "out"
        table = (out / "edl_table.md").read_text()
        for clip in EDL["clips"]:
            self.assertIn(clip["why"], table)
        self.assertIn("| 2 | 1.000–2.000 | 20 | green.mp4 | 1.000–2.000 | 1× | white flash 3f | drop |", table)
        self.assertIn("| | 3.000–3.200 | 4 | black |", table)
        with Image.open(out / "proof.jpg") as im:
            self.assertGreater(im.height, 5 * 128)

    def test_rerender_reuses_segments_and_ignores_why(self) -> None:
        out = self.root / "reworded"
        shutil.copytree(self.root / "out" / render.CACHE_DIR, out / render.CACHE_DIR)
        edl = copy.deepcopy(EDL)
        edl["clips"][0]["why"] = "Reworded reasoning."
        summary = render.render(write_edl(self.root, "reworded.json", edl), out / "edit.mp4")
        self.assertIn("0 segments rendered", summary)
        self.assertIn("Reworded reasoning.", (out / "edl_table.md").read_text())

    def test_source_timing_reverse_and_speed(self) -> None:
        out = self.root / "ramp" / "ramp.mp4"
        render.render(write_edl(self.root, "ramp.json", RAMP_EDL), out)
        luma = decode_luma(out, 32, 32)[:, 8:24, 8:24].mean(axis=(1, 2))
        self.assertEqual(len(luma), 36)
        source_frame = np.round((luma - 16) / 4).astype(int).tolist()
        self.assertEqual(source_frame[:12], list(range(12, 24)))
        self.assertEqual(source_frame[12:24], list(range(23, 11, -1)))
        self.assertEqual(source_frame[24:], list(range(0, 24, 2)))
        song = decode_audio(out)
        self.assertGreater(tone(song, 880, 0.2, 1.2), 5 * tone(song, 440, 0.2, 1.2))  # 2x speed doubles pitch


def ramp_advance(t: float) -> float:
    """Source seconds after `t` s of RETIME_EDL's ramp clip, integrated numerically (Simpson, exact for cubics)."""
    def speed(x: float) -> float:
        if x < 0.5:
            return 0.5 + 1.5 * x / 0.5
        u = (x - 0.5) / 0.5
        return 2.0 - 1.5 * u * u * (3 - 2 * u)

    def simpson(a: float, b: float, steps: int = 64) -> float:
        h = (b - a) / steps
        return h / 3 * sum((1 if k in (0, steps) else 4 if k % 2 else 2) * speed(a + k * h) for k in range(steps + 1))

    return simpson(0, min(t, 0.5)) + (simpson(0.5, t) if t > 0.5 else 0.0)


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class RetimeTests(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]
    root: Path
    luma: np.ndarray
    frames: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="retime-test-")
        cls.root = Path(cls.tmp.name)
        make_sources(cls.root)
        out = cls.root / "out" / "retime.mp4"
        render.render(write_edl(cls.root, "retime.json", RETIME_EDL), out, jobs=3)
        cls.luma = decode_luma(out, 90, 160)
        cls.frames = decode_rgb(out, 90, 160)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def source_frames(self, start: int, end: int) -> list[float]:
        """Source frame shown by output frames [start, end) of the ramp clips (luma = 16 + 4 * frame)."""
        return [float(v) for v in (self.luma[start:end, 40:120, 20:70].mean(axis=(1, 2)) - 16) / 4]

    def test_speed_ramp_follows_the_integrated_curve(self) -> None:
        self.assertEqual(len(self.luma), 96)
        expected = [6 + int(np.floor(ramp_advance(j / 24) * 24 + 1e-6)) for j in range(24)]
        self.assertEqual([round(v) for v in self.source_frames(0, 24)], expected)
        self.assertEqual(expected[:3], [6, 6, 7])  # starts at half speed: frames repeat
        self.assertEqual(expected[12] - expected[11], 2)  # the middle runs at 2x: frames skip

    def test_freeze_and_hold(self) -> None:
        self.assertEqual([round(v) for v in self.source_frames(24, 30)], [12] * 6)
        self.assertEqual([round(v) for v in self.source_frames(30, 48)],
                         [0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 7, 8, 9, 10, 11])

    def test_smooth_slow_motion_makes_in_between_frames(self) -> None:
        halves = [j / 2 for j in range(12)]
        for start in (48, 60):  # blend, then motion interpolation
            got = self.source_frames(start, start + 12)
            self.assertTrue(all(abs(g - h) < 0.2 for g, h in zip(got, halves)), (start, got))

    def test_zoom_punch_and_shake(self) -> None:
        def box(frame: int) -> tuple[float, float, float]:
            f = self.frames[frame]
            ys, xs = np.nonzero((f[:, :, 0] > 150) & (f[:, :, 1] < 100))
            return float(xs.max() - xs.min() + 1), float(xs.mean()), float(ys.mean())

        widths = [box(f)[0] for f in range(72, 96)]
        self.assertAlmostEqual(widths[0], 40 / (180 * 9 / 16 / 2) * 90, delta=3)  # zoom 2 on the first frame
        self.assertTrue(all(b <= a + 1 for a, b in zip(widths, widths[1:12])), widths)  # zooming out
        rest = [box(f) for f in range(84, 96)]  # from t = 0.5: zoom 1, no shake, centred
        for width, cx, cy in rest:
            self.assertAlmostEqual(width, 40 / (180 * 9 / 16) * 90, delta=3)
            self.assertAlmostEqual(cx, 44.5, delta=1)
            self.assertAlmostEqual(cy, 79.5, delta=1)
        shaken = [max(abs(box(f)[1] - 44.5), abs(box(f)[2] - 79.5)) for f in range(73, 82)]
        self.assertGreater(max(shaken), 1.5)

    def test_derived_src_out_is_reported_and_checked(self) -> None:
        summary = render.render(self.root / "retime.json", None)
        self.assertIn("clips[0].src_out = 1.5000 (derived from timeline_out and speed)", summary)
        edl = copy.deepcopy(RETIME_EDL)
        edl["clips"][0]["src_out"] = 1.0
        with self.assertRaises(EdlError) as ctx:
            parse_edl(edl)
        self.assertIn("clips[0].src_out: 1 s does not match the speed, which reaches source 1.5000 s by timeline_out; "
                      "set src_out to 1.5000 or leave it out", ctx.exception.problems)
        edl["clips"][0]["src_out"] = 1.5
        parse_edl(edl)

    def test_table_describes_motion(self) -> None:
        table = (self.root / "out" / "edl_table.md").read_text()
        self.assertIn("| 1 | 0.000–1.000 | 24 | ramp.mp4 | 0.250–1.500 | ramp 0.5–2× | cut | beat | ramp |", table)
        self.assertIn("| freeze |", table)
        self.assertIn("1×, keys zoom/shake", table)
        (unannotated,) = [row for row in table.splitlines() if row.startswith("| 5 |")]
        self.assertTrue(unannotated.endswith("| cut |  |  |"), unannotated)


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class LongClipTests(unittest.TestCase):
    """A clip with more than 100 kept source frames, from a 23.976 fps source like most anime."""

    def test_long_clip_renders_every_source_frame_in_order(self) -> None:
        with tempfile.TemporaryDirectory(prefix="long-clip-test-") as tmp:
            root = Path(tmp)
            # Luma encodes the source frame number: Y = 16 + frame.
            ffmpeg("-f", "lavfi", "-i", "color=c=black:s=32x32:r=24000/1001:d=7,geq=lum='16+N':cb=128:cr=128",
                   "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p",
                   str(root / "long.mp4"))
            ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=7:sample_rate=44100", str(root / "song.wav"))
            edl = {
                "output": {"width": 32, "height": 32, "fps": 24},
                "song": {"source": "song.wav", "src_in": 0.0, "src_out": 6.0},
                "clips": [{"source": "long.mp4", "src_in": 0.0, "timeline_in": 0.0, "timeline_out": 6.0}],
            }
            out = root / "out" / "long.mp4"
            render.render(write_edl(root, "long.json", edl), out)
            luma = decode_luma(out, 32, 32)[:, 8:24, 8:24].mean(axis=(1, 2))
            shown = np.round(luma - 16).astype(int).tolist()
            self.assertEqual(len(shown), 144)
            steps = np.diff(shown).tolist()
            self.assertTrue(set(steps) <= {0, 1}, steps)  # never skips or goes back
            self.assertLessEqual(steps.count(0), 1)  # 23.976 into 24 repeats at most one frame in 6 s
            self.assertGreaterEqual(shown[-1] - shown[0], 142)


def keep_band(mix: np.ndarray, freq: float, width: float = 300) -> np.ndarray:
    """Only the part of `mix` (channels, samples at separate.RATE) within `width` Hz of `freq`."""
    spectrum = np.fft.rfft(mix, axis=1)
    freqs = np.fft.rfftfreq(mix.shape[1], 1 / separate.RATE)
    spectrum[:, np.abs(freqs - freq) > width] = 0
    return np.fft.irfft(spectrum, n=mix.shape[1], axis=1).astype(np.float32)


# Output 0-4 s over a 440 Hz song. film.mp4's soundtrack: a 3 kHz "score" throughout and a 1 kHz "voice"
# from 2.0 to 3.0 s. Layer 0 plays source 1.5-3.5 from 0.5 s, so the voice lands at 1.0-2.0 s; layer 1
# plays source 0.3-0.8 (no voice) from 3.0 s, where the context before it is cut short by the source start.
STEM_EDL: dict[str, Any] = {
    "output": {"width": 32, "height": 32, "fps": 24},
    "song": {"source": "song.wav", "src_in": 0.0, "src_out": 4.0},
    "clips": [{"source": "red.mp4", "src_in": 0.0, "timeline_in": 0.0, "timeline_out": 4.0}],
    "audio": [{"source": "film.mp4", "src_in": 1.5, "src_out": 3.5, "timeline_in": 0.5, "stem": "voice",
               "fade_in": 0.0, "fade_out": 0.0, "why": "the line"},
              {"source": "film.mp4", "src_in": 0.3, "src_out": 0.8, "timeline_in": 3.0, "stem": "voice"}],
}


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class VoiceStemTests(unittest.TestCase):
    """stem "voice" with a stand-in separator that keeps only 1 kHz, so no model is needed."""
    tmp: tempfile.TemporaryDirectory[str]
    root: Path
    calls: list[int]
    summary: str
    voice: np.ndarray
    mix: np.ndarray

    @staticmethod
    def fake(separator: VoiceSeparator, mix: np.ndarray) -> np.ndarray:
        VoiceStemTests.calls.append(mix.shape[1])
        return keep_band(mix, 1000)

    @classmethod
    def render(cls, edl: dict[str, Any], name: str) -> str:
        with mock.patch.object(VoiceSeparator, "separate", cls.fake):
            return render.render(write_edl(cls.root, f"{name}.json", edl), cls.root / name / "edit.mp4")

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="stem-test-")
        cls.root = Path(cls.tmp.name)
        cls.calls = []
        ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x36:r=24:d=5", "-c:v", "libx264", "-preset", "ultrafast",
               "-pix_fmt", "yuv420p", str(cls.root / "red.mp4"))
        ffmpeg("-f", "lavfi", "-i", "sine=f=440:d=5:sample_rate=44100,volume=0.25", str(cls.root / "song.wav"))
        ffmpeg("-f", "lavfi", "-i", "color=c=gray:s=64x36:r=24:d=5", "-filter_complex",
               "sine=f=3000:d=5:sample_rate=48000[s];"
               "sine=f=1000:d=1:sample_rate=48000,volume=2,adelay=2000:all=1,apad=whole_dur=5[v];"
               "[s][v]amix=inputs=2:normalize=0[a]",
               "-map", "0:v", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "256k", "-shortest", str(cls.root / "film.mp4"))
        cls.summary = cls.render(STEM_EDL, "voice")
        cls.voice = decode_audio(cls.root / "voice" / "edit.mp4")
        mixed = copy.deepcopy(STEM_EDL)
        for layer in mixed["audio"]:
            del layer["stem"]
        render.render(write_edl(cls.root, "mix.json", mixed), cls.root / "mix" / "edit.mp4")
        cls.mix = decode_audio(cls.root / "mix" / "edit.mp4")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_keeps_the_voice_and_drops_the_score(self) -> None:
        self.assertGreater(tone(self.mix, 3000, 0.6, 2.4), 0.05)
        self.assertLess(tone(self.voice, 3000, 0.6, 2.4), 0.002)
        self.assertLess(tone(self.voice, 3000, 3.05, 3.45), 0.002)
        level = tone(self.voice, 1000, 1.1, 1.9) / tone(self.mix, 1000, 1.1, 1.9)
        self.assertAlmostEqual(level, 1.0, delta=0.03)
        self.assertGreater(tone(self.voice, 440, 0.1, 0.4), 0.02)  # the song is untouched

    def test_voice_lands_where_the_mix_would(self) -> None:
        a = keep_band(self.voice[None, round(0.8 * SR):round(1.2 * SR)], 1000)[0]
        b = keep_band(self.mix[None, round(0.8 * SR):round(1.2 * SR)], 1000)[0]
        lags = range(-48, 49)
        lag = max(lags, key=lambda d: float(np.dot(a[48 + d:len(a) - 48 + d], b[48:len(b) - 48])))
        self.assertLessEqual(abs(lag), 2)
        self.assertLess(tone(self.voice, 1000, 0.9, 0.98), 0.01)
        self.assertGreater(tone(self.voice, 1000, 1.02, 1.1), 0.1)

    def test_separator_hears_context_clamped_to_the_source(self) -> None:
        rate = separate.RATE
        expected = [(2.0 + 2 * render.STEM_CONTEXT) * rate, (0.5 + 0.3 + render.STEM_CONTEXT) * rate]
        self.assertEqual(len(self.calls), 2)
        for got, want in zip(self.calls, expected):
            self.assertLessEqual(abs(got - want), rate // 1000)
        self.assertIn("2 voice stems separated, 0 cached", self.summary)
        self.assertIn("film.mp4 (track 0, voice)", (self.root / "voice" / "edl_table.md").read_text())

    def test_stems_are_cached_by_source_span(self) -> None:
        out = self.root / "cached"
        shutil.copytree(self.root / "voice" / render.CACHE_DIR, out / render.CACHE_DIR)
        edl = copy.deepcopy(STEM_EDL)
        edl["audio"][0].update(gain_db=-3.0, duck_db=-6.0, timeline_in=0.6, why="moved and quieter")
        before = len(self.calls)
        self.assertIn("0 voice stems separated, 2 cached", self.render(edl, "cached"))
        self.assertEqual(len(self.calls), before)
        edl["audio"][0]["src_in"] = 1.6
        self.assertIn("1 voice stems separated, 1 cached", self.render(edl, "cached"))
        self.assertEqual(len(list((out / render.CACHE_DIR).glob("stem_*.wav"))), 2)  # the stale one is gone

    def test_audio_near_the_source_start_is_not_shifted(self) -> None:
        film = self.root / "film.mp4"
        for start in (0.0, 0.01, 1.5):  # an input seek this close to the start would drop 21 ms of AAC
            spec, trim = render.audio_source(film, render.probe(film), start, 3.0)
            raw = subprocess.run(["ffmpeg", "-v", "error", *spec, "-map", "0:a:0", "-af", f"{trim},aresample={SR}",
                                  "-ac", "1", "-f", "f32le", "-"], capture_output=True, check=True).stdout
            audio = np.frombuffer(raw, np.float32)
            self.assertLess(abs(len(audio) / SR - (3.0 - start)), 0.001, start)
            onset = 2.0 - start  # the 1 kHz voice starts at source 2.0 s
            self.assertLess(tone(audio, 1000, onset - 0.03, onset - 0.005), 0.01, start)
            self.assertGreater(tone(audio, 1000, onset + 0.005, onset + 0.03), 0.1, start)

    def test_separator_failure_names_the_layer(self) -> None:
        def broken(separator: VoiceSeparator, mix: np.ndarray) -> np.ndarray:
            raise SeparationError("model exploded")
        with mock.patch.object(VoiceSeparator, "separate", broken), self.assertRaises(render.RenderError) as ctx:
            render.render(write_edl(self.root, "broken.json", STEM_EDL), self.root / "broken" / "edit.mp4")
        self.assertIn("separating the voice in film.mp4 from 1.500 s: model exploded", str(ctx.exception))
        self.assertFalse(list((self.root / "broken" / render.CACHE_DIR).glob("stem_*")))


def make_song_sources(root: Path) -> None:
    """steps.wav: 440 Hz for 2 s, then 660, then 880. chord.wav: 200 Hz + 5 kHz. tone.wav: 1 kHz. beep.wav: a
    0.3 s 3 kHz sound effect. All at 48 kHz, with a red picture to cut to."""
    ffmpeg("-f", "lavfi", "-i", "color=c=red:s=64x36:r=24:d=8", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", str(root / "red.mp4"))
    for name, expr, seconds in (("steps", "0.1*sin(2*PI*if(lt(t,2),440,if(lt(t,4),660,880))*t)", 6),
                                ("chord", "0.05*sin(2*PI*200*t)+0.05*sin(2*PI*5000*t)", 7),
                                ("tone", "0.1*sin(2*PI*1000*t)", 3),
                                ("beep", "0.1*sin(2*PI*3000*t)", 0.3)):
        ffmpeg("-f", "lavfi", "-i", f"aevalsrc='{expr}':d={seconds}:s=48000", str(root / f"{name}.wav"))


def song_edl(song: dict[str, Any], seconds: float, audio: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"output": {"width": 32, "height": 32, "fps": 24}, "song": song,
            "clips": [{"source": "red.mp4", "src_in": 0.0, "timeline_in": 0.0, "timeline_out": seconds}],
            "audio": audio or []}


# 0-1 s open | 1-2 low-passed at 500 Hz | 2-3 open | 3-4 at -12 dB | 4-5 high-passed at 1 kHz | 5-7 open, but
# the beep at 5.25-5.55 muffles the song to 500 Hz while it plays.
SHAPED_EDL = song_edl({"source": "chord.wav", "src_in": 0.0, "src_out": 7.0, "keyframes": [
    {"t": 0.0, "gain_db": 0, "lowpass": 20000, "highpass": 20},
    {"t": 1.0, "lowpass": 500, "ease": "hold"},
    {"t": 2.0, "lowpass": 20000, "ease": "hold"},
    {"t": 3.0, "gain_db": -12, "ease": "hold"},
    {"t": 4.0, "gain_db": 0, "highpass": 1000, "ease": "hold"},
    {"t": 5.0, "highpass": 20, "ease": "hold"}]}, 7.0,
    [{"source": "beep.wav", "timeline_in": 5.25, "muffle_hz": 500, "fade_in": 0, "why": "an impact"}])


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class SongShapingTests(unittest.TestCase):
    """Splices, gain and filter keyframes, muffles, tails and sound effects, rendered as audio alone."""
    tmp: tempfile.TemporaryDirectory[str]
    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="song-test-")
        cls.root = Path(cls.tmp.name)
        make_song_sources(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def mix(self, edl: dict[str, Any], name: str) -> np.ndarray:
        out = self.root / name / "mix.wav"
        summary = render.render(write_edl(self.root, f"{name}.json", edl), out)
        self.assertIn("audio only", summary)
        self.assertFalse((self.root / name / "proof.jpg").exists())
        return decode_audio(out)

    def test_pieces_join_at_their_output_time_without_clicks(self) -> None:
        audio = self.mix(song_edl({"source": "steps.wav", "pieces": [{"src_in": 0.5, "src_out": 1.5},
                                                                      {"src_in": 4.5, "src_out": 5.5}]}, 2.0), "splice")
        self.assertEqual(len(audio), 2 * SR)
        first, second = tone(audio, 440, 0.1, 0.9), tone(audio, 880, 1.1, 1.9)
        self.assertGreater(first, 0.05)
        self.assertAlmostEqual(second / first, 1.0, delta=0.02)
        self.assertLess(tone(audio, 660, 0.0, 2.0), 0.002)  # the verse in between is cut out
        self.assertGreater(tone(audio, 440, 0.98, 0.997), 5 * tone(audio, 880, 0.98, 0.997))
        self.assertGreater(tone(audio, 880, 1.003, 1.02), 5 * tone(audio, 440, 1.003, 1.02))
        # The two sines meet out of phase; the 5 ms crossfade keeps every step a sine's own size.
        steps = np.abs(np.diff(audio[round(0.99 * SR):round(1.01 * SR)]))
        self.assertLess(float(steps.max()), 1.2 * 2 * np.pi * 880 / SR * second)

    def test_gain_and_filter_keys_and_muffle_under_a_layer(self) -> None:
        audio = self.mix(SHAPED_EDL, "shaped")
        high, low = tone(audio, 5000, 0.1, 0.9), tone(audio, 200, 0.1, 0.9)
        self.assertGreater(high, 0.02)
        self.assertAlmostEqual(tone(audio, 200, 0.1, 0.9) / low, 1.0)

        def ratio(freq: float, start: float, end: float) -> float:
            return tone(audio, freq, start, end) / (high if freq == 5000 else low)

        self.assertLess(ratio(5000, 1.1, 1.9), 0.01)  # muffled
        self.assertAlmostEqual(ratio(200, 1.1, 1.9), 1.0, delta=0.03)  # with the bass intact
        self.assertAlmostEqual(ratio(5000, 2.1, 2.9), 1.0, delta=0.03)  # open again
        self.assertAlmostEqual(ratio(5000, 3.1, 3.9), 10 ** (-12 / 20), delta=0.01)
        self.assertAlmostEqual(ratio(200, 3.1, 3.9), 10 ** (-12 / 20), delta=0.01)
        self.assertLess(ratio(200, 4.1, 4.9), 0.01)  # thinned
        self.assertAlmostEqual(ratio(5000, 4.1, 4.9), 1.0, delta=0.03)
        self.assertLess(ratio(5000, 5.27, 5.53), 0.01)  # muffled under the sound effect
        self.assertAlmostEqual(ratio(200, 5.27, 5.53), 1.0, delta=0.03)
        self.assertGreater(tone(audio, 3000, 5.27, 5.53), 0.05)  # which is itself not muffled
        self.assertLess(tone(audio, 3000, 4.8, 5.2), 0.002)
        self.assertAlmostEqual(ratio(5000, 5.7, 6.9), 1.0, delta=0.03)
        table = (self.root / "shaped" / "edl_table.md").read_text()
        self.assertIn("| 2 | 1.000 | hold |  | 500 Hz |  |", table)
        self.assertIn("| 5.250–5.550 | beep.wav (track 0) | 0.000–0.300 | +0 dB | 0 dB, muffled to 500 Hz | an impact |", table)

    def test_filter_slope_sets_how_steeply_the_low_pass_cuts(self) -> None:
        song = {"source": "chord.wav", "src_in": 0.0, "src_out": 1.0, "keyframes": [{"t": 0.0, "lowpass": 500}]}
        for slope in (12, 48):
            audio = self.mix(song_edl({**song, "filter_slope": slope}, 1.0), f"slope{slope}")
            # A Butterworth low-pass of order n passes 1 / sqrt(1 + (f / fc) ** 2n) at f.
            expected = 1 / math.sqrt(1 + (5000 / 500) ** (slope // 3))
            self.assertAlmostEqual(tone(audio, 5000, 0.2, 0.9) / tone(audio, 200, 0.2, 0.9), expected,
                                   delta=0.3 * expected + 1e-4)

    def test_echo_tail_repeats_the_end_after_the_song_stops(self) -> None:
        audio = self.mix(song_edl({"source": "tone.wav", "src_in": 0.0, "src_out": 1.0,
                                   "tail": {"seconds": 1.0, "input": 0.1, "delay": 0.25, "feedback": 0.5}}, 2.0), "echo")
        self.assertEqual(len(audio), 2 * SR)
        dry = tone(audio, 1000, 0.2, 0.8)
        self.assertGreater(dry, 0.05)
        self.assertLess(tone(audio, 1000, 1.01, 1.14), 0.002)  # the song has stopped
        self.assertAlmostEqual(tone(audio, 1000, 1.16, 1.24) / dry, 1.0, delta=0.03)  # first repeat at wet_db 0
        self.assertLess(tone(audio, 1000, 1.26, 1.39), 0.002)
        self.assertAlmostEqual(tone(audio, 1000, 1.41, 1.49) / dry, 0.5, delta=0.02)
        self.assertAlmostEqual(tone(audio, 1000, 1.66, 1.74) / dry, 0.25, delta=0.02)
        self.assertLess(float(np.abs(audio[-SR // 100:]).max()), 0.002)  # faded to silence

    def test_reverb_tail_decays(self) -> None:
        audio = self.mix(song_edl({"source": "tone.wav", "src_in": 0.0, "src_out": 1.0,
                                   "tail": {"type": "reverb", "seconds": 1.0, "decay": 1.5, "wet_db": -6}}, 2.0),
                         "reverb")

        def rms(start: float, end: float) -> float:
            return float(np.sqrt(np.mean(audio[round(start * SR):round(end * SR)] ** 2)))

        levels = [rms(1.05, 1.15), rms(1.3, 1.4), rms(1.55, 1.65)]
        self.assertGreater(levels[0], 0.1 * rms(0.2, 0.8))
        self.assertGreater(levels[0], levels[1])
        self.assertGreater(levels[1], levels[2])
        self.assertGreater(levels[2], 0.0005)
        self.assertTrue((self.root / "reverb" / "render_cache" / "reverb.wav").exists())

    def test_mp4_carries_the_same_mix(self) -> None:
        edl = copy.deepcopy(SHAPED_EDL)
        edl["song"]["tail"] = {"seconds": 0.5, "type": "reverb", "decay": 0.5}
        edl["clips"][0]["timeline_out"] = 7.5
        wav = self.mix(edl, "both")
        render.render(write_edl(self.root, "both.json", edl), self.root / "both" / "edit.mp4")
        mp4 = decode_audio(self.root / "both" / "edit.mp4")
        self.assertEqual(len(mp4), len(wav))
        for start in np.arange(0.1, 7.3, 0.4):
            for freq in (200, 5000):
                a, b = tone(wav, freq, start, start + 0.2), tone(mp4, freq, start, start + 0.2)
                self.assertLess(abs(a - b), 0.05 * max(a, b) + 2e-4, f"{freq} Hz at {start:.1f} s")

    def test_sound_effect_plays_its_whole_file_and_must_fit(self) -> None:
        edl = song_edl({"source": "tone.wav", "src_in": 0.0, "src_out": 1.0}, 1.0,
                       [{"source": "beep.wav", "timeline_in": 0.5, "gain_db": -6, "fade_in": 0, "fade_out": 0}])
        audio = self.mix(edl, "sfx")
        self.assertLess(tone(audio, 3000, 0.3, 0.48), 0.002)
        self.assertAlmostEqual(tone(audio, 3000, 0.52, 0.78) / tone(audio, 1000, 0.1, 0.4), 0.5, delta=0.02)
        self.assertLess(tone(audio, 3000, 0.82, 1.0), 0.002)
        edl["audio"][0]["timeline_in"] = 0.875
        with self.assertRaises(EdlError) as ctx:
            render.render(write_edl(self.root, "late.json", edl), None)
        self.assertEqual(ctx.exception.problems, [
            "audio[0]: ends at 1.175 s, after the output ends at 1.000 s; to cut it there, set src_out to 0.1250"])

    def test_sources_cover_the_crossfade_handles(self) -> None:
        edl = song_edl({"source": "tone.wav", "pieces": [{"src_in": 0.0, "src_out": 3.0},
                                                         {"src_in": 0.0, "src_out": 1.0, "crossfade": 0.1}]}, 4.0)
        with self.assertRaises(EdlError) as ctx:
            render.render(write_edl(self.root, "handles.json", edl), None)
        self.assertIn("song.pieces[1].src_in: the 0.100 s crossfade into this piece plays the source from -0.050 s; "
                      "start the piece later or shorten the crossfade", ctx.exception.problems)
        edl["song"]["pieces"][1]["src_in"] = 0.05
        edl["song"]["pieces"][1]["src_out"] = 1.05
        with self.assertRaises(EdlError) as ctx:
            render.render(write_edl(self.root, "handles.json", edl), None)
        self.assertEqual(ctx.exception.problems, ["song.pieces[0].src_out: 3.050 s is past the end of tone.wav (3.000 s) "
                                                  "(with the 0.1 s crossfade out of it)"])


class SongCurveTests(unittest.TestCase):
    def test_cutoffs_sweep_in_octaves_and_muffles_take_the_lowest(self) -> None:
        edl = parse_edl(song_edl({"source": "chord.wav", "src_in": 0.0, "src_out": 4.0, "keyframes": [
            {"t": 0.0, "lowpass": 20000}, {"t": 1.0, "lowpass": 500}]}, 4.0,
            [{"source": "beep.wav", "src_out": 0.3, "timeline_in": 2.0, "muffle_hz": 1000}]))
        self.assertAlmostEqual(edl_module.song_lowpass(edl, 0.5), math.sqrt(20000 * 500), places=6)
        self.assertAlmostEqual(edl_module.song_lowpass(edl, 1.5), 500)
        self.assertAlmostEqual(edl_module.song_lowpass(edl, 2.1), 500)  # the keyframes are lower than the muffle
        self.assertAlmostEqual(edl_module.song_highpass(edl.song, 1.0), 20)

    def test_commands_follow_the_curve_and_skip_holds(self) -> None:
        edl = copy.deepcopy(SHAPED_EDL)
        edl["audio"][0]["src_out"] = 0.3  # render.resolve_layers fills this in from the file
        text = render.song_command_text(parse_edl(edl))
        assert text is not None
        lines = text.splitlines()
        times = [float(line.split()[0]) for line in lines]
        self.assertEqual(times, sorted(times))
        at_one = [line for line in lines if abs(float(line.split()[0]) - 1.0) < 0.006]
        self.assertTrue(any("lowpass@song_lowpass0 frequency 500.000, lowpass@song_lowpass0 mix 1" in line
                            for line in at_one), at_one)
        # Each hold sends one command and steady stretches none; only the muffle's ramps move block by block.
        # A value changes at the start of the BLOCK nearest its time.
        held = [t for t in times if not 5.1 < t < 5.7]
        self.assertEqual(len(held), 5)
        for t, expected in zip(held, [1.0, 2.0, 3.0, 4.0, 5.0]):
            self.assertLessEqual(abs(t - expected), render.BLOCK / 2 / SR + 1e-6)
        block = render.BLOCK / SR
        ramps = [t for t in times if 5.1 < t < 5.7]
        self.assertTrue(all(5.17 - block < t < 5.25 + block or 5.55 - block < t < 5.63 + block for t in ramps), ramps)
        self.assertIsNone(render.song_command_text(parse_edl(EDL)))
        with tempfile.TemporaryDirectory() as tmp:
            awkward = Path(tmp) / "a b, c:d's [x].txt"
            awkward.write_text("0.5 volume@song_gain volume 0.5;\n")
            out = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=d=1:sample_rate=48000",
                                  "-af", ",".join(["asetnsamples=n=256:p=0", f"asendcmd=f={render.filter_arg(str(awkward))}",
                                                   "volume@song_gain=volume=1:eval=once"]),
                                  "-f", "f32le", "-"], capture_output=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            samples = np.frombuffer(out.stdout, np.float32)
            self.assertAlmostEqual(float(np.abs(samples[SR // 2 + 256:]).max()) / float(np.abs(samples[:SR // 2]).max()),
                                   0.5, delta=0.01)


class EdlValidationTests(unittest.TestCase):
    def problems(self, edl: dict[str, object]) -> list[str]:
        with self.assertRaises(EdlError) as ctx:
            parse_edl(edl)
        return ctx.exception.problems

    def test_valid_edl_keeps_reasoning(self) -> None:
        edl = parse_edl(EDL)
        self.assertEqual([c.sync for c in edl.clips], ["free", "drop", "beat", "phrase_end"])
        self.assertEqual(edl.clips[1].why, "White flash into green on the drop.")

    def test_reasoning_is_optional(self) -> None:
        edl = copy.deepcopy(EDL)
        del edl["clips"][0]["why"]
        del edl["clips"][0]["sync"]
        clip = parse_edl(edl).clips[0]
        self.assertIsNone(clip.sync)
        self.assertIsNone(clip.why)
        edl["clips"][1]["sync"] = "vibes"
        self.assertIn('clips[1].sync: must be one of beat, drop, word, phrase_end, motion, free, got "vibes"',
                      self.problems(edl))

    def test_audio_stem_defaults_to_the_mix(self) -> None:
        self.assertEqual(parse_edl(EDL).audio[0].stem, "mix")
        edl = copy.deepcopy(EDL)
        edl["audio"][0]["stem"] = "music"
        self.assertIn('audio[0].stem: must be one of mix, voice, got "music"', self.problems(edl))

    def test_song_shaping_rules(self) -> None:
        edl = copy.deepcopy(EDL)
        edl["song"]["pieces"] = [{"src_in": 0.5, "src_out": 2.5, "crossfade": 0.01}, {"src_in": 3.0, "src_out": 5.0}]
        self.assertIn("song: give src_in and src_out, or pieces, not both", self.problems(edl))
        del edl["song"]["src_in"], edl["song"]["src_out"]
        self.assertEqual(self.problems(edl), ["song.pieces[0].crossfade: the first piece has no join before it"])
        del edl["song"]["pieces"][0]["crossfade"]
        self.assertEqual(parse_edl(edl).song.pieces[1].crossfade, None)
        edl["song"]["keyframes"] = [{"t": 0.5}, {"t": 0.4, "lowpass": 10}, {"t": 6.0, "gain_db": -3}]
        edl["song"]["tail"] = {"seconds": 1.0, "decay": 2.0, "input": 9}
        edl["audio"][0]["muffle_hz"] = 30000
        edl["song"]["filter_slope"] = 18
        problems = self.problems(edl)
        self.assertIn("song.filter_slope: must be one of 12, 24, 36, 48, got 18", problems)
        self.assertIn("song.keyframes[0]: sets none of gain_db, lowpass, highpass", problems)
        self.assertIn("song.keyframes[1].t: must be after the previous keyframe, got 0.4", problems)
        self.assertIn("song.keyframes[1].lowpass: must be from 20 to 20000, got 10", problems)
        self.assertIn("song.keyframes: the last keyframe (6.000 s) is past the 5.000 s output", problems)
        self.assertIn("song.tail.decay: is for type reverb; an echo fades by feedback", problems)
        self.assertIn("song.tail.input: must be above 0 and at most the song's 4.000 s, got 9", problems)
        self.assertIn("audio[0].muffle_hz: must be from 20 to 20000, got 30000", problems)

    def test_tail_lengthens_the_output(self) -> None:
        edl = copy.deepcopy(EDL)
        edl["song"]["tail"] = {"seconds": 0.5, "type": "reverb"}
        parsed = parse_edl(edl)
        self.assertEqual(edl_module.output_frames(parsed), 90)
        edl["song"]["tail"]["feedback"] = 0.3
        self.assertIn("song.tail: delay and feedback are for type echo; a reverb fades by decay", self.problems(edl))

    def test_unknown_field_suggests_fix(self) -> None:
        edl = copy.deepcopy(EDL)
        edl["clips"][0]["src_ot"] = 1.0
        self.assertIn("clips[0].src_ot: unknown field (did you mean 'src_out'?)", self.problems(edl))

    def test_overlap_and_overrun(self) -> None:
        edl = copy.deepcopy(EDL)
        edl["clips"][0]["src_out"] = 1.1
        edl["clips"][3]["src_out"] = 3.2
        problems = self.problems(edl)
        self.assertTrue(any(p.startswith("clips[1]: starts at 1.000 s (frame 20), 2 frame(s) before clips[0] ends")
                            and "clips[0].src_out = 1.0000" in p for p in problems), problems)
        self.assertTrue(any(p.startswith("clips[3]: ends at 4.400 s (frame 88), after the song window ends at 4.000 s")
                            for p in problems), problems)

    def test_transition_and_reveal_checks(self) -> None:
        edl = copy.deepcopy(EDL)
        edl["clips"][1]["transition"]["frames"] = 20
        edl["captions"][0]["reveal"] = {"unit": "word", "times": [0.3, 0.4]}
        problems = self.problems(edl)
        self.assertIn("clips[1].transition.frames: a flash needs 1 to 19 frames for this 20-frame clip, got 20", problems)
        self.assertIn("captions[0].reveal.times: needs one time per word (3), got 2", problems)

    def test_missing_and_short_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ffmpeg("-f", "lavfi", "-i", "color=c=red:s=32x18:r=24:d=1", "-pix_fmt", "yuv420p", str(root / "short.mp4"))
            ffmpeg("-f", "lavfi", "-i", "sine=d=6", str(root / "song.wav"))
            edl = copy.deepcopy(EDL)
            for clip in edl["clips"]:
                clip["source"] = "short.mp4"
            edl["clips"][1]["source"] = "missing.mp4"
            with self.assertRaises(EdlError) as ctx:
                render.render(write_edl(root, "edl.json", edl), None)
            problems = ctx.exception.problems
            self.assertIn(f"clips[1].source: file not found: {(root / 'missing.mp4').resolve()}", problems)
            self.assertIn("clips[3]: shows short.mp4 at 2.750 s, past its end (1.000 s)", problems)
            self.assertIn(f"audio[0].source: file not found: {(root / 'dialogue.mp4').resolve()}", problems)


    def test_retiming_rules(self) -> None:
        edl = copy.deepcopy(RETIME_EDL)
        del edl["clips"][0]["timeline_out"]
        edl["clips"][0]["src_out"] = 1.5
        del edl["clips"][1]["timeline_out"]
        edl["clips"][1]["src_out"] = 0.5
        edl["clips"][2]["keyframes"][1]["t"] = 0.0
        edl["clips"][3]["keyframes"] = [{"t": 0.1}]
        problems = self.problems(edl)
        self.assertIn("clips[0].keyframes: a speed curve needs timeline_out; src_out is derived from it", problems)
        self.assertIn("clips[1].src_out: must be after src_in (0.5), got 0.5", problems)
        self.assertIn("clips[1].speed: a freeze (speed 0) needs timeline_out", problems)
        self.assertIn("clips[2].keyframes[1].t: must be after the previous keyframe, got 0", problems)
        self.assertIn("clips[3].keyframes[0]: sets none of speed, zoom, x, y, shake", problems)

class CaptionLayoutTests(unittest.TestCase):
    def test_reveal_units_map_to_characters(self) -> None:
        caption = parse_edl(EDL).captions[0]
        font = render.load_font(None, 20, Path("."))
        lines = render.layout_caption(caption, font, Output(width=1000, height=200, fps=20))
        self.assertEqual([line.text for line in lines], ["Hello there world"])
        self.assertEqual(lines[0].words, (0,) * 5 + (-1,) + (1,) * 5 + (-1,) + (2,) * 5)
        self.assertEqual(lines[0].chars, tuple(range(5)) + (-1,) + tuple(range(5, 10)) + (-1,) + tuple(range(10, 15)))

    def test_long_words_wrap_and_split(self) -> None:
        font = render.load_font(None, 20, Path("."))
        lines = render.wrap(["a", "Supercalifragilistic", "b"], font, font.getlength("Supercal"))
        self.assertEqual([[w for w, _ in line] for line in lines][0], [0])
        self.assertEqual("".join(t for line in lines for w, t in line if w == 1), "Supercalifragilistic")
        self.assertTrue(all(font.getlength(" ".join(t for _, t in line)) <= font.getlength("Supercal") for line in lines))

if __name__ == "__main__":
    unittest.main()
