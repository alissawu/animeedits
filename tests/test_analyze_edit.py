import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scene_frames"))
import analyze_edit as ae  # noqa: E402

# The reference edit is not committed. Point ANALYZE_EDIT_VIDEO at it from another checkout.
VIDEO = Path(os.environ.get("ANALYZE_EDIT_VIDEO", ROOT / "denreze-whitelight.mov"))
# Measured by hand in breakdowns/denreze-white-light.md.
HARD_CUTS = [1.50, 6.95, 7.90, 8.65, 9.55, 10.70, 16.30, 17.90, 18.50]
FLASHES = [(3.95, 4.10), (10.95, 11.00), (11.40, 11.45), (13.15, 13.30), (15.30, 15.35),
           (15.55, 15.60), (19.30, 19.35)]  # first frame, first frame after


def write_song(path: Path, rate: int = 22050) -> None:
    """24 s at 120 BPM: quiet clicks, then a loud drop with bass at 8 s, then a lighter section at 16 s."""
    t = np.arange(24 * rate) / rate
    clicks = np.zeros_like(t)
    burst = np.random.default_rng(0).standard_normal(int(0.02 * rate)) * np.exp(-np.linspace(0, 6, int(0.02 * rate)))
    for beat in np.arange(0, 24, 0.5):
        start = int(beat * rate)
        clicks[start:start + burst.size] += burst
    a, b = t < 8, (t >= 8) & (t < 16)
    c = t >= 16
    song = (np.where(a, 0.05, np.where(b, 0.5, 0.2)) * clicks + a * 0.02 * np.sin(2 * np.pi * 880 * t)
            + b * (0.4 * np.sin(2 * np.pi * 55 * t) + 0.2 * np.sign(np.sin(2 * np.pi * 220 * t)))
            + c * 0.1 * np.sin(2 * np.pi * 440 * t))
    pcm = (np.clip(song, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())


def texture(seed: int) -> np.ndarray:
    """A smooth random picture at analysis size, so each one has its own edges."""
    coarse = np.random.default_rng(seed).integers(0, 256, (9, 16, 3), dtype=np.uint8)
    return np.asarray(Image.fromarray(coarse).resize((128, 72), Image.Resampling.BILINEAR))


class SyntheticPictureTests(unittest.TestCase):
    def test_transitions(self):
        a, b, c, d, e, glitch = (texture(seed) for seed in range(6))
        white, black = np.full_like(a, 255), np.zeros_like(a)
        blend = [(d * (1 - k / 9) + e * (k / 9)).astype(np.uint8) for k in range(1, 9)]
        frames = ([a] * 10 + [white] + [b] * 10 + [glitch] + [c] * 5 + [black] + [c] * 5
                  + [black] * 4 + [d] * 10 + [(d * 0.4).astype(np.uint8)] + [d] * 9 + blend + [e] * 10)
        picture = ae.analyze_picture(np.stack(frames), 20.0)
        self.assertEqual([(s.kind, s.start, s.end, s.enter) for s in picture.segments], [
            ("shot", 0, 10, "start"),
            ("flash", 10, 11, None),
            ("shot", 11, 21, "flash"),
            ("shot", 21, 33, "cut"),        # the one-frame glitch joins the incoming shot
            ("black", 33, 37, None),
            ("shot", 37, 57, "black"),      # a darkened frame inside a shot is not a cut
            ("shot", 57, 75, "dissolve"),
        ])
        self.assertEqual([(x.kind, x.start, x.end, x.shot) for x in picture.in_shot], [("black", 27, 28, 3)])
        self.assertEqual([(x.start, x.end, x.shot) for x in picture.dissolves], [(57, 65, 5)])
        self.assertEqual(len(picture.frames), len(frames))

    def test_short_bright_piece_is_a_flash(self):
        # Too dim to be a white flash, but far brighter than the dark shots around it.
        a, b = ((texture(seed) * 0.3).astype(np.uint8) for seed in (1, 2))
        burst = texture(3)
        picture = ae.analyze_picture(np.stack([a] * 6 + [burst] + [b] * 6), 20.0)
        self.assertEqual([(s.kind, s.start, s.end) for s in picture.segments],
                         [("shot", 0, 6), ("flash", 6, 7), ("shot", 7, 13)])


def similar_shot(seed: int, gain: float, width: int, height: int) -> np.ndarray:
    """A pale blue picture lit from the top, with its own fine detail. Shots made from it
    share their layout and palette, like two angles on one character in the same light."""
    layout = (np.array([200, 215, 240], np.float32) * np.linspace(1.0, 0.3, height)[:, None, None]
              * np.linspace(0.8, 1.0, width)[None, :, None])
    coarse = np.random.default_rng(seed).integers(0, 256, (18, 32, 3), dtype=np.uint8)
    detail = np.asarray(Image.fromarray(coarse).resize((width, height), Image.Resampling.BILINEAR), np.float32)
    return np.clip(gain * (0.75 * layout + 0.25 * detail), 0, 255)


def on_threes(picture: np.ndarray, count: int) -> list[np.ndarray]:
    """A held background with a dark figure that moves on every third frame."""
    frames = []
    for k in range(count):
        frame = picture.copy()
        x = 30 + 8 * (k // 3)
        frame[40:110, x:x + 40] *= 0.45
        frames.append(frame.astype(np.uint8))
    return frames


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not found")
class SimilarShotCutTests(unittest.TestCase):
    def test_cut_between_similar_shots(self) -> None:
        width, height, fps = 256, 144, 24
        first = on_threes(similar_shot(1, 0.8, width, height), 24)
        second = on_threes(similar_shot(2, 1.0, width, height), 24)
        # Halfway through the second shot the editor nudges the picture: a jump in every
        # pixel that keeps the shot's colours, which must not read as a cut.
        second[12:] = [np.roll(frame, (6, 8), (0, 1)) for frame in second[12:]]
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "similar.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            str(video)], input=np.stack(first + second).tobytes(), check=True)
            info = ae.probe(video)
            assert info.video is not None
            picture = ae.analyze_picture(ae.decode_frames(info), info.video.fps)
        # The two shots' layouts correlate too well for the distance test alone.
        self.assertLess(picture.frames[24].structure_dist, ae.CUT_THRESHOLD)
        self.assertEqual([(s.kind, s.start, s.end, s.enter) for s in picture.segments],
                         [("shot", 0, 24, "start"), ("shot", 24, 48, "cut")])


class TimelineTests(unittest.TestCase):
    def test_nearest(self):
        self.assertEqual(ae.nearest(1.04, [0.5, 1.0, 1.5]), ae.Nearest(1.0, 0.04))
        self.assertIsNone(ae.nearest(1.0, []))
        words = [ae.Word(1.0, 1.4, 0.9), ae.Word(2.0, 2.5, 0.9)]
        self.assertEqual(ae.nearest_word_edge(1.45, words), ae.Nearest(1.4, 0.05, "end"))

    def test_phrases_and_sections(self):
        words = [ae.Word(0.0, 0.5, 1), ae.Word(0.6, 3.9, 1), ae.Word(6.6, 9.2, 1)]
        phrases = ae.phrases_from(words)
        self.assertEqual(phrases, [ae.Phrase(0.0, 3.9, 2), ae.Phrase(6.6, 9.2, 1)])
        sections = ae.sections_from(phrases, [ae.Drop(4.0, 18.0)], [0.0, 1.5, 4.1, 6.95, 9.55, 12.0], 12.0)
        self.assertEqual([(s.start, s.end, s.phrase, s.drop is not None) for s in sections], [
            (0.0, 4.1, 1, False), (4.1, 6.95, None, True), (6.95, 9.55, 2, False), (9.55, 12.0, None, False)])


@unittest.skipUnless(VIDEO.is_file(), f"reference edit not found: {VIDEO}")
class ReferenceEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = ae.probe(VIDEO)
        cls.picture = ae.analyze_picture(ae.decode_frames(cls.info), cls.info.video.fps)
        cls.frame = 1 / cls.info.video.fps
        cls.shots = [(s.start * cls.frame, s.end * cls.frame) for s in cls.picture.segments if s.kind == "shot"]
        cls.flashes = [(s.start * cls.frame, s.end * cls.frame)
                       for s in cls.picture.segments if s.kind == "flash"]

    def close(self, a, b):
        return abs(a - b) <= self.frame + 1e-6

    def test_hard_cuts(self):
        for t in HARD_CUTS:
            self.assertTrue(any(self.close(t, start) for start, _ in self.shots), f"no shot starts at {t}")

    def test_flashes(self):
        for a, b in FLASHES:
            self.assertTrue(any(self.close(a, s) and self.close(b, e) for s, e in self.flashes),
                            f"no flash at {a}-{b}")

    def test_flashes_are_not_shots(self):
        for a, b in FLASHES:
            for start, end in self.shots:
                self.assertLessEqual(min(end, b) - max(start, a), self.frame + 1e-6,
                                     f"shot {start:.2f}-{end:.2f} covers the flash at {a}")

    def test_shots(self):
        # The breakdown's cut table has 19 shots; the kiss holds as one shot through the drop.
        self.assertEqual(len(self.shots), 19)
        kiss = [(s, e) for s, e in self.shots if s < 6.9 and e > 4.15]
        self.assertEqual(len(kiss), 1)
        self.assertTrue(self.close(kiss[0][0], 4.10) and self.close(kiss[0][1], 6.95))

    def test_dissolve(self):
        self.assertEqual(len(self.picture.dissolves), 1)
        dissolve = self.picture.dissolves[0]
        self.assertTrue(12.1 <= dissolve.start * self.frame <= 12.3)
        self.assertTrue(12.8 <= dissolve.end * self.frame <= 13.05)

    def test_audio(self):
        audio = ae.analyze_audio(VIDEO, self.info.audio_offset)
        self.assertAlmostEqual(audio.tempo, 123, delta=2)
        self.assertAlmostEqual(audio.drop.time, 4.0, delta=0.1)
        self.assertGreater(audio.drop.jump_db, 12)
        # The same soundtrack as a standalone file gives the same audio map.
        with tempfile.TemporaryDirectory() as tmp:
            song = Path(tmp) / "song.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(VIDEO), "-vn", str(song)], check=True)
            info = ae.probe(song)
            self.assertIsNone(info.video)
            alone = ae.analyze_audio(song, info.audio_offset)
        self.assertAlmostEqual(alone.tempo, audio.tempo, delta=0.5)
        self.assertAlmostEqual(alone.drop.time, audio.drop.time, delta=0.05)


class AudioOnlyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.song = Path(cls.tmp.name) / "song.wav"
        write_song(cls.song)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_audio_map(self):
        info = ae.probe(self.song)
        self.assertIsNone(info.video)
        self.assertTrue(info.has_audio)
        audio = ae.analyze_audio(self.song)
        self.assertAlmostEqual(audio.tempo, 120, delta=3)
        self.assertAlmostEqual(audio.drop.time, 8.0, delta=0.1)
        self.assertTrue(any(abs(b.time - 16) <= 0.6 for b in audio.boundaries), audio.boundaries)
        sections = ae.audio_sections(audio, [], audio.duration)
        self.assertEqual([round(s.start) for s in sections], [0, 8, 16])
        self.assertIsNotNone(sections[1].drop)

    def test_cli_writes_audio_report_only(self):
        out = Path(self.tmp.name) / "out"
        subprocess.run([sys.executable, str(ROOT / "scene_frames/analyze_edit.py"), str(self.song), str(out),
                        "--no-words"], check=True, capture_output=True)
        report = (out / "report.md").read_text()
        self.assertIn("# Audio analysis", report)
        self.assertIn("## Sections", report)
        self.assertNotIn("Cut by cut", report)
        timeline = json.loads((out / "timeline.json").read_text())
        self.assertIsNone(timeline["picture"])
        self.assertEqual(len(timeline["sections"]), 3)
        self.assertFalse((out / "sheets").exists() or (out / "frames.json").exists())

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not found")
    def test_cover_art_is_not_video(self):
        cover, mp3 = Path(self.tmp.name) / "cover.png", Path(self.tmp.name) / "song.mp3"
        Image.new("RGB", (64, 64), "red").save(cover)
        made = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(self.song), "-i", str(cover), "-map", "0",
                               "-map", "1", "-c:a", "libmp3lame", "-disposition:v", "attached_pic", str(mp3)],
                              capture_output=True)
        if made.returncode != 0:
            self.skipTest("ffmpeg cannot write mp3 with cover art")
        info = ae.probe(mp3)
        self.assertIsNone(info.video)
        self.assertTrue(info.has_audio)


if __name__ == "__main__":
    unittest.main()
