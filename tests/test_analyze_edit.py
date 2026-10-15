import os
from pathlib import Path
import sys
import unittest

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
        sections = ae.sections_from(phrases, 4.0, [0.0, 1.5, 4.1, 6.95, 9.55, 12.0], 12.0)
        self.assertEqual([(s.start, s.end, s.phrase, s.drop) for s in sections], [
            (0.0, 4.1, 1, False), (4.1, 6.95, None, True), (6.95, 9.55, 2, False), (9.55, 12.0, None, False)])


@unittest.skipUnless(VIDEO.is_file(), f"reference edit not found: {VIDEO}")
class ReferenceEditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.info = ae.probe(VIDEO)
        cls.picture = ae.analyze_picture(ae.decode_frames(cls.info), cls.info.fps)
        cls.frame = 1 / cls.info.fps
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
        self.assertAlmostEqual(audio.drop, 4.0, delta=0.1)
        self.assertGreater(audio.drop_jump_db, 12)


if __name__ == "__main__":
    unittest.main()
