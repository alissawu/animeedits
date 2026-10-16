import copy
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import word_stress


class StressTests(unittest.TestCase):
    def fixture(self):
        rate = 1000
        signal = np.zeros(4000, dtype=np.float32)
        words = []
        for i, amplitude in enumerate([.05, .05, .8, .05]):
            start = i + .1
            signal[round(start * rate):round((start + .3) * rate)] = amplitude
            words.append(dict(text=str(i), start=start, end=start + .3, uncertain=True))
        return rate, signal, [dict(words=words)]

    def test_loud_attack_is_prominent_and_uncertainty_retained(self):
        rate, signal, lines = self.fixture()
        word_stress.annotate(lines, signal, rate)
        words = lines[0]["words"]
        self.assertTrue(words[2]["stressed"])
        self.assertGreater(words[2]["stress"], words[0]["stress"])
        self.assertTrue(all(w["uncertain"] for w in words))
        self.assertTrue(all(0 <= w["stress"] <= 1 for w in words))

    def test_offset_gain_and_determinism(self):
        rate, signal, lines = self.fixture()
        shifted = copy.deepcopy(lines)
        for w in shifted[0]["words"]:
            w["start"] += 2
            w["end"] += 2
        word_stress.annotate(lines, signal, rate)
        word_stress.annotate(shifted, signal, rate, 2)
        self.assertEqual([w["stress"] for w in lines[0]["words"]],
                         [w["stress"] for w in shifted[0]["words"]])

    def test_silence_empty_out_of_bounds_and_singleton(self):
        lines = [dict(words=[dict(start=0., end=2.), dict(start=5., end=6.)])]
        word_stress.annotate(lines, np.zeros(100, np.float32), 100)
        self.assertTrue(all(w["stress"] == 0 and not w["stressed"] for w in lines[0]["words"]))
        word_stress.annotate([], np.array([], np.float32), 100)
        singleton = [dict(words=[dict(start=.1, end=.3)])]
        word_stress.annotate(singleton, np.ones(100, np.float32), 100)
        self.assertEqual(singleton[0]["words"][0]["stress"], .5)

    def test_empty_signal_and_bad_input(self):
        lines = [dict(words=[dict(start=0., end=.5)])]
        word_stress.annotate(lines, np.array([], np.float32), 100)
        self.assertEqual(lines[0]["words"][0]["stress"], 0)
        with self.assertRaises(ValueError):
            word_stress.annotate(lines, np.array([np.nan], np.float32), 100)
        with self.assertRaises(ValueError):
            word_stress.annotate([dict(words=[dict(start=2., end=1.)])], np.zeros(300, np.float32), 100)
