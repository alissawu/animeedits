import dataclasses
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import analyze_edit as ae
import music
import rhythm


class RhythmTests(unittest.TestCase):
    def test_pickup_variable_meter_and_offset(self):
        grid = rhythm.make_grid([.5, 1., 1.5, 2., 2.5, 3., 3.5, 4.],
                                [1., 2.5], 1.5)
        self.assertEqual(grid.downbeats, [2.5, 4.])
        self.assertIsNone(grid.beats[0].bar)
        self.assertEqual(grid.beats[0].strength, "unknown")
        self.assertEqual([b.beat_in_bar for b in grid.beats], [None, 1, 2, 3, 1, 2, 3, 4])
        self.assertEqual(grid.beats[4].bar, 2)
        self.assertEqual(grid.beats[4].strength, "strong")
        self.assertEqual(grid.beats[-1].strength, "weak")
        self.assertEqual(music.from_json(rhythm.Rhythm, dataclasses.asdict(grid)), grid)

    def test_empty_and_no_downbeats(self):
        self.assertEqual(rhythm.make_grid([], []).status, "no downbeats detected")
        self.assertTrue(all(b.bar is None for b in rhythm.make_grid([1., 2.], []).beats))

    def test_invalid_predictions(self):
        for beats, downbeats in [([1., 1.], []), ([float("nan")], []),
                                  ([1., 2.], [1.5]), ([], [1.]),
                                  ([1.], [1., 1.0001]), ([-1.], [])]:
            with self.subTest(beats=beats, downbeats=downbeats), self.assertRaises(ValueError):
                rhythm.make_grid(beats, downbeats)

    def test_model_api_and_cpu_no_dbn(self):
        tracker = mock.Mock(return_value=(np.array([.5, 1., 1.5]), np.array([1.])))
        constructor = mock.Mock(return_value=tracker)
        with mock.patch.dict(sys.modules, {"beat_this.inference": types.SimpleNamespace(Audio2Beats=constructor)}):
            grid = rhythm.estimate(np.zeros(22050, np.float32), 22050, .25)
        constructor.assert_called_once_with(checkpoint_path="final0", device="cpu", dbn=False)
        self.assertEqual(grid.downbeats, [1.25])
        self.assertEqual(tracker.call_args.args[1], 22050)

    def test_cli_runs_and_reuses_separate_rhythm_cache(self):
        grid = rhythm.make_grid([.5, 1.], [1.], .25)
        audio = ae.AudioMap(2., .25, 120., None, [.5, 1.], [], [], [], [], [], [], [], [],
                            music.Harmony(1, []), [], [])
        with tempfile.TemporaryDirectory() as tmp:
            media = Path(tmp) / "audio.wav"
            media.write_bytes(b"fixture")
            out = Path(tmp) / "out"
            info = ae.MediaInfo(str(media), 2.25, None, True, .25)
            def cache(path, signature):
                if path.name == "audio.json":
                    return dataclasses.asdict(audio)
                return None
            with mock.patch.object(sys, "argv", ["analyze_edit", str(media), str(out), "--no-words"]), \
                 mock.patch.object(ae, "probe", return_value=info), \
                 mock.patch.object(ae, "cached", side_effect=cache), \
                 mock.patch.object(ae, "decode_audio", return_value=np.zeros(22050, np.float32)), \
                 mock.patch.object(rhythm, "estimate", return_value=grid) as estimate:
                ae.main()
                estimate.assert_called_once()
            # Second invocation uses the real separate cache, not another model call.
            with mock.patch.object(sys, "argv", ["analyze_edit", str(media), str(out), "--no-words"]), \
                 mock.patch.object(ae, "probe", return_value=info), \
                 mock.patch.object(ae, "cached", side_effect=lambda path, sig:
                                   dataclasses.asdict(audio) if path.name == "audio.json" else
                                   ae.json.loads(path.read_text())["data"]), \
                 mock.patch.object(rhythm, "estimate", side_effect=AssertionError("cache missed")):
                ae.main()
            result = ae.json.loads((out / "timeline.json").read_text())
            self.assertEqual(result["planning"]["downbeats"], [1.25])

    def test_json_report_and_audio_roundtrip(self):
        grid = rhythm.make_grid([.5, 1., 1.5], [1.], .5)
        audio = ae.AudioMap(2., .5, 120., None, [.5, 1.], [], [], [], [], [], [], [], [],
                            music.Harmony(1, []), [], [], grid)
        self.assertEqual(ae.audio_from_json(dataclasses.asdict(audio)), audio)
        an = ae.Analysis(ae.MediaInfo("song.wav", 2.5, None, True, .5), None, audio,
                         [], [], [], None, ae.Settings(.45, 2, (.75, 1.), .25, None, None))
        data = ae.timeline_json(an)
        self.assertEqual(data["planning"]["downbeats"], [1.5])
        self.assertEqual(data["planning"]["beat_grid"][1]["bar"], 1)
        self.assertEqual(data["audio"]["rhythm"]["beats"][1]["strength"], "strong")
        self.assertIn("| 1.500 | 1 | 1 | strong |", ae.render_report(an))
        self.assertNotIn("texture", data)


if __name__ == "__main__":
    unittest.main()
