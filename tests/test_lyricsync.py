"""Synthetic CTC paths exercise real forced_align, without downloaded models."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import lyricsync as ls


try:
    import torch
    import torchaudio
    HAS_ALIGNMENT = True
except (ImportError, OSError):
    HAS_ALIGNMENT = False


@unittest.skipUnless(HAS_ALIGNMENT, "install requirements-lyricsync.txt for optional alignment tests")
class AlignmentTests(unittest.TestCase):
    def test_normalization_and_rejection(self):
        self.assertEqual(ls.tokens("I'm here—here!"), ["I'M", "HERE", "HERE"])
        for text in ("こんにちは", "café", "123"):
            with self.assertRaises(ValueError):
                ls.tokens(text)

    def make_alignment(self):
        import torch
        labels = ["-", "|", "A", "B"]
        # Dominant blank, deliberately low reference scores. Still retain all words/repeats.
        ids = [2, 0, 2, 1, 3, 1, 2]
        emission = torch.full((1, len(ids), len(labels)), -10.)
        for t, i in enumerate(ids):
            emission[0, t, 0] = 0.
            emission[0, t, i] = -3. if i else 0.
        emission = emission.log_softmax(-1)
        return ls.align(["aa b", "a"], emission, [i * .02 for i in range(8)], labels)

    def test_real_ctc_preserves_low_score_and_repeated_words(self):
        words = self.make_alignment()
        self.assertEqual([w.text for w in words], ["aa", "b", "a"])
        self.assertEqual([w.reference_index for w in words], [0, 1, 2])
        self.assertEqual([w.line_index for w in words], [0, 0, 1])
        self.assertTrue(all(w.low_score for w in words))
        self.assertAlmostEqual(words[-1].end, .14)  # final frame's END, not its start
        with self.assertRaises(ValueError):
            ls.validate(["aa b", "a"], words[:-1], .14)

    def test_caption_words_not_filtered(self):
        words = self.make_alignment()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            ls.captions(words, out)
            srt = (out / "lyrics.srt").read_text()
            self.assertIn("aa b", srt)
            self.assertIn("\na\n", srt)
            self.assertIn("66E8FF", (out / "lyrics.ass").read_text())

    def test_inference_chunks_align_on_original_clock_and_keep_tiny_tail(self):
        import torch
        def model(wave):
            frames = max(1, int(wave.shape[1]) // 320)
            return torch.zeros((1, frames, 4)), None
        wave = np.zeros(20 * ls.RATE + 100, np.float32)
        emission, edges = ls.infer(wave, model)
        self.assertEqual(emission.shape[1] + 1, len(edges))
        self.assertEqual(edges[-1], len(wave) / ls.RATE)
        self.assertEqual(edges[-2], 20.)
        self.assertTrue(all(a < b for a, b in zip(edges, edges[1:])))

    def test_lrclib_exact_match_and_edit_guard(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps([
            dict(id=1, trackName="Song", artistName="Other", duration=120, plainLyrics="wrong"),
            dict(id=2, trackName="Song", artistName="Artist", duration=120, plainLyrics="a"),
        ]).encode()
        with mock.patch.object(ls, "urlopen", return_value=response):
            self.assertEqual(ls.fetch_reference("Song", "Artist", 120)["id"], 2)
            with self.assertRaisesRegex(ValueError, "ordered sung slice"):
                ls.fetch_reference("Song", "Artist", 30)

    def test_render_caption_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            media = out / "tone.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                            "sine=frequency=440:duration=0.5", str(media)], check=True)
            ls.captions([ls.Word("a", .1, .3, .1, 0, 0, True)], out)
            ls.render_video(media, out, .5)
            self.assertGreater((out / "lyrics.mp4").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()

class PlanningTimelineTests(unittest.TestCase):
    def test_lines_preserve_uncertain_words_and_media_offset(self):
        import analyze_edit as ae
        import lyrics
        ref = lyrics.Reference("ref.txt", "checked", [lyrics.Line(None, "a b")])
        an = ae.Analysis(ae.MediaInfo("song.wav", 5., None, True, .5),
                         None, None, [], [], [], None,
                         ae.Settings(.45, 2, (.75, 1.), .25, None, None),
                         ref, [ls.Word("a", 1., 1.2, .1, 0, 0, True),
                                     ls.Word("b", 1.3, 1.5, .9, 1, 0, False)])
        data = ae.timeline_json(an)
        line = data["aligned_lyrics"]["lines"][0]
        self.assertEqual((line["start"], line["end"]), (1.5, 2.))
        self.assertEqual([w["uncertain"] for w in line["words"]], [True, False])
        self.assertIsNone(data["planning"]["downbeats"])
        self.assertEqual(data["words"], [])
        self.assertIn("review hints", data["planning"]["voice_shifts_status"])
        scored = ae.timeline_json(an, np.ones(ae.SAMPLE_RATE * 4, np.float32))
        scored_words = scored["aligned_lyrics"]["lines"][0]["words"]
        self.assertEqual([w["uncertain"] for w in scored_words], [True, False])
        self.assertTrue(all("stress" in w and "stressed" in w for w in scored_words))
        self.assertEqual(scored["words"], [])  # never score raw recognition instead
        self.assertNotEqual(scored["aligned_lyrics"]["stress_method"], "not computed")


    def test_reused_alignment_validates_reference_and_never_renders_by_default(self):
        import lyrics
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "song.wav"
            file = root / "alignment.json"
            word = ls.Word("a", .1, .2, .1, 0, 0, True)
            from dataclasses import asdict
            file.write_text(json.dumps({"media": str(media), "words": [asdict(word)]}))
            ref = lyrics.Reference("ref.txt", "checked", [lyrics.Line(None, "a")])
            with mock.patch.object(ls, "decode", return_value=np.zeros(ls.RATE, np.float32)), \
                 mock.patch.object(ls, "render_video") as render:
                self.assertEqual(ls.align_reference(media, ref, root, root, file), [word])
                render.assert_not_called()
                wrong = lyrics.Reference("ref.txt", "checked", [lyrics.Line(None, "b")])
                with self.assertRaises(ValueError):
                    ls.align_reference(media, wrong, root, root, file)
