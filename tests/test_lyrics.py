"""Keep uncertain recognition and published references distinct."""
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import analyze_edit as ae
import lyrics


class LyricReferenceTests(unittest.TestCase):
    def test_lrc_provenance_repeats_offset_and_unknown_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "song.lrc"
            file.write_text("[ar:Artist]\n[offset:-250]\n[00:01.00][00:03.00] <00:01.00>published line\n"
                            "unplaced line\n[00:04.00]\n", encoding="utf-8-sig")
            ref = lyrics.read(file, "https://example.com/published-lyrics")
            self.assertEqual(ref.source, "https://example.com/published-lyrics")
            self.assertEqual(ref.path, str(file.resolve()))
            self.assertEqual(ref.lines, [lyrics.Line(.75, "published line"), lyrics.Line(2.75, "published line"),
                                         lyrics.Line(None, "unplaced line"), lyrics.Line(3.75, "")])
            self.assertIn("no word alignment", ref.alignment)

    def test_plain_text_never_invents_word_or_line_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "song.txt"
            file.write_text("first line\n\nsecond line\n")
            ref = lyrics.read(file)
            self.assertEqual(ref.source, str(file.resolve()))
            self.assertEqual([line.start for line in ref.lines], [None, None])

    def test_transcription_retains_uncertain_text_and_audio_offset(self):
        raw = SimpleNamespace(start=.1234, end=.5678, probability=.23456, word=" mistaken ")
        model = mock.Mock()
        model.transcribe.return_value = ([SimpleNamespace(words=[raw])], None)
        factory = mock.Mock(return_value=model)
        with mock.patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=factory)}):
            words = ae.transcribe_words(np.zeros(ae.SAMPLE_RATE, dtype=np.float32), "test-model", "en", 2)
        self.assertEqual(words, [ae.Word(2.123, 2.568, .235, "mistaken")])
        self.assertTrue(model.transcribe.call_args.kwargs["word_timestamps"])

    def test_report_and_json_keep_reference_separate_from_recognition(self):
        recognized = [ae.Word(.2, .6, .31, "misheard"), ae.Word(.7, .9, .8)]
        ref = lyrics.Reference("/reference.lrc", "published source", [lyrics.Line(.1, "different words"),
                                                                     lyrics.Line(None, "missing time")])
        an = ae.Analysis(ae.MediaInfo("/song.wav", 1, None, True, 0), None, None, recognized,
                         ae.phrases_from(recognized), [], None, ae.Settings(.45, 2, (.75, 1), .25, "test", "en", False),
                         lyrics=ref)
        data = ae.timeline_json(an)
        self.assertEqual(data["words"], [asdict(w) for w in recognized])
        self.assertEqual(data["lyrics"], asdict(ref))
        report = ae.render_report(an)
        for text in ['"misheard" (p=0.31)', '[text unknown]', 'published source', 'time unknown: missing time',
                     'no word alignment', 'different words']:
            self.assertIn(text, report)
        empty = ae.Analysis(an.info, None, None, [], [], [], None, an.settings, lyrics=ref)
        self.assertIn("word timing has not been checked", ae.render_report(empty))


if __name__ == "__main__":
    unittest.main()
