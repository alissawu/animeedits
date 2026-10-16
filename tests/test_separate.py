"""separate.py: the voice separator's plumbing with a stand-in model, and the real model when it is cached."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import separate
from separate import RATE, SeparationError, VoiceSeparator


def stereo_tone(seconds: float, peak: float) -> np.ndarray:
    t = np.arange(round(seconds * RATE)) / RATE
    return (peak * np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 660 * t)])).astype(np.float32)


def read_peak(path: str) -> float:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-"], capture_output=True, check=True).stdout
    return float(np.max(np.abs(np.frombuffer(raw, "<f4"))))


class StandInModel:
    """Plays audio-separator: writes the input at half level as the custom-named stem."""

    def __init__(self, out_dir: Path, written: list[str] | None = None, error: Exception | None = None) -> None:
        self.out_dir = out_dir
        self.written = written
        self.error = error
        self.peaks: list[float] = []

    def separate(self, path: str, custom_output_names: dict[str, str]) -> list[str]:
        if self.error is not None:
            print("CUDA? MPS? neither", file=sys.stderr)
            raise self.error
        self.peaks.append(read_peak(path))
        name = f"{custom_output_names['Vocals']}.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", path, "-af", "volume=0.5", "-c:a", "pcm_f32le",
                        str(self.out_dir / name)], check=True)
        return self.written if self.written is not None else [name]


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class PlumbingTests(unittest.TestCase):
    def separate_with(self, mix: np.ndarray, **kwargs: Any) -> tuple[np.ndarray, StandInModel]:
        with VoiceSeparator() as separator:
            model = StandInModel(separator._workdir(), **kwargs)
            with mock.patch.object(VoiceSeparator, "_load", lambda self: model):
                return separator.separate(mix), model

    def test_round_trip_keeps_length_and_level(self) -> None:
        mix = stereo_tone(1.5, 0.3)
        voice, model = self.separate_with(mix)
        self.assertEqual(voice.shape, mix.shape)
        self.assertEqual(voice.dtype, np.float32)
        np.testing.assert_allclose(voice, 0.5 * mix, atol=1e-6)
        self.assertAlmostEqual(model.peaks[0], 0.3, places=4)

    def test_loud_input_is_lowered_for_the_model_and_restored(self) -> None:
        mix = stereo_tone(1.0, 2.0)  # over full scale, as a downmix can be
        voice, model = self.separate_with(mix)
        self.assertAlmostEqual(model.peaks[0], separate.HEADROOM, places=4)
        np.testing.assert_allclose(voice, 0.5 * mix, atol=1e-5)

    def test_mono_stays_mono(self) -> None:
        mix = stereo_tone(0.5, 0.2)[:1]
        voice, _ = self.separate_with(mix)
        self.assertEqual(voice.shape, mix.shape)

    def test_rejects_audio_it_cannot_separate(self) -> None:
        with VoiceSeparator() as separator:
            for bad in (np.zeros((3, 100), np.float32), np.zeros((2, 0), np.float32), np.zeros(100, np.float32)):
                with self.assertRaises(ValueError):
                    separator.separate(bad)
            with self.assertRaisesRegex(ValueError, "NaN"):
                separator.separate(np.full((2, 100), np.nan, np.float32))

    def test_unexpected_output_is_an_error(self) -> None:
        with self.assertRaisesRegex(SeparationError, "expected only voice.wav"):
            self.separate_with(stereo_tone(0.5, 0.2), written=["voice.wav", "other.wav"])

    def test_model_failure_carries_its_log(self) -> None:
        with self.assertRaises(SeparationError) as ctx:
            self.separate_with(stereo_tone(0.5, 0.2), error=RuntimeError("out of memory"))
        self.assertIn("voice separation failed: out of memory", str(ctx.exception))
        self.assertIn("CUDA? MPS? neither", str(ctx.exception))

    def test_work_files_are_removed(self) -> None:
        with VoiceSeparator() as separator:
            work = separator._workdir()
            self.assertTrue(work.is_dir())
        self.assertFalse(work.exists())


MODEL_CACHED = importlib.util.find_spec("audio_separator") is not None and (separate.MODEL_DIR / separate.MODEL).is_file()


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
@unittest.skipUnless(MODEL_CACHED, f"audio-separator or {separate.MODEL} in {separate.MODEL_DIR} is missing")
class ModelTests(unittest.TestCase):
    """The real model, only when it needs no download. Checks the contract, not separation quality."""

    def test_separates_at_the_input_level(self) -> None:
        rng = np.random.default_rng(0)
        mix = (0.05 * rng.standard_normal((2, 3 * RATE)) + stereo_tone(3.0, 0.1)).astype(np.float32)
        with VoiceSeparator() as separator:
            quiet = separator.separate(mix)
            loud = separator.separate(8 * mix)  # lowered under HEADROOM for the model, then restored
        self.assertEqual(quiet.shape, mix.shape)
        self.assertTrue(np.all(np.isfinite(quiet)))
        self.assertLess(float(np.sqrt(np.mean(quiet ** 2))), 2 * float(np.sqrt(np.mean(mix ** 2))))
        ratio = float(np.sqrt(np.mean(loud ** 2)) / max(np.sqrt(np.mean(quiet ** 2)), 1e-12))
        self.assertAlmostEqual(ratio, 8, delta=0.4)


if __name__ == "__main__":
    unittest.main()
