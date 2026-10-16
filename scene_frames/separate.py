"""Voice separation: the dialogue of a soundtrack without its score or sound effects.

render.py uses this for audio layers with stem "voice". The model is BS-RoFormer (the ViperX 1297
checkpoint, audio-separator's default). Of the separators compared on Chainsaw Man and Jujutsu Kaisen 0
dialogue under their own scores, it let the least score through and kept the voice at full level;
docs/pipeline.md#voice-stems has the comparison. torch and the model load on first use, and the
checkpoint (about 640 MB) downloads once to MODEL_DIR.
"""
import contextlib
import io
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import TracebackType
from typing import Any

import numpy as np
import numpy.typing as npt

MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
RATE = 44100  # the model's sample rate
OVERLAP = 2  # chunks per window; the model's default of 4 measured the same on the test lines at twice the time
MODEL_DIR = Path.home() / ".cache" / "animeedits" / "separator"
HEADROOM = 0.5  # audio-separator rescales peaks above 0.9, so keep the input well under that

Audio = npt.NDArray[np.float32]


class SeparationError(RuntimeError):
    pass


def _ffmpeg(args: list[str], data: bytes | None = None) -> bytes:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args],
                          input=data, capture_output=True)
    if proc.returncode != 0:
        raise SeparationError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()[-500:]}")
    return proc.stdout


class VoiceSeparator:
    """Loads the model on the first call and keeps it until the `with` block ends."""

    def __init__(self, model_dir: Path = MODEL_DIR, verbose: bool = False) -> None:
        self.model_dir = model_dir
        self.verbose = verbose
        self._work: Path | None = None
        self._model: Any = None

    def __enter__(self) -> "VoiceSeparator":
        return self

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: TracebackType | None) -> None:
        self._model = None
        if self._work is not None:
            shutil.rmtree(self._work, ignore_errors=True)
            self._work = None

    def separate(self, mix: Audio) -> Audio:
        """The voice in `mix`, a (channels, samples) float32 array at RATE, at the level it has in `mix`."""
        if mix.ndim != 2 or mix.shape[0] not in (1, 2) or mix.shape[1] == 0:
            raise ValueError(f"expected (1 or 2 channels, samples) audio, got shape {mix.shape}")
        if not np.all(np.isfinite(mix)):
            raise ValueError("audio contains NaN or infinity")
        channels, samples = mix.shape
        peak = float(np.max(np.abs(mix)))
        scale = max(1.0, peak / HEADROOM)
        work = self._workdir()
        source, voice = work / "mix.wav", work / "voice.wav"
        voice.unlink(missing_ok=True)
        _ffmpeg(["-f", "f32le", "-ar", str(RATE), "-ac", str(channels), "-i", "-", "-c:a", "pcm_f32le", str(source)],
                np.ascontiguousarray((mix / scale).T, dtype="<f4").tobytes())
        model = self._load()  # outside the redirect, so a first-use download shows its progress
        log = io.StringIO()
        with contextlib.ExitStack() as quiet:
            if not self.verbose:  # one progress bar per call otherwise
                quiet.enter_context(contextlib.redirect_stderr(log))
            try:
                written = model.separate(str(source), custom_output_names={"Vocals": "voice"})
            except Exception as e:
                tail = log.getvalue().strip()[-800:]
                raise SeparationError(f"voice separation failed: {e}" + (f"\n{tail}" if tail else "")) from e
        if not isinstance(written, list) or [Path(str(p)).name for p in written] != [voice.name] or not voice.is_file():
            raise SeparationError(f"the separator wrote {written!r}, expected only {voice.name}")
        raw = _ffmpeg(["-i", str(voice), "-f", "f32le", "-ar", str(RATE), "-ac", str(channels), "-"])
        out = np.frombuffer(raw, dtype="<f4").reshape(-1, channels).T * np.float32(scale)
        if abs(out.shape[1] - samples) > RATE // 100:
            raise SeparationError(f"the separator returned {out.shape[1]} samples for {samples}")
        result = np.zeros((channels, samples), dtype=np.float32)
        keep = min(samples, out.shape[1])
        result[:, :keep] = out[:, :keep]
        return result

    def _workdir(self) -> Path:
        if self._work is None:
            self._work = Path(tempfile.mkdtemp(prefix="voice-separator-"))
        return self._work

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from audio_separator.separator import Separator
        except ImportError as e:
            raise SeparationError(f"voice stems need the audio-separator package ({e}); install "
                                  "scene_frames/requirements.txt into the venv") from e
        self.model_dir.mkdir(parents=True, exist_ok=True)
        if not (self.model_dir / MODEL).exists():
            print(f"downloading the voice separation model (about 640 MB) to {self.model_dir}", file=sys.stderr)
        # Its warnings are notes (e.g. that it adapts the chunk size to clips under 10 s), not problems.
        separator = Separator(log_level=logging.INFO if self.verbose else logging.ERROR,
                              model_file_dir=str(self.model_dir), output_dir=str(self._workdir()),
                              output_format="WAV", output_single_stem="Vocals", sample_rate=RATE,
                              use_soundfile=True,  # float WAV in, float WAV out; pydub would write 16-bit
                              mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                           "batch_size": 1, "overlap": OVERLAP, "pitch_shift": 0})
        separator.load_model(model_filename=MODEL)
        self._model = separator
        return separator
