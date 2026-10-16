"""Four-stem separation of a song: drums, bass, other and vocals, for analyze_edit.py's song map.

The model is Demucs v4 htdemucs, run through audio-separator (the package separate.py already uses for voice
stems), so it brings no new dependency. The song-analysis trial (out/song-analysis-trial/report.md) ran the same
checkpoint through the demucs CLI; on the same file the stems correlate at 0.99. Its stems came out 25 ms late,
because demucs decodes an mp3 without trimming the encoder delay; these are decoded by ffmpeg, like the mix.

Stems are cached per song by file content, as mono FLAC at SAMPLE_RATE, in CACHE_DIR/htdemucs/<key>/, so every
output folder and checkout shares them. A four-minute song takes about 25 s on an M2 Max (MPS) the first time,
and the checkpoint (80 MB) downloads once to MODEL_DIR.
"""
import contextlib
from collections.abc import Callable
import hashlib
import io
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Final

import numpy as np
import numpy.typing as npt

from separate import MODEL_DIR

MODEL: Final = "htdemucs.yaml"
STEMS: Final = ("drums", "bass", "other", "vocals")
RATE: Final = 44100            # the model's sample rate
SAMPLE_RATE: Final = 22050     # cached and returned rate, the same as analyze_edit.py's mix
SHIFTS, OVERLAP = 1, 0.25      # the demucs CLI defaults the trial used
HEADROOM: Final = 0.5          # audio-separator rescales peaks above 0.9, so keep the input well under that
CACHE_DIR: Final = Path.home() / ".cache" / "animeedits" / "stems"
CACHE_VERSION: Final = 1       # bump when the cached stems would change

Audio = npt.NDArray[np.float32]
Stems = dict[str, Audio]       # stem name -> mono samples at SAMPLE_RATE


class StemError(RuntimeError):
    pass


def _ffmpeg(args: list[str]) -> bytes:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args], capture_output=True)
    if proc.returncode != 0:
        raise StemError(f"ffmpeg failed: {proc.stderr.decode(errors='replace').strip()[-500:]}")
    return proc.stdout


def cache_key(media: Path) -> str:
    """Content hash of the file plus what the stems depend on, so a copied or renamed song still hits."""
    digest = hashlib.sha256(f"{MODEL}|{SHIFTS}|{OVERLAP}|{SAMPLE_RATE}|{CACHE_VERSION}|".encode())
    with media.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:24]


def stem_dir(media: Path, cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / "htdemucs" / cache_key(media)


def write_stems(folder: Path, stems: Stems, source: Path) -> None:
    """Write a complete cache entry atomically: a folder appears with every stem or not at all."""
    import soundfile as sf
    missing = [n for n in STEMS if n not in stems]
    if missing:
        raise ValueError(f"missing stems: {', '.join(missing)}")
    folder.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=folder.parent))
    try:
        for name in STEMS:
            sf.write(tmp / f"{name}.flac", np.clip(stems[name], -1, 1), SAMPLE_RATE, subtype="PCM_24")
        (tmp / "source.json").write_text(json.dumps({"source": str(source), "model": MODEL, "rate": SAMPLE_RATE}))
        if folder.exists():
            shutil.rmtree(folder)
        tmp.replace(folder)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_stems(folder: Path, samples: int) -> Stems | None:
    """The cached stems trimmed or zero-padded to `samples`, or None when the entry is incomplete."""
    import soundfile as sf
    if not all((folder / f"{n}.flac").is_file() for n in STEMS):
        return None
    out: Stems = {}
    for name in STEMS:
        data, rate = sf.read(folder / f"{name}.flac", dtype="float32", always_2d=True)
        if rate != SAMPLE_RATE:
            return None
        out[name] = _fit(data.mean(axis=1).astype(np.float32), samples)
    return out


def _fit(y: Audio, samples: int) -> Audio:
    if y.size >= samples:
        return np.ascontiguousarray(y[:samples])
    return np.concatenate([y, np.zeros(samples - y.size, np.float32)])


def load_or_separate(media: Path, samples: int, cache_dir: Path = CACHE_DIR, *,
                     separate: Callable[[Path], Stems] | None = None) -> Stems:
    """The four stems of `media`'s first audio track, each `samples` long at SAMPLE_RATE."""
    folder = stem_dir(media, cache_dir)
    stems = read_stems(folder, samples)
    if stems is None:
        print(f"separating stems with {MODEL} (cached in {folder})", file=sys.stderr, flush=True)
        made = (separate or separate_stems)(media)
        write_stems(folder, made, media)
        stems = {n: _fit(made[n], samples) for n in STEMS}
    return stems


def separate_stems(media: Path, model_dir: Path = MODEL_DIR, verbose: bool = False) -> Stems:
    """Run htdemucs on `media`'s first audio track. Each stem is mono at SAMPLE_RATE, at its level in the mix."""
    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise StemError(f"stems need the audio-separator package ({e}); install "
                        "scene_frames/requirements.txt into the venv") from e
    work = Path(tempfile.mkdtemp(prefix="stems-"))
    try:
        source = work / "mix.wav"
        _ffmpeg(["-i", str(media), "-map", "0:a:0", "-ac", "2", "-ar", str(RATE), "-c:a", "pcm_f32le", str(source)])
        import soundfile as sf
        mix, _ = sf.read(source, dtype="float32", always_2d=True)
        if mix.size == 0:
            raise StemError(f"{media} has no audio samples")
        peak = float(np.max(np.abs(mix)))
        scale = max(1.0, peak / HEADROOM) if peak > 0 else 1.0
        if scale != 1.0:
            sf.write(source, mix / scale, RATE, subtype="FLOAT")
        model_dir.mkdir(parents=True, exist_ok=True)
        log = io.StringIO()
        with contextlib.ExitStack() as quiet:
            if not verbose:
                quiet.enter_context(contextlib.redirect_stderr(log))
            try:
                separator = Separator(log_level=logging.INFO if verbose else logging.ERROR,
                                      model_file_dir=str(model_dir), output_dir=str(work), output_format="WAV",
                                      sample_rate=RATE, use_soundfile=True,
                                      demucs_params={"segment_size": "Default", "shifts": SHIFTS,
                                                     "overlap": OVERLAP, "segments_enabled": True})
                separator.load_model(model_filename=MODEL)
                written = separator.separate(str(source), custom_output_names={
                    "Drums": "drums", "Bass": "bass", "Other": "other", "Vocals": "vocals"})
            except Exception as e:
                tail = log.getvalue().strip()[-800:]
                raise StemError(f"stem separation failed: {e}" + (f"\n{tail}" if tail else "")) from e
        names = sorted(Path(str(p)).name for p in written) if isinstance(written, list) else written
        if names != sorted(f"{n}.wav" for n in STEMS):
            raise StemError(f"the separator wrote {written!r}, expected {', '.join(STEMS)}")
        stems: Stems = {}
        for name in STEMS:
            raw = _ffmpeg(["-i", str(work / f"{name}.wav"), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"])
            stems[name] = np.frombuffer(raw, dtype="<f4").astype(np.float32) * np.float32(scale)
        return stems
    finally:
        shutil.rmtree(work, ignore_errors=True)
