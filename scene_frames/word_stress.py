"""Relative acoustic emphasis, not lexical stress or a calibrated probability.

Only aligned lyric words are annotated. Their times are on the media clock;
the mono Demucs vocal signal starts at audio_offset.
"""
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

METHOD = "vocal RMS onset-rise + start RMS peak + duration; line/song z-scores v1"
THRESHOLD = 0.65


def _z(values: NDArray[np.float64]) -> NDArray[np.float64]:
    spread = values.std(axis=0)
    return (values - values.mean(axis=0)) / np.where(spread > 1e-8, spread, 1.0)


def annotate(lines: Sequence[dict[str, Any]], vocals: NDArray[np.float32],
             rate: int, audio_offset: float = 0.0) -> None:
    """In-place word annotations; retain every word and its uncertainty.

    20 ms RMS frames / 10 ms hop. Peak and positive RMS rise in the
    first 160 ms (or word end); onset search allows 30 ms alignment jitter.
    Equal line/song normalization; 45% onset, 45% level, 10% duration.
    Silence/out-of-signal words get zero, never prominence from duration alone.
    """
    if rate <= 0 or vocals.ndim != 1 or not np.isfinite(vocals).all():
        raise ValueError("expected finite mono vocals and a positive sample rate")
    words = [word for line in lines for word in line.get("words", [])]
    if not words:
        return
    width, hop = max(1, round(rate * .02)), max(1, round(rate * .01))
    # Cumulative energy avoids allocating an entire sliding-window matrix.
    energy = np.concatenate(([0.0], np.cumsum(vocals.astype(np.float64) ** 2)))
    starts = np.arange(0, len(vocals), hop)
    ends = np.minimum(starts + width, len(vocals))
    rms = np.sqrt((energy[ends] - energy[starts]) / np.maximum(ends - starts, 1))
    times = (starts + (ends - starts) / 2) / rate
    rise = np.maximum(0, np.diff(rms, prepend=0))
    features = np.zeros((len(words), 3), dtype=np.float64)
    active = np.zeros(len(words), dtype=bool)
    for i, word in enumerate(words):
        start, end = float(word["start"]) - audio_offset, float(word["end"]) - audio_offset
        if not np.isfinite([start, end]).all() or end < start:
            raise ValueError("word times must be finite and ordered")
        duration = end - start
        word["duration"] = round(duration, 6)
        a, b = np.searchsorted(times, [max(0, start - .03), min(end, start + .16)])
        peak = float(rms[a:b].max()) if b > a else 0.0
        onset = float(rise[a:b].max()) if b > a else 0.0
        active[i] = peak > 1e-6 and duration > 0 and start < len(vocals) / rate and end > 0
        features[i] = np.log1p([onset * 1000, peak * 1000, duration])
    song_z = np.zeros_like(features)
    if active.any():
        song_z[active] = _z(features[active])
    line_z = np.zeros_like(features)
    cursor = 0
    for line in lines:
        count = len(line.get("words", []))
        indices = np.arange(cursor, cursor + count)
        indices = indices[active[indices]]
        if len(indices):
            line_z[indices] = _z(features[indices])
        cursor += count
    combined = (.5 * song_z + .5 * line_z) @ np.array([.45, .45, .10])
    scores = 1 / (1 + np.exp(-np.clip(combined, -20, 20)))
    for i, word in enumerate(words):
        score = float(scores[i]) if active[i] else 0.0
        word["stress"] = round(score, 6)
        word["stressed"] = score >= THRESHOLD
