"""Joint beat/downbeat inference, independent of the legacy librosa beat grid."""
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

MODEL = "final0"
METHOD = "beat-this 1.1.0 / final0; minimal postprocessing, no DBN"


@dataclass(frozen=True)
class Beat:
    time: float
    bar: int | None
    beat_in_bar: int | None
    strength: Literal["strong", "weak", "unknown"]


@dataclass(frozen=True)
class Rhythm:
    method: str
    status: str
    downbeats: list[float]
    beats: list[Beat]


def make_grid(beats: list[float], downbeats: list[float], offset: float = 0.0) -> Rhythm:
    """Number observed bars from 1, leaving pickup positions unknown. No fixed meter."""
    if not math.isfinite(offset):
        raise ValueError("nonfinite audio offset")
    for values in (beats, downbeats):
        if any(not math.isfinite(t) or t < 0 for t in values):
            raise ValueError("beat times must be finite and nonnegative")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError("beat times must increase strictly")
    # Upstream minimal postprocessing makes downbeats a subset of beats.
    # Match only numeric jitter, never project them onto a different tracker.
    indices: set[int] = set()
    for t in downbeats:
        i = int(np.searchsorted(beats, t))
        candidates = [j for j in (i - 1, i) if 0 <= j < len(beats)]
        if not candidates:
            raise ValueError("downbeat without a beat")
        j = min(candidates, key=lambda j: abs(beats[j] - t))
        if abs(beats[j] - t) > .001 or j in indices:
            raise ValueError("downbeats must be distinct members of the beat grid")
        indices.add(j)
    bar = 0
    position = 0
    rows: list[Beat] = []
    for i, t in enumerate(beats):
        if i in indices:
            bar += 1
            position = 1
        elif bar:
            position += 1
        rows.append(Beat(round(t + offset, 3), bar or None, position or None,
                         "strong" if i in indices else "weak" if bar else "unknown"))
    return Rhythm(METHOD, "estimated" if downbeats else "no downbeats detected",
                  [rows[i].time for i in sorted(indices)], rows)


def estimate(y: NDArray[np.float32], rate: int, offset: float = 0.0) -> Rhythm:
    """Run on decoded mono audio; first use downloads the upstream model."""
    from beat_this.inference import Audio2Beats

    tracker = Audio2Beats(checkpoint_path=MODEL, device="cpu", dbn=False)
    beats, downbeats = tracker(y, rate)
    return make_grid([float(t) for t in beats], [float(t) for t in downbeats], offset)
