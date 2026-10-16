"""Measurements of a song from its four stems, for analyze_edit.py's song map.

Everything here is a fact about what changes in the signal: which stems play when, where drums or bass come in
or leave, where the band drops away under the voice, where everything stops, where a stem stops repeating its
loop, what the bass root is per bar, and how loud and how high the voice sits per section. Nothing is labelled
with a mood. The methods and thresholds come from the song-analysis trial (out/song-analysis-trial/report.md),
which checked them against lane plots, the White Light breakdown and reference edits; docs/edit_analysis.md
explains each one.
"""
from collections.abc import Mapping, Sequence
import dataclasses
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import shutil
import types
import typing
import warnings
from typing import Any, Final, Literal, TypeVar, Union, cast
import urllib.request

import numpy as np
from numpy.typing import NDArray

Arr = NDArray[np.float64]
Audio = NDArray[np.float32]
Change = Literal["enters", "exits", "up", "down", "loop_break"]

STEMS: Final = ("drums", "bass", "other", "vocals")
BAND: Final = ("drums", "bass", "other")   # everything but the vocal
BLOCK: Final = 0.25           # seconds per loudness block
FINE: Final = 0.05            # seconds per block when refining an event's time
FLOOR_DB: Final = -60.0
ENVELOPE_BLOCKS: Final = 3    # 0.75 s max filter, so a sparse drum part counts as playing between hits
ABSENT_DB: Final = -42.0      # a stem whose 95th-percentile envelope is below this isn't in the song
FAINT_DB: Final = 20.0        # a stem this far under the mix's 95th percentile may be bleed
ACTIVE_RANGE_DB: Final = 20.0  # a stem plays when within this of its own 95th percentile
ACTIVE_FLOOR_DB: Final = -50.0
MIN_RUN: Final = 2.0          # seconds; shorter runs of playing or silence are absorbed
MIN_VOCAL_RUN: Final = 3.0
BEAT_SNAP: Final = 0.12       # an event moves to a beat this close
STEP_DB, STEP_WINDOW, STEP_GAP = 6.0, 4.0, 6.0    # a stem's level step: medians of 4 s either side
BAND_DB, BAND_WINDOW, BAND_GAP = 9.0, 2.0, 4.0    # the band dropping out or coming back: 2 s medians
CHANGE_WINDOW: Final = 8.0    # seconds either side for a turn's mix and low-end change
TURN_GAP: Final = 1.25        # events this close to the first of a group are one turn
STOP_HOP: Final = 0.05
STOP_SUDDEN_DB, STOP_UNUSUAL_DB, STOP_DEEP_DB = 18.0, 6.0, 22.0
STOP_MIN, STOP_MAX = 0.25, 4.5
LOOP_LAGS: Final = (8, 16)    # beats back: two bars, or two bars at a doubled beat grid
LOOP_REPEATS: Final = 0.8     # a stem repeats its loop when beats correlate this well with the same beat back
LOOP_DROP: Final = 0.12       # a break: the next bar's median correlation falls this far under the last two bars'
LOOP_STEMS: Final = ("drums", "bass", "other")
VOICE_RATE, VOICE_HOP = 16000, 320            # pyin on the vocal stem: 20 ms frames
VOICE_MIN_FRAMES: Final = 25  # half a second of voiced frames before a section gets a pitch
VOICE_MIN_RANGE: Final = 1.5  # semitones; a narrower p10-p90 is bleed locked on by pyin, not singing
VOICE_MIN_MIDI: Final = 45.0  # A2; a lower median is bass bleed
VOICE_SHIFT_ST, VOICE_SHIFT_DB = 3.0, 4.0
BASS_RATE, BASS_HOP = 8000, 160
BASS_MIN_VOICED: Final = 0.3  # share of a bar's frames that must carry a pitch for it to get a root
NOTE_NAMES: Final = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


# Results

@dataclass(frozen=True)
class StemActivity:
    stem: str
    p95_db: float             # 95th percentile of the 0.75 s envelope, dBFS
    threshold_db: float       # playing above this
    present: bool
    faint: bool               # FAINT_DB or more under the mix: may be bleed from another stem
    runs: list[tuple[float, float]]  # seconds it plays, edges refined to 50 ms and snapped to beats


@dataclass(frozen=True)
class StemEvent:
    time: float
    part: str                 # a stem, or "band" for everything but the vocal
    change: Change
    db: float | None = None   # up/down: median level after minus before
    final: bool = False       # exits for the rest of the song
    voice: bool | None = None  # band only: whether the vocal stem plays across it
    loop: tuple[float, float] | None = None  # loop_break: correlation with two bars back, before and after


@dataclass(frozen=True)
class Turn:
    time: float
    events: list[StemEvent]
    mix_db: float | None      # median loudness of the CHANGE_WINDOW after minus before, whole mix
    low_db: float | None      # the same for drums plus bass


@dataclass(frozen=True)
class Stop:
    start: float
    hit: float                # where the level comes back
    depth_db: float           # under the 4 s before
    silent: bool              # below -45 dBFS throughout, not just near-silent
    hit_db: float             # the second after the hit against the 4 s before the stop


@dataclass(frozen=True)
class Bar:
    start: float
    end: float
    root: str | None          # pitch class of the bass stem, None when the bass doesn't sound a clear pitch


@dataclass(frozen=True)
class Harmony:
    phase: int                # which beat of four starts a bar, chosen so root changes land on bar starts
    bars: list[Bar]


@dataclass(frozen=True)
class VoiceSection:
    start: float
    end: float
    voiced_share: float
    loudness_db: float | None  # median RMS of the vocal stem over voiced frames
    pitch: str | None          # median, p10 and p90 of the sung pitch
    pitch_low: str | None
    pitch_high: str | None
    range_st: float | None
    median_midi: float | None
    reliable: bool             # pitch passes the guard against pyin locking onto bleed


@dataclass(frozen=True)
class VoiceShift:
    time: float
    semitones: float
    db: float
    before: str
    after: str


@dataclass(frozen=True)
class Texture:
    key: str | None
    key_strength: float | None
    instruments: list[tuple[str, float]]
    genre: list[tuple[str, float]]
    error: str | None = None   # why the tagger didn't run


# JSON round trip for the cache

T = TypeVar("T")


def from_json(cls: type[T], data: Any) -> T:
    """Rebuild a dataclass (nested dataclasses, lists, tuples, dicts, optionals) from its asdict() JSON."""
    return cast(T, _value(cls, data))


def _value(tp: Any, data: Any) -> Any:
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        hints = typing.get_type_hints(tp)
        return tp(**{f.name: _value(hints[f.name], data[f.name]) for f in dataclasses.fields(tp) if f.name in data})
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (Union, types.UnionType):
        if data is None:
            return None
        inner = [a for a in args if a is not type(None)]
        if len(inner) != 1:
            raise TypeError(f"unsupported union {tp}")
        return _value(inner[0], data)
    if origin is list:
        return [_value(args[0], x) for x in data]
    if origin is tuple:
        return tuple(_value(a, x) for a, x in zip(args, data, strict=True))
    if origin is dict:
        return {k: _value(args[1], x) for k, x in data.items()}
    if tp is float:
        return float(data)
    return data


# Levels

def block_db(y: Audio, rate: int, block: float = BLOCK) -> Arr:
    """RMS in dBFS per block, floored at FLOOR_DB."""
    n = max(1, round(rate * block))
    m = y.size // n
    if m == 0:
        return np.full(1, FLOOR_DB)
    rms = np.sqrt(np.mean(y[:m * n].astype(np.float64).reshape(m, n) ** 2, axis=1))
    return cast(Arr, np.maximum(20 * np.log10(np.maximum(rms, 1e-9)), FLOOR_DB))


def power_sum(levels: Sequence[Arr]) -> Arr:
    """dB levels added as power."""
    return cast(Arr, 10 * np.log10(np.sum([10 ** (x / 10) for x in levels], axis=0)))


def _runs(mask: NDArray[np.bool_]) -> list[tuple[int, int, bool]]:
    out: list[tuple[int, int, bool]] = []
    start = 0
    for i in range(1, mask.size + 1):
        if i == mask.size or mask[i] != mask[start]:
            out.append((start, i, bool(mask[start])))
            start = i
    return out


def debounce(mask: NDArray[np.bool_], min_blocks: int) -> NDArray[np.bool_]:
    """Absorb runs shorter than min_blocks into their neighbours, shortest first."""
    m = mask.copy()
    while True:
        rs = _runs(m)
        if len(rs) <= 1:
            return m
        a, b, v = min(rs, key=lambda r: r[1] - r[0])
        if b - a >= min_blocks:
            return m
        m[a:b] = not v


def snap(t: float, beats: Sequence[float], tol: float = BEAT_SNAP) -> float:
    if not beats:
        return t
    i = int(np.searchsorted(beats, t))
    best = min(beats[max(0, i - 1):i + 1], key=lambda b: abs(b - t))
    return float(best) if abs(best - t) <= tol else t


def sounding_end(level: Arr) -> int:
    """One past the last block that isn't trailing silence."""
    loud = np.nonzero(level > -50)[0]
    return int(loud[-1]) + 1 if loud.size else level.size


def sustained_steps(series: Arr, *, window: float, min_db: float, min_gap: float,
                    limit: int | None = None) -> list[tuple[int, float]]:
    """(block, dB) where the median of the next `window` differs from the previous one by at least min_db,
    the largest first, min_gap apart, each moved to the sharpest single-block change within a second.
    `limit` ends the after-window at the song's trailing silence, so the end isn't a step down."""
    w = round(window / BLOCK)
    n = min(series.size, limit) if limit else series.size
    d = np.zeros(n)
    for i in range(w, n - w):
        d[i] = np.median(series[i:i + w]) - np.median(series[i - w:i])
    gap = round(min_gap / BLOCK)
    peaks: list[tuple[int, float]] = []
    for i in np.argsort(-np.abs(d), kind="stable"):
        if abs(d[i]) < min_db:
            break
        if all(abs(int(i) - p) >= gap for p, _ in peaks):
            peaks.append((int(i), float(d[i])))
    refined = []
    for i, v in peaks:
        lo, hi = max(1, i - 4), min(n - 1, i + 4)
        if hi <= lo:
            refined.append((i, v))
            continue
        diffs = series[lo:hi] - series[lo - 1:hi - 1]
        refined.append((lo + int(np.argmax(diffs if v > 0 else -diffs)), v))
    return sorted(refined)


def refine_edge(fine: Arr, coarse: float, threshold: float, *, entering: bool) -> float:
    """Move a 0.25 s on/off time to the 50 ms block where the level crosses `threshold`. The 0.75 s envelope
    makes entries look up to half a second early and exits late: an entry is the first fine block above the
    threshold in [t - 0.5, t + 1.0], an exit one past the last in [t - 1.0, t + 0.5]."""
    lo, hi = (coarse - 0.5, coarse + 1.0) if entering else (coarse - 1.0, coarse + 0.5)
    a, b = max(0, int(lo / FINE)), min(fine.size, int(hi / FINE))
    above = np.nonzero(fine[a:b] > threshold)[0]
    if not above.size:
        return coarse
    return (a + int(above[0])) * FINE if entering else (a + int(above[-1]) + 1) * FINE


def refine_step(fine: Arr, coarse: float, up: bool) -> float:
    """Move a step to the sharpest 50 ms rise or fall of a 0.15 s-smoothed level within 0.75 s."""
    from scipy.ndimage import uniform_filter1d
    a, b = max(1, int((coarse - 0.75) / FINE)), min(fine.size - 1, int((coarse + 0.75) / FINE))
    if b - a < 3:
        return coarse
    smooth = uniform_filter1d(fine, size=3)
    diffs = smooth[a:b] - smooth[a - 1:b - 1]
    return (a + int(np.argmax(diffs if up else -diffs))) * FINE


# Stem activity and turns

@dataclass(frozen=True)
class Levels:
    """0.25 s series the turns are measured on."""
    mix: Arr
    stems: dict[str, Arr]     # 0.75 s envelopes
    low: Arr                  # drums plus bass
    band: Arr                 # drums, bass and other


def stem_levels(stems: Mapping[str, Audio], mix: Audio, rate: int) -> Levels:
    from scipy.ndimage import maximum_filter1d
    env = {n: cast(Arr, maximum_filter1d(block_db(stems[n], rate), size=ENVELOPE_BLOCKS)) for n in STEMS}
    size = min(x.size for x in env.values())
    env = {n: x[:size] for n, x in env.items()}
    return Levels(block_db(mix, rate)[:size], env, power_sum([env["drums"], env["bass"]]),
                  power_sum([env[n] for n in BAND]))


def activity(stems: Mapping[str, Audio], levels: Levels, rate: int,
             beats: Sequence[float]) -> tuple[list[StemActivity], list[StemEvent]]:
    """When each stem plays, and the events between: drums, bass and other entering and exiting, level steps
    of a playing stem, and the band dropping out or coming back. Vocal entries are phrases, not events, and a
    faint stem (likely bleed) gets runs but no events."""
    end = sounding_end(levels.mix)
    mix_ref = float(np.percentile(levels.mix, 95))
    fine = {n: block_db(stems[n], rate, FINE) for n in STEMS}
    acts: list[StemActivity] = []
    events: list[StemEvent] = []
    masks: dict[str, NDArray[np.bool_]] = {}
    for name in STEMS:
        env = levels.stems[name]
        ref = float(np.percentile(env, 95))
        present = ref > ABSENT_DB
        faint = ref < mix_ref - FAINT_DB
        threshold = max(ref - ACTIVE_RANGE_DB, ACTIVE_FLOOR_DB)
        min_run = round((MIN_VOCAL_RUN if name == "vocals" else MIN_RUN) / BLOCK)
        mask = debounce(env > threshold, min_run) if present else np.zeros(env.size, dtype=bool)
        masks[name] = mask
        runs: list[tuple[float, float]] = []
        stem_events: list[StemEvent] = []
        for a, b, playing in _runs(mask):
            if not playing:
                continue
            start = 0.0 if a == 0 else snap(refine_edge(fine[name], a * BLOCK, threshold, entering=True), beats)
            stop = (b * BLOCK if b == mask.size
                    else snap(refine_edge(fine[name], b * BLOCK, threshold, entering=False), beats))
            runs.append((round(start, 2), round(stop, 2)))
            if name == "vocals" or faint:
                continue
            if a > 0:
                stem_events.append(StemEvent(round(start, 2), name, "enters"))
            if b < mask.size:
                stem_events.append(StemEvent(round(stop, 2), name, "exits", final=not mask[b:].any()))
        acts.append(StemActivity(name, round(ref, 1), round(threshold, 1), present, faint, runs))
        if not present or faint:
            continue
        for i, d in sustained_steps(env, window=STEP_WINDOW, min_db=STEP_DB, min_gap=STEP_GAP, limit=end):
            w = round(STEP_WINDOW / BLOCK)
            if name == "vocals" and not (i >= w and mask[i - w:i + w].all()):
                continue  # a phrase starting after a gap is in the phrases; keep level changes in mid-singing
            if not (mask[max(0, i - 4)] or mask[min(mask.size - 1, i + 4)]):
                continue
            t = snap(refine_step(fine[name], i * BLOCK, d > 0), beats)
            if any(abs(e.time - t) < 2.0 for e in stem_events if e.change in ("enters", "exits")):
                continue
            stem_events.append(StemEvent(round(t, 2), name, "up" if d > 0 else "down", round(d, 1)))
        events += stem_events
    band_fine = block_db(sum((stems[n] for n in BAND), np.zeros_like(stems["drums"])), rate, FINE)
    for i, d in sustained_steps(levels.band, window=BAND_WINDOW, min_db=BAND_DB, min_gap=BAND_GAP, limit=end):
        t = snap(refine_step(band_fine, i * BLOCK, d > 0), beats)
        voice = bool(masks["vocals"][max(0, i - 2):i + 6].mean() > 0.5)
        events.append(StemEvent(round(t, 2), "band", "up" if d > 0 else "down", round(d, 1), voice=voice))
    return acts, sorted(events, key=lambda e: e.time)


def change_db(series: Arr, t: float, window: float = CHANGE_WINDOW) -> float | None:
    """Median of the window after `t` minus the window before, shortened at the ends; None under 2 s."""
    i, n = round(t / BLOCK), round(window / BLOCK)
    n = min(n, i, series.size - i)
    if n < round(2 / BLOCK):
        return None
    return round(float(np.median(series[i:i + n]) - np.median(series[i - n:i])), 1)


def _is_edge(e: StemEvent) -> bool:
    return e.change in ("enters", "exits") or e.part == "band"


def turns(events: Sequence[StemEvent], levels: Levels) -> list[Turn]:
    """Events grouped within TURN_GAP of a group's first, each group timed by its first entry, exit or band
    change, with the whole mix's and the low end's change over CHANGE_WINDOW either side."""
    groups: list[list[StemEvent]] = []
    for e in sorted(events, key=lambda e: e.time):
        if groups and e.time - groups[-1][0].time <= TURN_GAP:
            groups[-1].append(e)
        else:
            groups.append([e])
    out = []
    for g in groups:
        t = next((e.time for e in g if _is_edge(e)), g[0].time)
        out.append(Turn(t, g, change_db(levels.mix, t), change_db(levels.low, t)))
    return out


def drops(found: Sequence[Turn]) -> list[Turn]:
    """Entries (drums or bass coming in, or the band coming back) raising the low end DROP_MIN_DB or more, whose
    rise is within DROP_TIE_DB of the largest. The first is the drop; the others re-enter at a similar size."""
    entries = [t for t in found if t.low_db is not None and t.low_db >= DROP_MIN_DB and any(
        (e.part in ("drums", "bass") and e.change == "enters") or (e.part == "band" and e.change == "up")
        for e in t.events)]
    if not entries:
        return []
    top = max(t.low_db or 0.0 for t in entries)
    return [t for t in entries if (t.low_db or 0.0) >= top - DROP_TIE_DB]


DROP_TIE_DB: Final = 6.0
DROP_MIN_DB: Final = 6.0  # a bass fading in under drums that already play isn't a drop


def stops(mix: Audio, rate: int, beats: Sequence[float]) -> list[Stop]:
    """Near-silences of 0.25-4.5 s that are unusual for their context, then a return to level. The dip is at
    least 18 dB under the median of the 4 s before and 6 dB under the quietest 5% of the 8 s before, so the
    routine gaps of sparse playing don't count; or 22 dB under the song's median, a fade to near-silence out of
    a passage at normal level."""
    lv = block_db(mix, rate, STOP_HOP)
    ref_n, floor_n = round(4 / STOP_HOP), round(8 / STOP_HOP)
    song_ref = float(np.median(lv[:sounding_end(lv)]))
    out: list[Stop] = []
    i = floor_n
    while i < lv.size:
        ref = float(np.median(lv[i - ref_n:i]))
        floor = float(np.percentile(lv[i - floor_n:i], 5))
        sudden = lv[i] < ref - STOP_SUDDEN_DB and lv[i] < floor - STOP_UNUSUAL_DB
        deep = lv[i] < song_ref - STOP_DEEP_DB and ref > song_ref - 10
        if not (sudden or deep):
            i += 1
            continue
        j = i
        while j < lv.size and (lv[j] < ref - 12 or (deep and lv[j] < song_ref - 16)):
            j += 1
        length = (j - i) * STOP_HOP
        depth = ref - float(np.median(lv[i:j]))
        after = float(np.median(lv[j:j + round(1 / STOP_HOP)])) if j < lv.size else FLOOR_DB
        if STOP_MIN <= length <= STOP_MAX and after >= ref - 6 and depth >= 12:
            out.append(Stop(round(i * STOP_HOP, 2), round(snap(j * STOP_HOP, beats, 0.15), 2), round(depth, 1),
                            bool(np.max(lv[i:j]) < -45), round(after - ref, 1)))
        i = j + 1
    return out


# Loops

def _beat_patterns(y: Audio, rate: int, beats: Sequence[float]) -> tuple[Arr, NDArray[np.bool_]]:
    """One row per beat: the log-mel spectrogram of the beat in four equal slices, and whether it has sound."""
    import librosa
    hop = 512
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=rate, n_fft=2048, hop_length=hop, n_mels=48,
                                                             fmax=8000), ref=1.0, top_db=None)
    mel = np.maximum(mel, -100.0)
    frames = mel.shape[1]
    rows, sound = [], []
    for a, b in zip(beats, beats[1:]):
        edges = np.linspace(a, b, 5) * rate / hop
        cells = []
        for k in range(4):
            lo, hi = int(edges[k]), max(int(edges[k + 1]), int(edges[k]) + 1)
            cells.append(mel[:, min(lo, frames - 1):min(hi, frames)].mean(axis=1))
        v = np.concatenate(cells)
        rows.append(v)
        sound.append(bool(v.max() > -60))
    return np.array(rows, dtype=np.float64).reshape(len(rows), -1), np.array(sound, dtype=bool)


def loop_similarity(y: Audio, rate: int, beats: Sequence[float]) -> Arr:
    """Per beat, the correlation of its pattern with the same beat LOOP_LAGS beats back (the better of the
    two); NaN where either beat is silent or there is nothing that far back."""
    rows, sound = _beat_patterns(y, rate, beats)
    sim = np.full(rows.shape[0], np.nan)
    centred = rows - rows.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centred, axis=1) + 1e-9
    for i in range(rows.shape[0]):
        best = np.nan
        for lag in LOOP_LAGS:
            j = i - lag
            if j >= 0 and sound[i] and sound[j]:
                c = float(centred[i] @ centred[j] / (norm[i] * norm[j]))
                best = c if np.isnan(best) else max(best, c)
        sim[i] = best
    return sim


def loop_breaks(stems: Mapping[str, Audio], rate: int, beats: Sequence[float],
                acts: Sequence[StemActivity]) -> list[StemEvent]:
    """Beats where a stem that was repeating itself (the two bars before correlate at LOOP_REPEATS or better
    with two bars earlier) stops matching: the bar after drops LOOP_DROP or more. One per stem per two bars,
    only inside a run where the stem plays from the comparison's start to its end."""
    playing = {a.stem: a for a in acts}
    out: list[StemEvent] = []
    for name in LOOP_STEMS:
        act = playing[name]
        if not act.present or act.faint or len(beats) < 3 * LOOP_LAGS[0]:
            continue
        sim = loop_similarity(stems[name], rate, beats)

        def window(i: int) -> tuple[float, float] | None:
            """Median correlation over the two bars before beat i and the bar from it."""
            before, after = sim[i - 8:i], sim[i:i + 4]
            if np.isnan(before).sum() > 2 or np.isnan(after).sum() > 1:
                return None
            return float(np.nanmedian(before)), float(np.nanmedian(after))

        hits = []
        for i in range(LOOP_LAGS[0], sim.size - 4):
            w = window(i)
            if (w is not None and w[0] >= LOOP_REPEATS and w[0] - w[1] >= LOOP_DROP
                    and _within(beats[max(0, i - 8 - LOOP_LAGS[-1])], beats[i + 4], act.runs)):
                hits.append(i)
        kept: list[tuple[int, float, float]] = []
        for group in _groups(hits, 8):
            # The break is the first beat in the group whose own correlation falls under the two bars before it.
            k = next((i for i in group if (w := window(i)) and sim[i] < w[0] - LOOP_DROP), group[0])
            b, a = window(k) or (math.nan, math.nan)
            kept.append((k, b, a))
        out += [StemEvent(round(beats[i], 2), name, "loop_break", loop=(round(b, 2), round(a, 2)))
                for i, b, a in kept]
    return sorted(out, key=lambda x: x.time)


def _groups(indices: Sequence[int], gap: int) -> list[list[int]]:
    """Sorted indices split where consecutive ones are `gap` or more apart."""
    out: list[list[int]] = []
    for i in indices:
        if out and i - out[-1][-1] < gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _within(a: float, b: float, runs: Sequence[tuple[float, float]]) -> bool:
    """One run covers all of [a, b]."""
    return any(start <= a and b <= end for start, end in runs)


# Pitch

def _pyin(y: Audio, *, fmin: float, fmax: float, rate: int, hop: int) -> tuple[Arr, NDArray[np.bool_]]:
    """librosa's pyin with frame_length 1024. On some inputs numba warns about a cast inside librosa's own
    autocorrelation of finite audio; the result is unaffected, so that one warning is silenced."""
    import librosa
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
        f0, voiced, _ = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=rate, frame_length=1024, hop_length=hop)
    return np.asarray(f0, dtype=np.float64), np.asarray(voiced, dtype=bool)


def note_name(midi: float) -> str:
    m = int(round(midi))
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"


def bass_roots(bass: Audio, rate: int, beats: Sequence[float], runs: Sequence[tuple[float, float]]) -> Harmony:
    """The bass stem's pitch class per bar of four beats, counting only frames inside the stem's runs (outside
    them it holds bleed). Which beat starts a bar is the phase that puts the most root changes on bar starts;
    the beat tracker gives no downbeats."""
    import librosa
    if not runs or len(beats) < 5:
        return Harmony(0, [])
    y = librosa.resample(bass, orig_sr=rate, target_sr=BASS_RATE)
    f0, voiced = _pyin(y, fmin=30, fmax=400, rate=BASS_RATE, hop=BASS_HOP)
    times = np.arange(f0.size) * BASS_HOP / BASS_RATE
    playing = np.zeros(f0.size, dtype=bool)
    for a, b in runs:
        playing |= (times >= a) & (times < b)
    pc = np.where(voiced & np.isfinite(f0) & playing,
                  np.round(librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))) % 12, -1)

    def root(a: float, b: float) -> int | None:
        cell = pc[(times >= a) & (times < b)]
        sounded = cell[cell >= 0].astype(int)
        if cell.size == 0 or sounded.size < BASS_MIN_VOICED * cell.size:
            return None
        return int(np.argmax(np.bincount(sounded, minlength=12)))

    per_beat = [root(a, b) for a, b in zip(beats, beats[1:])]
    changes = [i for i in range(1, len(per_beat))
               if per_beat[i] is not None and per_beat[i - 1] is not None and per_beat[i] != per_beat[i - 1]]
    phase = max(range(4), key=lambda p: (sum(i % 4 == p for i in changes), -p))
    starts = list(beats[phase::4])
    if starts[0] > beats[0]:
        starts.insert(0, beats[0])
    bars = []
    for a, b in zip(starts, [*starts[1:], beats[-1]]):
        if b - a <= 0:
            continue
        r = root(a, b)
        bars.append(Bar(round(a, 2), round(b, 2), NOTE_NAMES[r] if r is not None else None))
    return Harmony(phase, bars)


def voice_sections(vocals: Audio, rate: int, bounds: Sequence[float],
                   runs: Sequence[tuple[float, float]]) -> list[VoiceSection]:
    """Loudness and sung pitch of the vocal stem per section, from pyin at 16 kHz, over frames inside the vocal
    stem's runs where pyin hears a pitch and the stem is above ACTIVE_FLOOR_DB. Outside its runs the vocal stem
    holds bleed, which pyin tracks as readily as singing."""
    import librosa
    y = librosa.resample(vocals, orig_sr=rate, target_sr=VOICE_RATE)
    f0, voiced = _pyin(y, fmin=70, fmax=1100, rate=VOICE_RATE, hop=VOICE_HOP)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=VOICE_HOP)[0]
    n = min(f0.size, rms.size)
    loud = np.maximum(20 * np.log10(np.maximum(rms[:n].astype(np.float64), 1e-9)), FLOOR_DB)
    # A pitch counts only where the stem is above ACTIVE_FLOOR_DB: pyin also tracks bleed at -60 dBFS.
    t = np.arange(n) * VOICE_HOP / VOICE_RATE
    singing = np.zeros(n, dtype=bool)
    for a, b in runs:
        singing |= (t >= a) & (t < b)
    sung = voiced[:n] & np.isfinite(f0[:n]) & (loud > ACTIVE_FLOOR_DB) & singing
    midi = np.where(sung, librosa.hz_to_midi(np.nan_to_num(f0[:n], nan=1.0)), np.nan)
    out = []
    for a, b in zip(bounds, bounds[1:]):
        inside = (t >= a) & (t < b)
        v = inside & np.isfinite(midi)
        share = round(float(v.sum() / max(int(inside.sum()), 1)), 2)
        if v.sum() < VOICE_MIN_FRAMES:
            out.append(VoiceSection(a, b, share, None, None, None, None, None, None, False))
            continue
        p10, p50, p90 = (float(x) for x in np.percentile(midi[v], [10, 50, 90]))
        out.append(VoiceSection(a, b, share, round(float(np.median(loud[v])), 1), note_name(p50), note_name(p10),
                                note_name(p90), round(p90 - p10, 1), round(p50, 1),
                                p90 - p10 >= VOICE_MIN_RANGE and p50 >= VOICE_MIN_MIDI))
    return out


def voice_shifts(sections: Sequence[VoiceSection]) -> list[VoiceShift]:
    """Between consecutive sections with a reliable pitch: the voice moving VOICE_SHIFT_ST semitones or
    VOICE_SHIFT_DB dB or more."""
    usable = [s for s in sections if s.reliable and s.median_midi is not None and s.loudness_db is not None]
    out = []
    for prev, cur in zip(usable, usable[1:]):
        assert prev.median_midi is not None and cur.median_midi is not None
        assert prev.loudness_db is not None and cur.loudness_db is not None
        st, db = cur.median_midi - prev.median_midi, cur.loudness_db - prev.loudness_db
        if abs(st) >= VOICE_SHIFT_ST or abs(db) >= VOICE_SHIFT_DB:
            out.append(VoiceShift(cur.start, round(st, 1), round(db, 1), prev.pitch or "?", cur.pitch or "?"))
    return out


# Tempo

TEMPO_TAPPED: Final = (75.0, 140.0)  # BPM; outside this, listeners usually tap half or double the tracker's beat


def tempo_alternative(tempo: float) -> float | None:
    """Half or double the tracked tempo when the tracked one is outside TEMPO_TAPPED. Beat trackers often lock
    onto twice or half the beat a listener taps (the trial found 184.6 for a 92 BPM song and 58.7 for a 122
    one), and the onset signal alone can't tell which, so this is a flag to check by ear, not a correction."""
    if tempo > TEMPO_TAPPED[1]:
        return round(tempo / 2, 1)
    if 0 < tempo < TEMPO_TAPPED[0]:
        return round(tempo * 2, 1)
    return None


# Texture: key, instruments, genre (Essentia)

MODEL_URL: Final = "https://essentia.upf.edu/models"
ESSENTIA_MODELS: Final = {
    "discogs-effnet-bs64-1.pb": ("feature-extractors/discogs-effnet",
                                 "3ed9af50d5367c0b9c795b294b00e7599e4943244f4cbd376869f3bfc87721b1"),
    "mtg_jamendo_instrument-discogs-effnet-1.pb": ("classification-heads/mtg_jamendo_instrument",
                                                   "2e8c3003c722e098da371b6a1f7ad0ce62fac0dcfc09c7c7997d430941196c2a"),
    "mtg_jamendo_instrument-discogs-effnet-1.json": ("classification-heads/mtg_jamendo_instrument",
                                                     "7d02204c6451b5615e2968ec6364bbae3b915c886e608f05f00d3a38dc5177c4"),
    "genre_discogs400-discogs-effnet-1.pb": ("classification-heads/genre_discogs400",
                                             "3885ba078a35249af94b8e5e4247689afac40deca4401a4bc888daf5a579c01c"),
    "genre_discogs400-discogs-effnet-1.json": ("classification-heads/genre_discogs400",
                                               "2d367319d9b782ffa10f69abf0e805b3ac4e10899025e5bdbaceda3919b243e0"),
}
ESSENTIA_DIR: Final = Path.home() / ".cache" / "animeedits" / "essentia"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def essentia_model(name: str, model_dir: Path = ESSENTIA_DIR) -> Path:
    """A tagger model file, downloaded once and checked against its pinned SHA-256."""
    folder, digest = ESSENTIA_MODELS[name]
    path = model_dir / name
    if path.is_file() and _sha256(path) == digest:
        return path
    model_dir.mkdir(parents=True, exist_ok=True)
    url = f"{MODEL_URL}/{folder}/{name}"
    part = model_dir / f"{name}.part"
    error: OSError | None = None
    for _ in range(DOWNLOAD_TRIES):  # the server sometimes stalls mid-file
        try:
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as f:
                shutil.copyfileobj(response, f)
            break
        except OSError as e:
            error = e
    else:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"couldn't download {url}: {error}")
    if _sha256(part) != digest:
        part.unlink()
        raise RuntimeError(f"{url} doesn't match its pinned checksum")
    part.replace(path)
    return path


DOWNLOAD_TRIES: Final = 3


def texture(mix44: Audio, mix16: Audio, model_dir: Path = ESSENTIA_DIR) -> Texture:
    """Song key with its strength (Essentia's EDMA profile), the top two instruments (MTG-Jamendo) and the top
    genres (Discogs-400) of the whole mix. The instrument tags say what the mix sounds like, not whether a
    stem plays: the stems answer that."""
    import json
    try:
        import essentia
        import essentia.standard as es
    except ImportError as e:
        return Texture(None, None, [], [], f"essentia-tensorflow is not installed ({e})")
    essentia.log.warningActive = False
    key, scale, strength = es.KeyExtractor(profileType="edma", sampleRate=44100)(mix44)
    try:
        effnet = es.TensorflowPredictEffnetDiscogs(graphFilename=str(essentia_model("discogs-effnet-bs64-1.pb",
                                                                                     model_dir)),
                                                   output="PartitionedCall:1")
        emb = effnet(mix16)
        inst_model = essentia_model("mtg_jamendo_instrument-discogs-effnet-1.pb", model_dir)
        genre_model = essentia_model("genre_discogs400-discogs-effnet-1.pb", model_dir)
        inst = es.TensorflowPredict2D(graphFilename=str(inst_model), output="model/Sigmoid")(emb).mean(axis=0)
        genre = es.TensorflowPredict2D(graphFilename=str(genre_model), input="serving_default_model_Placeholder",
                                       output="PartitionedCall:0")(emb).mean(axis=0)
        inst_names = json.loads(essentia_model("mtg_jamendo_instrument-discogs-effnet-1.json",
                                               model_dir).read_text())["classes"]
        genre_names = json.loads(essentia_model("genre_discogs400-discogs-effnet-1.json",
                                                model_dir).read_text())["classes"]
    except (OSError, RuntimeError, ValueError, KeyError) as e:
        return Texture(f"{key} {scale}", round(float(strength), 2), [], [], f"tagger unavailable: {e}")
    top_inst = [(str(inst_names[i]), round(float(inst[i]), 2)) for i in np.argsort(-inst)[:2]]
    top_genre = [(str(genre_names[i]), round(float(genre[i]), 2)) for i in np.argsort(-genre)[:2]]
    return Texture(f"{key} {scale}", round(float(strength), 2), top_inst, top_genre)


def genre_name(label: str) -> str:
    """Discogs labels are 'Parent---Style'."""
    parent, _, style = label.partition("---")
    return f"{style} ({parent.lower()})" if style else parent



TIME_FIELDS: Final = frozenset({"time", "start", "end", "hit", "novelty_time"})


def shift_times(obj: T, dt: float) -> T:
    """`obj` with `dt` added to every time in it: fields named in TIME_FIELDS, and the edges of stem runs.
    Measurements are made on the audio's clock; a video's audio can start after its first frame."""
    if dt == 0:
        return obj
    if isinstance(obj, list):
        return cast(T, [shift_times(x, dt) for x in obj])
    if not (dataclasses.is_dataclass(obj) and not isinstance(obj, type)):
        return obj
    changes: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if f.name in TIME_FIELDS and isinstance(value, float):
            changes[f.name] = round(value + dt, 3)
        elif f.name == "runs":
            changes[f.name] = [(round(a + dt, 3), round(b + dt, 3)) for a, b in value]
        elif isinstance(value, list) or dataclasses.is_dataclass(value):
            changes[f.name] = shift_times(value, dt)
    return cast(T, dataclasses.replace(obj, **changes))
