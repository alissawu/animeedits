"""Measure a reference edit's timeline, or map a song: analyze_edit.py MEDIA OUT_DIR.

For video, finds hard cuts, white flashes, dips to black and cross-dissolves frame by
frame. For anything with sound, maps the audio with librosa (tempo, beats, strong onsets,
loudness, drops, rough sections) and vocal word timings with faster-whisper. Writes
report.md and timeline.json, plus frames.json and contact sheets for video. Audio-only
input (mp3, wav, m4a, flac and so on) gets the audio report only. Word text is never
stored, only timings. See docs/edit_analysis.md.
"""
import argparse
import bisect
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

Label = Literal["content", "white", "black"]
SegmentKind = Literal["shot", "flash", "black"]

# Bump a stage's version when its output changes, so cached results are recomputed.
PICTURE_VERSION, AUDIO_VERSION, WORDS_VERSION = 2, 2, 1
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
ANALYSIS_SIDE = 128           # long side of the frames used for detection
WHITE_LUMA = 140.0            # a flash frame averages at least this bright
WHITE_SHARE = 0.25            # and has at least this share of pixels above 200
BLACK_P98 = 30.0              # a black frame keeps 98% of its pixels below this
BLACK_LUMA = 10.0             # and averages below this
MAX_FLASH = 0.5               # seconds; a longer white run is bright footage
SPIKE_MARGIN = 40.0           # a too-short piece this much brighter than both neighbours is a flash
CUT_THRESHOLD = 0.45          # adjacent frames cut when structure and edge distance both exceed this
SAME_SHOT = 0.35              # across a flash or black, both distances below this mean the same shot
PIXEL_GATE = 3.0              # mean RGB change below which nothing is a cut
SPIKE_RATIO = 3.0             # a cut the distances miss changes this many times more than any frame near it
SPIKE_REACH = 0.3             # seconds either side that count as near
SPIKE_HIST = 0.15             # and shifts the colour histogram at least this much
SPIKE_EDGE = 0.55             # and changes its edges this much, more than CUT_THRESHOLD asks
DISSOLVE_HALF_WIDTHS = (0.1, 0.2, 0.3, 0.4)  # seconds either side of a dissolve's centre
DISSOLVE_FIT = 0.35           # blend residual as a share of the distance to the nearer end
SAMPLE_RATE = 22050
WINDOW = 0.25                 # loudness window, seconds
LOW_BAND = 150.0              # Hz
ONSET_PERCENTILE = 90.0
DROP_SKIP = 0.5               # seconds at the start where the audio's own onset is not a drop
SILENCE_POWER = 1e-6          # -60 dBFS; a drop needs sound before it
DROP_MIN_DB = 6.0             # besides the biggest jump, a drop needs at least this
DROP_SHARE = 0.5              # and at least this share of the biggest jump
DROP_GAP = 4.0                # seconds between separate drops
NOVELTY_BEATS = (4, 16)       # novelty kernel half-width in beats, scaled with the song's length
NOVELTY_FLOOR = 0.25          # weakest section boundary, relative to the strongest
MIN_SECTION = 1.0             # seconds; shortest section from audio boundaries
LONG_AUDIO = 60.0             # seconds; longer audio gets a coarser loudness table
PHRASE_GAP = 1.0              # seconds without words that end a vocal phrase
SNAP = 0.5                    # snap section bounds to a shot boundary this close
SYNC = 0.1                    # "on" an event when within this many seconds
REACH = 1.0                   # the report leaves out events further than this from a row
SHEET_WIDTH = 1280            # widest frame decoded for contact sheets
SHEET_HEIGHT = 1600           # tallest caption sheet
THUMB_WIDTH, THUMB_COLUMNS, THUMBS_PER_SHEET = 384, 4, 24


@dataclass(frozen=True)
class VideoStream:
    width: int
    height: int
    rate: str                 # exact frame rate for ffmpeg, e.g. "30000/1001"
    fps: float


@dataclass(frozen=True)
class MediaInfo:
    path: str
    duration: float
    video: VideoStream | None # None for audio-only input
    has_audio: bool
    audio_offset: float       # audio start minus video start, seconds; 0 without video


@dataclass(frozen=True)
class FrameStats:
    luma: float               # mean luma, 0-255
    white: float              # share of pixels with luma above 200
    contrast: float           # luma standard deviation
    hist_dist: float          # RGB histogram distance to the previous frame, 0-1
    pixel_diff: float         # mean absolute RGB change from the previous frame
    structure_dist: float     # 1 - correlation of brightness-normalised luma with the previous frame
    edge_dist: float          # the same for gradient magnitude
    label: Label


@dataclass(frozen=True)
class Segment:
    kind: SegmentKind
    start: int                # first frame
    end: int                  # one past the last frame
    shot: int | None = None   # shot number, shots only
    enter: str | None = None  # shots only: "start", "cut", "dissolve", or the flash/black before it
    keyframe: int | None = None


@dataclass(frozen=True)
class InShotEvent:
    kind: Literal["flash", "black"]
    start: int
    end: int
    shot: int


@dataclass(frozen=True)
class Dissolve:
    start: int                # first blended frame; the incoming shot starts here
    end: int                  # one past the last blended frame
    shot: int                 # incoming shot


@dataclass(frozen=True)
class Picture:
    frames: list[FrameStats]
    segments: list[Segment]   # shots, flashes and black, in order, covering every frame
    in_shot: list[InShotEvent]
    dissolves: list[Dissolve]


@dataclass(frozen=True)
class Drop:
    time: float
    jump_db: float            # loudness of the WINDOW after minus the WINDOW before


@dataclass(frozen=True)
class Boundary:
    time: float               # a beat where timbre and loudness change
    strength: float           # novelty peak relative to the strongest, 0-1


@dataclass(frozen=True)
class AudioMap:
    duration: float
    offset: float             # time of the first audio sample on the video clock
    tempo: float
    beats: list[float]
    onsets: list[float]       # onsets at or above the ONSET_PERCENTILE of onset strength
    rms_db: list[float]       # dBFS per WINDOW, starting at offset
    low_db: list[float]       # dBFS below LOW_BAND per WINDOW
    drops: list[Drop]         # in time order
    boundaries: list[Boundary]  # rough section boundaries, in time order

    @property
    def drop(self) -> Drop | None:
        """The biggest loudness jump."""
        return max(self.drops, key=lambda d: d.jump_db, default=None)


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    probability: float


@dataclass(frozen=True)
class Phrase:
    start: float
    end: float
    words: int


@dataclass(frozen=True)
class Section:
    start: float
    end: float
    phrase: int | None        # 1-based vocal phrase covering most of the section
    vocal_share: float        # share of the section inside vocal phrases
    drop: Drop | None         # the drop the section starts on


@dataclass(frozen=True)
class Nearest:
    time: float
    delta: float              # cut time minus event time
    edge: str | None = None   # "start" or "end" for word boundaries


@dataclass(frozen=True)
class SheetIndex:
    shots: list[tuple[str, int, int]]            # file, first shot, last shot
    captions: list[tuple[str, float, float]]     # file, first time, last time


@dataclass(frozen=True)
class Settings:
    cut_threshold: float
    min_shot: int
    caption_band: tuple[float, float]
    caption_step: float
    whisper_model: str | None
    lang: str | None


@dataclass(frozen=True)
class Analysis:
    info: MediaInfo
    picture: Picture | None
    audio: AudioMap | None
    words: list[Word]
    phrases: list[Phrase]
    sections: list[Section]
    sheets: SheetIndex | None
    settings: Settings

    @property
    def fps(self) -> float:
        return _video(self.info).fps

    def time(self, frame: int) -> float:
        return round(frame / self.fps, 3)

    @property
    def shots(self) -> list[Segment]:
        return [s for s in self.picture.segments if s.kind == "shot"] if self.picture else []

    def sync(self, t: float) -> dict[str, Nearest | None]:
        audio = self.audio
        return {"beat": nearest(t, audio.beats) if audio else None,
                "onset": nearest(t, audio.onsets) if audio else None,
                "word": nearest_word_edge(t, self.words)}


@dataclass(frozen=True)
class _Features:
    luma: NDArray[np.float32]
    white: NDArray[np.float32]
    p98: NDArray[np.float32]
    contrast: NDArray[np.float32]
    structure: NDArray[np.float32]   # standardised half-size luma, one row per frame
    edges: NDArray[np.float32]       # standardised gradient magnitude of that luma
    hist_dist: NDArray[np.float32]
    pixel_diff: NDArray[np.float32]


@dataclass
class _Item:
    kind: Literal["piece", "flash", "black"]
    start: int
    end: int
    group: int = -1


# Cache and arguments

def save(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def cached(path: Path, signature: dict[str, Any]) -> Any | None:
    if path.exists():
        try:
            obj = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and obj.get("signature") == signature:
            return obj["data"]
    return None


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def unit_float(value: str) -> float:
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def caption_band(value: str) -> tuple[float, float]:
    try:
        top, bottom = (float(part) for part in value.split(":"))
    except ValueError:
        raise argparse.ArgumentTypeError("use TOP:BOTTOM, e.g. 0.75:1") from None
    if not 0 <= top < bottom <= 1:
        raise argparse.ArgumentTypeError("need 0 <= TOP < BOTTOM <= 1")
    return top, bottom


# Media input

def _run(cmd: list[str]) -> bytes:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout


def _rate(text: str | None) -> Fraction | None:
    if not text or "/" not in text:
        return None
    num, den = (int(part) for part in text.split("/"))
    return Fraction(num, den) if num > 0 and den > 0 else None


def probe(media: Path) -> MediaInfo:
    meta = json.loads(_run([
        "ffprobe", "-v", "error", "-of", "json", "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,start_time"
        ":stream_disposition=attached_pic:format=duration", str(media)]))
    streams = meta.get("streams", [])
    # Cover art in an mp3 or m4a is an attached picture, not video.
    vstream = next((s for s in streams if s.get("codec_type") == "video"
                    and not s.get("disposition", {}).get("attached_pic")), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if vstream is None and astream is None:
        raise ValueError(f"no video or audio stream in {media}")
    video = None
    offset = 0.0
    if vstream is not None:
        rate = _rate(vstream.get("avg_frame_rate")) or _rate(vstream.get("r_frame_rate"))
        if rate is None:
            raise ValueError(f"unknown frame rate in {media}")
        video = VideoStream(int(vstream["width"]), int(vstream["height"]),
                            f"{rate.numerator}/{rate.denominator}", float(rate))
        if astream is not None:
            offset = float(astream.get("start_time") or 0) - float(vstream.get("start_time") or 0)
    return MediaInfo(path=str(media), duration=float(meta.get("format", {}).get("duration") or 0),
                     video=video, has_audio=astream is not None, audio_offset=round(offset, 4))


def _video(info: MediaInfo) -> VideoStream:
    if info.video is None:
        raise ValueError(f"no video stream in {info.path}")
    return info.video


def _even(x: float) -> int:
    return max(2, 2 * round(x / 2))


def analysis_size(video: VideoStream) -> tuple[int, int]:
    scale = ANALYSIS_SIDE / max(video.width, video.height)
    return _even(video.width * scale), _even(video.height * scale)


def _ffmpeg_frames(info: MediaInfo, width: int, height: int, flags: str) -> list[str]:
    # The fps filter maps every frame onto a constant grid, so frame i is at i / fps.
    return ["ffmpeg", "-v", "error", "-nostdin", "-i", info.path, "-map", "0:v:0",
            "-vf", f"fps={_video(info).rate},scale={width}:{height}:flags={flags}",
            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]


def decode_frames(info: MediaInfo) -> NDArray[np.uint8]:
    """Decode every frame at analysis size."""
    width, height = analysis_size(_video(info))
    raw = _run(_ffmpeg_frames(info, width, height, "area"))
    size = width * height * 3
    if len(raw) < size or len(raw) % size:
        raise RuntimeError(f"ffmpeg returned {len(raw)} bytes, not whole {width}x{height} frames")
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


def iter_frames(info: MediaInfo, width: int, height: int) -> Iterator[NDArray[np.uint8]]:
    """Stream frames at the given size without holding the video in memory."""
    size = width * height * 3
    proc = subprocess.Popen(_ffmpeg_frames(info, width, height, "bicubic"),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    finished = False
    try:
        while len(buf := proc.stdout.read(size)) == size:
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        finished = True
    finally:
        if not finished:
            proc.kill()
        err = proc.stderr.read().decode(errors="replace").strip()
        code = proc.wait()
        proc.stdout.close()
        proc.stderr.close()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed: {err}")


# Picture analysis

def _standardize(x: NDArray[np.float32]) -> NDArray[np.float32]:
    flat = x.ravel() - x.mean()
    std = float(flat.std())
    return (flat / std).astype(np.float32) if std > 1e-3 else np.zeros_like(flat)


def _features(frames: NDArray[np.uint8]) -> _Features:
    n, h, w, _ = frames.shape
    hh, hw = h // 2, w // 2
    luma, white, p98, contrast, pixel_diff = (np.zeros(n, np.float32) for _ in range(5))
    structure = np.zeros((n, hh * hw), np.float32)
    edges = np.zeros((n, (hh - 1) * (hw - 1)), np.float32)
    hists = np.zeros((n, 512), np.float32)
    prev: NDArray[np.float32] | None = None
    for i, frame in enumerate(frames):
        rgb = frame.astype(np.float32)
        y = rgb @ LUMA
        luma[i], white[i], contrast[i] = y.mean(), (y > 200).mean(), y.std()
        p98[i] = np.percentile(y, 98)
        half = y[:hh * 2, :hw * 2].reshape(hh, 2, hw, 2).mean((1, 3))
        structure[i] = _standardize(half)
        edges[i] = _standardize(np.hypot(np.diff(half, axis=1)[:-1], np.diff(half, axis=0)[:, :-1]))
        bins = (frame >> 5).astype(np.int32)
        hists[i] = np.bincount((bins[..., 0] * 64 + bins[..., 1] * 8 + bins[..., 2]).ravel(),
                               minlength=512) / (h * w)
        if prev is not None:
            pixel_diff[i] = np.abs(rgb - prev).mean()
        prev = rgb
    hist_dist = np.zeros(n, np.float32)
    hist_dist[1:] = 0.5 * np.abs(np.diff(hists, axis=0)).sum(1)
    return _Features(luma, white, p98, contrast, structure, edges, hist_dist, pixel_diff)


def _distance(f: _Features, a: int, b: int) -> tuple[float, float]:
    """Structure and edge distance between two frames, each 0 (same) to 1 (unrelated)."""
    s = 1.0 - float(f.structure[a] @ f.structure[b]) / f.structure.shape[1]
    e = 1.0 - float(f.edges[a] @ f.edges[b]) / f.edges.shape[1]
    return min(max(s, 0.0), 1.0), min(max(e, 0.0), 1.0)


def _window_distance(f: _Features, before: Sequence[int], after: Sequence[int]) -> tuple[float, float]:
    """Smallest distances between two frame groups, so one flicker frame cannot fake a cut."""
    pairs = [_distance(f, a, b) for a in before for b in after]
    return min(p[0] for p in pairs), min(p[1] for p in pairs)


def _runs(mask: Sequence[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate(mask):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _labels(f: _Features, fps: float) -> list[Label]:
    labels: list[Label] = []
    for luma, white, p98 in zip(f.luma, f.white, f.p98):
        if p98 < BLACK_P98 and luma < BLACK_LUMA:
            labels.append("black")
        elif luma >= WHITE_LUMA and white >= WHITE_SHARE:
            labels.append("white")
        else:
            labels.append("content")
    longest = max(1, round(MAX_FLASH * fps))
    for s, e in _runs([label == "white" for label in labels]):
        if e - s > longest:
            labels[s:e] = ["content"] * (e - s)
    return labels


def _transition_spans(labels: Sequence[Label], min_shot: int) -> list[tuple[int, int]]:
    """Runs of flash or black frames, including strobes of footage too short to be a shot."""
    spans: list[tuple[int, int]] = []
    n, i = len(labels), 0
    while i < n:
        if labels[i] == "content":
            i += 1
            continue
        j = i
        while True:
            while j < n and labels[j] != "content":
                j += 1
            k = j
            while k < n and labels[k] == "content":
                k += 1
            if j < n and k < n and k - j < min_shot:
                j = k
                continue
            break
        spans.append((i, j))
        i = j
    return spans


def _span_items(labels: Sequence[Label], start: int, end: int) -> list[_Item]:
    """Split a span into flash and black parts; strobe frames take the kind before them."""
    items: list[_Item] = []
    kind: Literal["flash", "black"] = "black"
    for i in range(start, end):
        if labels[i] != "content":
            kind = "flash" if labels[i] == "white" else "black"
        if items and items[-1].kind == kind:
            items[-1].end = i + 1
        else:
            items.append(_Item(kind, i, i + 1))
    return items


def _spike_cuts(f: _Features, runs: Sequence[tuple[int, int]], reach: int) -> list[int]:
    """Cuts between shots alike in layout and palette, which the distance test misses.

    Each run is a stretch of adjacent footage frames with no cut found yet. Inside it, a frame
    is a cut when its pixel change stands out from every other change within `reach` frames
    in the same run, its edges change, and its colours shift. Held drawings, animation on 2s
    and 3s, fast action and shake change on many nearby frames, so none of them stands out.
    A drawing moved or zoomed by the editor stands out but keeps its colours; a grade stepped
    over a held picture keeps most of its edges, so the edge bar is higher than the distance
    test's. A frame needs `reach` other frames of the run near it, or there is nothing to stand
    out from.
    """
    cuts: list[int] = []
    for start, end in runs:
        for i in range(start + 1, end):
            near = [float(f.pixel_diff[j]) for j in range(max(start + 1, i - reach), min(end, i + reach + 1))
                    if j != i]
            if (len(near) < reach or f.pixel_diff[i] < SPIKE_RATIO * max(near + [PIXEL_GATE])
                    or f.hist_dist[i] < SPIKE_HIST):
                continue
            _, edge = _window_distance(f, range(max(start, i - 2), i), range(i, min(end, i + 2)))
            if edge >= SPIKE_EDGE:
                cuts.append(i)
    return cuts


def _build_items(f: _Features, labels: Sequence[Label], fps: float, min_shot: int,
                 cut_threshold: float) -> list[_Item]:
    spans = _transition_spans(labels, min_shot)
    in_span = np.zeros(len(labels), bool)
    for s, e in spans:
        in_span[s:e] = True
    content = np.flatnonzero(~in_span).tolist()
    new_shot: dict[int, bool] = {}
    runs: list[tuple[int, int]] = []
    run_start = content[0] if content else 0
    for k in range(1, len(content)):
        prev, cur = content[k - 1], content[k]
        structure, edge = _window_distance(f, content[max(0, k - 2):k], content[k:k + 2])
        if cur == prev + 1:
            if min(structure, edge) >= cut_threshold and f.pixel_diff[cur] >= PIXEL_GATE:
                new_shot[cur] = True
        else:
            # Editors put flashes and black frames on cuts, so across one we split
            # unless both sides plainly match.
            new_shot[cur] = max(structure, edge) >= SAME_SHOT
        if cur in new_shot or cur != prev + 1:
            runs.append((run_start, prev + 1))
            run_start = cur
    if content:
        runs.append((run_start, content[-1] + 1))
    for cut in _spike_cuts(f, runs, max(2, round(SPIKE_REACH * fps))):
        new_shot[cut] = True
    items = [item for s, e in spans for item in _span_items(labels, s, e)]
    span_starts = [s for s, _ in spans]
    starts = sorted({content[0]} | set(new_shot)) if content else []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(labels)
        k = bisect.bisect_right(span_starts, start)
        if k < len(span_starts):
            stop = min(stop, span_starts[k])
        items.append(_Item("piece", start, stop))
    items.sort(key=lambda item: item.start)
    group = -1
    for item in items:
        if item.kind == "piece":
            if group < 0 or new_shot.get(item.start, False):
                group += 1
            item.group = group
    return items


def _content_frames(items: Sequence[_Item], group: int) -> list[int]:
    return [i for item in items if item.kind == "piece" and item.group == group
            for i in range(item.start, item.end)]


def _resolve_short_shots(items: list[_Item], f: _Features, min_shot: int) -> None:
    """Turn pieces shorter than a shot into flashes or merge them into a neighbour."""
    while True:
        groups: list[int] = []
        for item in items:
            if item.kind == "piece" and item.group not in groups:
                groups.append(item.group)
        sizes = {g: len(_content_frames(items, g)) for g in groups}
        short = next((g for g in groups if sizes[g] < min_shot), None)
        if short is None or len(groups) == 1:
            return
        idx = [i for i, item in enumerate(items) if item.kind == "piece" and item.group == short]
        frames = _content_frames(items, short)
        prev = max((i for i, it in enumerate(items[:idx[0]]) if it.kind == "piece"), default=None)
        nxt = min((i for i in range(idx[-1] + 1, len(items)) if items[i].kind == "piece"), default=None)
        neighbours = []
        if prev is not None:
            neighbours.append(float(f.luma[items[prev].end - 1]))
        if nxt is not None:
            neighbours.append(float(f.luma[items[nxt].start]))
        if neighbours and float(f.luma[frames].mean()) > max(neighbours) + SPIKE_MARGIN:
            for i in idx:
                items[i].kind = "flash"
            continue
        # A transitional frame belongs to the incoming shot. Across a flash or black,
        # join whichever side it resembles.
        if nxt is not None and (prev is None or nxt == idx[-1] + 1):
            target = items[nxt].group
        elif nxt is None:
            assert prev is not None
            target = items[prev].group
        else:
            assert prev is not None
            prev_frames = _content_frames(items, items[prev].group)[-2:]
            next_frames = _content_frames(items, items[nxt].group)[:2]
            to_prev = sum(_window_distance(f, frames, prev_frames))
            to_next = sum(_window_distance(f, frames, next_frames))
            target = items[prev].group if to_prev < to_next else items[nxt].group
        for i in idx:
            items[i].group = target




def _find_dissolve(frames: NDArray[np.uint8], f: _Features, start: int, end: int, fps: float,
                   min_shot: int, cut_threshold: float) -> tuple[int, int] | None:
    """Blended frames of the clearest cross-dissolve in a run of content frames.

    A window passes when its two ends are different pictures, every frame between is close
    to a linear mix of them, and the mix only moves one way. The dissolve is the union of
    the blended frames of every passing window that overlaps the best one.
    """
    found: list[tuple[float, int, int]] = []
    for half in sorted({max(2, round(s * fps)) for s in DISSOLVE_HALF_WIDTHS}):
        for centre in range(start + half, end - half):
            a, b = centre - half, centre + half
            if min(_distance(f, a, b)) < cut_threshold:
                continue
            window = frames[a:b + 1].reshape(b - a + 1, -1).astype(np.float32)
            first, last = window[0], window[-1]
            span = last - first
            norm = float(span @ span)
            if norm <= 0:
                continue
            alpha = np.clip((window - first) @ span / norm, 0.0, 1.0)
            resid = np.abs(window - (first + alpha[:, None] * span)).mean(1)[1:-1]
            nearer = np.minimum(np.abs(window - first).mean(1), np.abs(window - last).mean(1))[1:-1]
            if nearer.mean() <= 0 or np.any(np.diff(alpha) < -0.15):
                continue
            fit = float(resid.mean() / nearer.mean())
            mixed = np.flatnonzero((alpha > 0.1) & (alpha < 0.9))
            if fit <= DISSOLVE_FIT and len(mixed) >= 2:
                found.append((fit, a + int(mixed[0]), a + int(mixed[-1]) + 1))
    if not found:
        return None
    _, d0, d1 = min(found)
    for _, s, e in sorted(found, key=lambda c: c[1]):
        if s < d1 and e > d0:
            d0, d1 = min(d0, s), max(d1, e)
    return (d0, d1) if d0 - start >= min_shot and end - d0 >= min_shot else None


def _split_dissolves(frames: NDArray[np.uint8], f: _Features, start: int, end: int, fps: float,
                     min_shot: int, cut_threshold: float, blocked: Sequence[tuple[int, int]]
                     ) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Shot ranges and dissolve ranges inside one shot. A dissolve belongs to the incoming
    shot, which starts on the first blended frame. Runs holding in-shot events are skipped."""
    edges = sorted({start, end} | {x for s, e in blocked for x in (s, e) if start < x < end})
    for lo, hi in zip(edges, edges[1:]):
        if any(s <= lo and hi <= e for s, e in blocked):
            continue
        found = _find_dissolve(frames, f, lo, hi, fps, min_shot, cut_threshold)
        if found:
            d0, d1 = found
            left, left_dissolves = _split_dissolves(frames, f, start, d0, fps, min_shot, cut_threshold, blocked)
            right, right_dissolves = _split_dissolves(frames, f, d1, end, fps, min_shot, cut_threshold, blocked)
            right[0] = (d0, right[0][1])
            return left + right, left_dissolves + [(d0, d1)] + right_dissolves
    return [(start, end)], []


def _keyframe(f: _Features, start: int, end: int, skip: Sequence[tuple[int, int]]) -> int:
    """Highest-contrast frame away from the shot's edges and its flashes or dark frames."""
    pad = max(1, (end - start) // 6) if end - start > 4 else 0
    usable = [i for i in range(start + pad, end - pad) if not any(s <= i < e for s, e in skip)]
    usable = usable or list(range(start, end))
    return max(usable, key=lambda i: float(f.contrast[i]))


def analyze_picture(frames: NDArray[np.uint8], fps: float, *, cut_threshold: float = CUT_THRESHOLD,
                    min_shot: int = 2) -> Picture:
    """Partition frames into shots, flashes and black; list in-shot events and dissolves."""
    if len(frames) == 0:
        raise ValueError("no frames to analyse")
    f = _features(frames)
    labels = _labels(f, fps)
    items = _build_items(f, labels, fps, min_shot, cut_threshold)
    _resolve_short_shots(items, f, min_shot)

    first: dict[int, int] = {}
    last: dict[int, int] = {}
    for i, item in enumerate(items):
        if item.kind == "piece":
            first.setdefault(item.group, i)
            last[item.group] = i
    shots: list[tuple[int, int]] = []
    dissolve_ranges: list[tuple[int, int]] = []
    inner: list[tuple[Literal["flash", "black"], int, int]] = []
    for g in sorted(first, key=first.__getitem__):
        blocked = []
        for item in items[first[g] + 1:last[g]]:
            if item.kind != "piece":
                inner.append((item.kind, item.start, item.end))
                blocked.append((item.start, item.end))
        parts, found = _split_dissolves(frames, f, items[first[g]].start, items[last[g]].end,
                                        fps, min_shot, cut_threshold, blocked)
        shots += parts
        dissolve_ranges += found
    covered = {i for s, e in shots for i in range(s, e)}
    rows: list[tuple[SegmentKind, int, int]] = [("shot", s, e) for s, e in shots]
    for item in items:
        if item.kind != "piece" and item.start not in covered:
            rows.append((item.kind, item.start, item.end))
    rows.sort(key=lambda r: r[1])

    segments: list[Segment] = []
    between: list[str] = []
    number = 0
    dissolve_at = {s: e for s, e in dissolve_ranges}
    for kind, s, e in rows:
        if kind != "shot":
            if segments and segments[-1].kind == kind and segments[-1].end == s:
                segments[-1] = Segment(kind, segments[-1].start, e)
            else:
                segments.append(Segment(kind, s, e))
                between.append(kind)
            continue
        number += 1
        if s in dissolve_at:
            enter = "dissolve"
        elif between:
            enter = "+".join(between)
        else:
            enter = "start" if s == 0 else "cut"
        skip = [(a, b) for _, a, b in inner if s <= a < e]
        segments.append(Segment("shot", s, e, number, enter, _keyframe(f, s, e, skip)))
        between = []
    shot_of: dict[int, int] = {i: seg.shot for seg in segments if seg.shot is not None
                               for i in range(seg.start, seg.end)}
    in_shot = [InShotEvent(kind, s, e, shot_of[s]) for kind, s, e in sorted(inner, key=lambda x: x[1])
               if shot_of.get(s) is not None]
    dissolves = [Dissolve(s, e, shot_of[s]) for s, e in dissolve_ranges if shot_of.get(s) is not None]

    stats = []
    for i, label in enumerate(labels):
        s_dist, e_dist = _distance(f, i - 1, i) if i else (0.0, 0.0)
        stats.append(FrameStats(round(float(f.luma[i]), 2), round(float(f.white[i]), 4),
                                round(float(f.contrast[i]), 2), round(float(f.hist_dist[i]), 4),
                                round(float(f.pixel_diff[i]), 2), round(s_dist, 4), round(e_dist, 4), label))
    return Picture(stats, segments, in_shot, dissolves)


def picture_from_json(data: dict[str, Any]) -> Picture:
    return Picture([FrameStats(**x) for x in data["frames"]], [Segment(**x) for x in data["segments"]],
                   [InShotEvent(**x) for x in data["in_shot"]], [Dissolve(**x) for x in data["dissolves"]])


# Audio

def decode_audio(media: Path, rate: int = SAMPLE_RATE) -> NDArray[np.float32]:
    raw = _run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(media), "-map", "0:a:0",
                "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"])
    return np.frombuffer(raw, np.float32)


def _window_db(y: NDArray[np.float32], size: int) -> list[float]:
    starts = np.arange(0, y.size, size)
    power = np.add.reduceat(y.astype(np.float64) ** 2, starts) / np.diff(np.r_[starts, y.size])
    return [round(10 * math.log10(max(float(p), 1e-12)), 2) for p in power]


def _drops(y: NDArray[np.float32], rate: int) -> list[tuple[float, float]]:
    """(time, dB) of loudness jumps between the WINDOW before and the WINDOW after a moment:
    the biggest, plus any other of at least DROP_MIN_DB and DROP_SHARE of it, DROP_GAP apart."""
    hop = 512
    count = y.size // hop
    power = (y[:count * hop].astype(np.float64) ** 2).reshape(count, hop).mean(1)
    k = max(1, round(WINDOW * rate / hop))
    total = np.r_[0.0, np.cumsum(power)]
    jumps: list[tuple[float, float]] = []
    for i in range(max(k, math.ceil(DROP_SKIP * rate / hop)), count - k + 1):
        before = (total[i] - total[i - k]) / k
        after = (total[i + k] - total[i]) / k
        if before > SILENCE_POWER and after > 0:
            jumps.append((10 * math.log10(after / before), i * hop / rate))
    jumps.sort(reverse=True)
    if not jumps or jumps[0][0] <= 0:
        return []
    floor = max(DROP_MIN_DB, DROP_SHARE * jumps[0][0])
    kept = [jumps[0]]
    for jump, t in jumps[1:]:
        if jump < floor:
            break
        if all(abs(t - other) >= DROP_GAP for _, other in kept):
            kept.append((jump, t))
    return sorted((t, jump) for jump, t in kept)


def _zscore_rows(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)


def _novelty(y: NDArray[np.float32], rate: int, beat_frames: NDArray[np.int_]) -> list[tuple[float, float]]:
    """(time, strength) of rough section boundaries: beats where timbre and loudness change,
    from a checkerboard kernel slid along a beat-synchronous self-similarity matrix."""
    import librosa
    if len(beat_frames) < 4 * NOVELTY_BEATS[0]:
        return []
    timbre = librosa.feature.mfcc(y=y, sr=rate, n_mfcc=13)[1:]
    loudness = librosa.amplitude_to_db(librosa.feature.rms(y=y), ref=1.0)
    frames = librosa.util.fix_frames(beat_frames, x_min=0)
    beats = librosa.util.sync(np.vstack([_zscore_rows(timbre), _zscore_rows(loudness)]), frames.tolist(),
                              aggregate=np.median)
    beats = beats - beats.mean(0, keepdims=True)
    beats /= np.linalg.norm(beats, axis=0, keepdims=True) + 1e-9
    similarity = beats.T @ beats
    n = similarity.shape[0]
    half = int(np.clip(round(n / 8), *NOVELTY_BEATS))
    sign = np.r_[-np.ones(half), np.ones(half)] * np.exp(-0.5 * np.linspace(-2, 2, 2 * half) ** 2)
    kernel = np.outer(sign, sign)
    padded = np.pad(similarity, half, mode="edge")
    novelty = np.maximum([float(np.sum(kernel * padded[i:i + 2 * half, i:i + 2 * half])) for i in range(n)], 0)
    if novelty.max() <= 0:
        return []
    novelty /= novelty.max()
    times = librosa.frames_to_time(frames, sr=rate)
    return [(float(times[i]), float(novelty[i])) for i in range(2, n - 2)
            if novelty[i] >= NOVELTY_FLOOR and novelty[i] == novelty[max(0, i - half):i + half + 1].max()]


def analyze_audio(media: Path, offset: float = 0.0) -> AudioMap:
    import librosa
    from scipy import signal
    y = decode_audio(media)
    if y.size < SAMPLE_RATE:
        raise ValueError("audio is shorter than one second")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SAMPLE_RATE)
    beats = librosa.frames_to_time(beat_frames, sr=SAMPLE_RATE)
    strength = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE)
    peaks = librosa.onset.onset_detect(onset_envelope=strength, sr=SAMPLE_RATE)
    strong = peaks[strength[peaks] >= np.percentile(strength, ONSET_PERCENTILE)]
    onsets = librosa.frames_to_time(strong, sr=SAMPLE_RATE)
    size = round(WINDOW * SAMPLE_RATE)
    low = signal.sosfiltfilt(signal.butter(4, LOW_BAND, "lowpass", fs=SAMPLE_RATE, output="sos"), y)
    return AudioMap(duration=round(y.size / SAMPLE_RATE, 3), offset=offset,
                    tempo=round(float(np.atleast_1d(tempo)[0]), 2),
                    beats=[round(float(t) + offset, 3) for t in beats],
                    onsets=[round(float(t) + offset, 3) for t in onsets],
                    rms_db=_window_db(y, size), low_db=_window_db(low.astype(np.float32), size),
                    drops=[Drop(round(t + offset, 3), round(jump, 2)) for t, jump in _drops(y, SAMPLE_RATE)],
                    boundaries=[Boundary(round(t + offset, 3), round(s, 3))
                                for t, s in _novelty(y, SAMPLE_RATE, beat_frames)])


def audio_from_json(data: dict[str, Any]) -> AudioMap:
    return AudioMap(**(data | {"drops": [Drop(**d) for d in data["drops"]],
                               "boundaries": [Boundary(**b) for b in data["boundaries"]]}))


def transcribe_words(media: Path, model_name: str, lang: str | None, offset: float = 0.0) -> list[Word]:
    """Word timings from faster-whisper. The words themselves are discarded."""
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    # Silero VAD drops vocals that sit under music, so every window is decoded.
    segments, _ = model.transcribe(str(media), language=lang, word_timestamps=True, vad_filter=False)
    return [Word(round(w.start + offset, 3), round(w.end + offset, 3), round(w.probability, 3))
            for s in segments for w in (s.words or []) if w.end > w.start]


def phrases_from(words: Sequence[Word]) -> list[Phrase]:
    phrases: list[Phrase] = []
    for w in sorted(words, key=lambda w: w.start):
        if phrases and w.start - phrases[-1].end <= PHRASE_GAP:
            p = phrases[-1]
            phrases[-1] = Phrase(p.start, max(p.end, w.end), p.words + 1)
        else:
            phrases.append(Phrase(w.start, w.end, 1))
    return phrases


# Timeline

def nearest(t: float, times: Sequence[float]) -> Nearest | None:
    if not times:
        return None
    i = bisect.bisect_left(times, t)
    best = min(times[max(0, i - 1):i + 1], key=lambda x: abs(x - t))
    return Nearest(best, round(t - best, 3))


def nearest_word_edge(t: float, words: Sequence[Word]) -> Nearest | None:
    edges = sorted([(w.start, "start") for w in words] + [(w.end, "end") for w in words])
    near = nearest(t, [time for time, _ in edges])
    if near is None:
        return None
    return Nearest(near.time, near.delta, next(edge for time, edge in edges if time == near.time))


def _sections(bounds: Sequence[float], phrases: Sequence[Phrase], drops: Sequence[Drop]) -> list[Section]:
    """Sections between consecutive bounds, each with its main phrase and the drop it starts on."""
    starts = list(bounds[:-1])
    drop_at: dict[int, Drop] = {}
    for d in drops:
        near = nearest(d.time, starts)
        if near is not None and abs(near.delta) <= SNAP:
            drop_at[starts.index(near.time)] = d
    sections = []
    for i, (a, b) in enumerate(zip(bounds, bounds[1:])):
        overlap = [max(0.0, min(b, p.end) - max(a, p.start)) for p in phrases]
        best = max(range(len(phrases)), key=overlap.__getitem__, default=None)
        phrase = best + 1 if best is not None and overlap[best] > (b - a) / 2 else None
        sections.append(Section(round(a, 3), round(b, 3), phrase, round(sum(overlap) / (b - a), 3),
                                drop_at.get(i)))
    return sections


def sections_from(phrases: Sequence[Phrase], drops: Sequence[Drop], boundaries: Sequence[float],
                  end: float) -> list[Section]:
    """Edit sections: split at vocal phrase edges and drops, each snapped to a shot boundary within SNAP."""
    def snap(t: float) -> float:
        near = nearest(t, boundaries)
        return near.time if near and abs(near.delta) <= SNAP else t

    marks = sorted([snap(t) for p in phrases for t in (p.start, p.end)] + [snap(d.time) for d in drops])
    bounds = [0.0]
    for t in marks:
        if WINDOW <= t - bounds[-1] and t <= end - WINDOW:
            bounds.append(t)
    return _sections(bounds + [end], phrases, drops)


def audio_sections(audio: AudioMap, phrases: Sequence[Phrase], end: float) -> list[Section]:
    """Song sections: split at drops and at novelty boundaries more than MIN_SECTION from a drop."""
    drop_times = [d.time for d in audio.drops]
    marks = sorted(drop_times + [b.time for b in audio.boundaries
                                 if all(abs(b.time - t) > MIN_SECTION for t in drop_times)])
    bounds = [0.0]
    for t in marks:
        if MIN_SECTION <= t - bounds[-1] and t <= end - MIN_SECTION:
            bounds.append(t)
    return _sections(bounds + [end], phrases, audio.drops)


def build_sections(info: MediaInfo, picture: Picture | None, audio: AudioMap | None,
                   phrases: Sequence[Phrase]) -> list[Section]:
    audio_end = audio.offset + audio.duration if audio else 0.0
    if picture is None:
        return audio_sections(audio, phrases, round(audio_end, 3)) if audio else []
    fps = _video(info).fps
    boundaries = sorted({round(x / fps, 3) for s in picture.segments if s.kind == "shot" for x in (s.start, s.end)})
    end = round(max(len(picture.frames) / fps, audio_end), 3)
    return sections_from(phrases, audio.drops if audio else [], boundaries, end)


def timeline_json(an: Analysis) -> dict[str, Any]:
    def segment(s: Segment) -> dict[str, Any]:
        row: dict[str, Any] = {"kind": s.kind, "start": an.time(s.start), "end": an.time(s.end),
                               "start_frame": s.start, "end_frame": s.end}
        if s.kind == "shot":
            row |= {"shot": s.shot, "enter": s.enter, "keyframe_frame": s.keyframe,
                    "keyframe_time": an.time(s.keyframe) if s.keyframe is not None else None}
        row["sync"] = {k: asdict(v) if v else None for k, v in an.sync(an.time(s.start)).items()}
        return row

    picture = None
    if an.picture is not None:
        picture = {
            "frames": len(an.picture.frames),
            "segments": [segment(s) for s in an.picture.segments],
            "in_shot": [asdict(e) | {"start_time": an.time(e.start), "end_time": an.time(e.end)}
                        for e in an.picture.in_shot],
            "dissolves": [asdict(d) | {"start_time": an.time(d.start), "end_time": an.time(d.end)}
                          for d in an.picture.dissolves],
        }
    return {
        "media": asdict(an.info),
        "picture": picture,
        "audio": asdict(an.audio) if an.audio else None,
        "words": [asdict(w) for w in an.words],
        "phrases": [asdict(p) for p in an.phrases],
        "sections": [asdict(s) for s in an.sections],
        "sheets": asdict(an.sheets) if an.sheets else None,
        "settings": asdict(an.settings),
    }


# Contact sheets

def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def write_sheets(info: MediaInfo, picture: Picture, out: Path, band: tuple[float, float],
                 step: float) -> SheetIndex:
    """One keyframe per shot, and the caption band every `step` seconds, in one decode pass."""
    video = _video(info)
    sheets = out / "sheets"
    sheets.mkdir(exist_ok=True)
    for old in [*sheets.glob("shots_*.jpg"), *sheets.glob("captions_*.jpg")]:
        old.unlink()
    width = _even(min(video.width, SHEET_WIDTH))
    height = _even(video.height * width / video.width)
    count = len(picture.frames)
    wanted: dict[int, list[float]] = {}
    for k in range(math.ceil(count / video.fps / step)):
        t = round(k * step, 3)
        wanted.setdefault(min(count - 1, round(t * video.fps)), []).append(t)
    shots = [s for s in picture.segments if s.kind == "shot" and s.keyframe is not None]
    keys = {s.keyframe for s in shots if s.keyframe is not None}
    top = min(height - 1, round(band[0] * height))
    bottom = max(top + 1, round(band[1] * height))
    scale = THUMB_WIDTH / max(width, height)
    thumb = (_even(width * scale), _even(height * scale))
    bands: list[tuple[float, Image.Image]] = []
    thumbs: dict[int, Image.Image] = {}
    last_needed = max([*wanted, *keys])
    for i, frame in enumerate(iter_frames(info, width, height)):
        if i in wanted:
            crop = Image.fromarray(frame[top:bottom])
            bands += [(t, crop) for t in wanted[i]]
        if i in keys:
            thumbs[i] = Image.fromarray(frame).resize(thumb, Image.Resampling.LANCZOS)
        if i >= last_needed:
            break
    if len(thumbs) < len(keys):
        raise RuntimeError("video ended before every keyframe was decoded")

    shot_files: list[tuple[str, int, int]] = []
    label = 32
    font = _font(20)
    cols = max(1, (THUMB_WIDTH * THUMB_COLUMNS) // thumb[0])
    per_sheet = min(THUMBS_PER_SHEET, cols * max(1, SHEET_HEIGHT // (thumb[1] + label)))
    for n, k in enumerate(range(0, len(shots), per_sheet), 1):
        group = shots[k:k + per_sheet]
        rows = math.ceil(len(group) / cols)
        sheet = Image.new("RGB", (thumb[0] * cols, (thumb[1] + label) * rows), "#181818")
        draw = ImageDraw.Draw(sheet)
        for j, s in enumerate(group):
            x, y = j % cols * thumb[0], j // cols * (thumb[1] + label)
            assert s.keyframe is not None and s.shot is not None
            sheet.paste(thumbs[s.keyframe], (x, y))
            draw.text((x + 4, y + thumb[1] + 3), f"S{s.shot}  {s.start / video.fps:.2f}-{s.end / video.fps:.2f}s",
                      fill="white", font=font)
        name = f"shots_{n:02d}.jpg"
        sheet.save(sheets / name, quality=92)
        shot_files.append((f"sheets/{name}", group[0].shot or 0, group[-1].shot or 0))

    caption_files: list[tuple[str, float, float]] = []
    band_height, gap, label_width = bottom - top, 4, 96
    per_sheet = max(1, SHEET_HEIGHT // (band_height + gap))
    for n, k in enumerate(range(0, len(bands), per_sheet), 1):
        chunk = bands[k:k + per_sheet]
        sheet = Image.new("RGB", (label_width + width, len(chunk) * (band_height + gap)), "#181818")
        draw = ImageDraw.Draw(sheet)
        for j, (t, crop) in enumerate(chunk):
            y = j * (band_height + gap)
            sheet.paste(crop, (label_width, y))
            draw.text((8, y + max(0, band_height // 2 - 12)), f"{t:.2f}s", fill="white", font=font)
        name = f"captions_{n:02d}.jpg"
        sheet.save(sheets / name, quality=92)
        caption_files.append((f"sheets/{name}", chunk[0][0], chunk[-1][0]))
    return SheetIndex(shot_files, caption_files)


# Report

def _cell(near: Nearest | None) -> str:
    if near is None or abs(near.delta) > REACH:
        return "—"
    edge = f" {near.edge}" if near.edge else ""
    return f"{near.time:.2f}{edge} ({near.delta:+.2f})"


def _count(n: int, noun: str, plural: str | None = None) -> str:
    return f"{n} {noun if n == 1 else plural or noun + 's'}"


def _power_db(values: Sequence[float]) -> float:
    """Mean of dB values taken as power; -120 for no values."""
    if not values:
        return -120.0
    return 10 * math.log10(sum(10 ** (v / 10) for v in values) / len(values))


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


def _windows(audio: AudioMap, start: float, end: float) -> list[int]:
    return [i for i in range(len(audio.rms_db)) if start <= audio.offset + (i + 0.5) * WINDOW < end]


def _loudness(audio: AudioMap, start: float, end: float) -> str:
    values = [audio.rms_db[i] for i in _windows(audio, start, end)]
    return f"{_power_db(values):.1f} / {max(values):.1f}" if values else "—"


def _header(an: Analysis) -> list[str]:
    name = Path(an.info.path).name
    if an.picture is None:
        length = an.audio.duration if an.audio else an.info.duration
        return [f"# Audio analysis: {name}", "",
                f"Measured by `scene_frames/analyze_edit.py` from audio only. {_clock(length)} long. "
                "Times are seconds from the first sample.",
                "Word timings are machine-made; no lyric text is stored.", ""]
    video, segments = _video(an.info), an.picture.segments
    visible = [s for s in segments if s.kind != "black"]
    span = (f"Picture runs {an.time(visible[0].start):.2f}–{an.time(visible[-1].end):.2f} s"
            if visible else "The picture is black throughout")
    return [f"# Edit analysis: {name}", "",
            f"Measured by `scene_frames/analyze_edit.py`. {an.info.duration:.2f} s, {video.width}×{video.height}, "
            f"{video.fps:g} fps, so one frame is {1000 / video.fps:.0f} ms. {span}. Times are seconds from the "
            "first frame.", "Word timings are machine-made; no lyric or dialogue text is stored.", ""]


def _skeleton(an: Analysis) -> list[str]:
    audio = an.audio
    lines = ["## Skeleton", ""]
    if audio:
        lines.append(f"- **Tempo:** {audio.tempo:.1f} BPM, so a beat is {60 / audio.tempo:.3f} s; "
                     f"{len(audio.beats)} beats tracked.")
        jumps = ", ".join(f"{d.time:.2f} s ({d.jump_db:+.1f} dB)" for d in audio.drops) or "none"
        lines.append(f"- **{'Drop' if len(audio.drops) == 1 else 'Drops'}:** {jumps}, measured as the loudness "
                     f"change from the {WINDOW:g} s before to the {WINDOW:g} s after.")
        bounds = ", ".join(f"{b.time:.2f}" for b in audio.boundaries) or "none found"
        lines.append(f"- **Section boundaries in the audio** (timbre and loudness novelty, on beats): {bounds}.")
    else:
        lines.append("- **Audio:** none.")
    if an.phrases:
        lines.append("- **Vocal phrases:** " + "; ".join(
            f"{i}. {p.start:.2f}–{p.end:.2f} ({p.words} words)" for i, p in enumerate(an.phrases, 1)))
    elif an.settings.whisper_model is None:
        lines.append("- **Vocal phrases:** not transcribed.")
    else:
        lines.append("- **Vocal phrases:** none found.")
    shots = an.shots
    if an.picture is not None:
        flashes = sum(s.kind == "flash" for s in an.picture.segments)
        blacks = sum(s.kind == "black" for s in an.picture.segments)
        mean_shot = sum(s.end - s.start for s in shots) / len(shots) / an.fps if shots else 0.0
        lines.append(f"- **Picture:** {_count(len(shots), 'shot')} averaging {mean_shot:.2f} s, "
                     f"{_count(flashes, 'flash', 'flashes')}, {_count(blacks, 'stretch', 'stretches')} of black, "
                     f"{_count(len(an.picture.dissolves), 'dissolve')}.")
        changes = [an.time(s.end) for s in shots[:-1]]  # when each outgoing shot leaves
        if changes:
            counts = [bisect.bisect_left(changes, round(t + 1.0, 3)) - i for i, t in enumerate(changes)]
            best = counts.index(max(counts))
            lines.append(f"- **Busiest second:** {_count(counts[best], 'shot change')} in "
                         f"{changes[best]:.2f}–{changes[best] + 1:.2f}.")
    return lines + [""]


def _pacing_table(an: Analysis) -> list[str]:
    assert an.picture is not None
    lines = ["| Section | Time (s) | Vocals | Shots | Avg shot (s) | Flashes | Loudness mean / peak (dB) |",
             "|---|---|---|---|---|---|---|"]
    flashes = [s for s in an.picture.segments if s.kind == "flash"]
    for n, sec in enumerate(an.sections, 1):
        inside = [s for s in an.shots if sec.start <= an.time(s.start) < sec.end]
        flash_count = sum(sec.start <= an.time(s.start) < sec.end for s in flashes)
        avg = f"{sum(s.end - s.start for s in inside) / len(inside) / an.fps:.2f}" if inside else "—"
        loud = _loudness(an.audio, sec.start, sec.end) if an.audio else "—"
        vocals = f"phrase {sec.phrase}" if sec.phrase else "none"
        drop = " (drop)" if sec.drop else ""
        lines.append(f"| {n}{drop} | {sec.start:.2f}–{sec.end:.2f} | {vocals} | {len(inside)} | {avg} | "
                     f"{flash_count} | {loud} |")
    return lines + [""]


def _cut_table(an: Analysis) -> list[str]:
    assert an.picture is not None
    lines = ["## Cut by cut", "",
             "Each row is measured at its first frame. For a shot after a dissolve, that is the first blended "
             "frame. Beat, onset and word give the nearest event and the row's offset from it (+ means the row "
             f"starts after the event); events more than {REACH:g} s away are left out. \"On\" lists events within "
             f"{SYNC:.2f} s.", "",
             "| # | Row | In–out (s) | Length | Enter | Beat | Strong onset | Word edge | On |",
             "|---|---|---|---|---|---|---|---|---|"]
    dissolve_of = {d.shot: d for d in an.picture.dissolves}
    for n, s in enumerate(an.picture.segments, 1):
        sync = an.sync(an.time(s.start))
        on = ", ".join(k for k, v in sync.items() if v and abs(v.delta) <= SYNC + 1e-9) or "—"
        if s.kind == "shot":
            row = f"S{s.shot}"
            enter = s.enter or "cut"
            if s.shot in dissolve_of:
                enter = f"dissolve to {an.time(dissolve_of[s.shot].end):.2f}"
        else:
            row, enter = s.kind, "—"
        lines.append(f"| {n} | {row} | {an.time(s.start):.2f}–{an.time(s.end):.2f} | "
                     f"{(s.end - s.start) / an.fps:.2f} s · {s.end - s.start}f | {enter} | {_cell(sync['beat'])} | "
                     f"{_cell(sync['onset'])} | {_cell(sync['word'])} | {on} |")
    lines.append("")
    if an.picture.in_shot:
        lines += ["In-shot flashes and dark frames (not cuts): " + "; ".join(
            f"S{e.shot} {'white flash' if e.kind == 'flash' else 'dark frames'} "
            f"{an.time(e.start):.2f}–{an.time(e.end):.2f}" for e in an.picture.in_shot) + ".", ""]
    return lines


def _section_table(an: Analysis) -> list[str]:
    """Song sections for audio-only input."""
    audio = an.audio
    if audio is None or not an.sections:
        return []
    lines = ["## Sections", "",
             "Split at drops and audio section boundaries. Vocals is the main phrase, or the share of the "
             "section with vocals.", "",
             "| # | Time (s) | Length (s) | Beats | Vocals | Loudness mean / peak (dB) | Low band mean (dB) "
             "| Starts on |",
             "|---|---|---|---|---|---|---|---|"]
    strength = {b.time: b.strength for b in audio.boundaries}
    for n, sec in enumerate(an.sections, 1):
        beats = sum(sec.start <= t < sec.end for t in audio.beats)
        vocals = f"phrase {sec.phrase}" if sec.phrase else (f"{sec.vocal_share:.0%}" if sec.vocal_share else "none")
        low = _power_db([audio.low_db[i] for i in _windows(audio, sec.start, sec.end)])
        if sec.drop:
            starts = f"drop {sec.drop.jump_db:+.1f} dB"
        elif sec.start in strength:
            starts = f"boundary {strength[sec.start]:.2f}"
        else:
            starts = "start"
        lines.append(f"| {n} | {sec.start:.2f}–{sec.end:.2f} | {sec.end - sec.start:.2f} | {beats} | {vocals} | "
                     f"{_loudness(audio, sec.start, sec.end)} | {low:.1f} | {starts} |")
    return lines + [""]


def _audio_detail(an: Analysis) -> list[str]:
    audio, lines = an.audio, []
    if audio:
        # A quarter-second grid for short clips; whole seconds for songs.
        group = 1 if audio.duration <= LONG_AUDIO else round(1 / WINDOW)
        cell = WINDOW * group
        rms = [_power_db(audio.rms_db[i:i + group]) for i in range(0, len(audio.rms_db), group)]
        low = [_power_db(audio.low_db[i:i + group]) for i in range(0, len(audio.low_db), group)]
        lines += ["## Audio detail", "",
                  f"Loudness in dBFS per {cell:g} s, with the band below {LOW_BAND:g} Hz after the slash.", "",
                  "| s | " + " | ".join(f"+{k * cell:.2f}" for k in range(4)) + " |", "|---|---|---|---|---|"]
        for r in range(0, len(rms), 4):
            cells = [f"{rms[i]:.1f} / {low[i]:.1f}" for i in range(r, min(r + 4, len(rms)))]
            lines.append(f"| {audio.offset + r * cell:g} | " + " | ".join(cells + [""] * (4 - len(cells))) + " |")
        lines += ["", f"**Beats:** {' '.join(f'{t:.2f}' for t in audio.beats)}", "",
                  f"**Strong onsets** (at or above the {ONSET_PERCENTILE:g}th percentile of onset strength): "
                  f"{' '.join(f'{t:.2f}' for t in audio.onsets)}", ""]
    if an.phrases:
        lines += ["**Word timings** by phrase:", ""]
        for i, p in enumerate(an.phrases, 1):
            timed = [w for w in an.words if p.start <= w.start and w.end <= p.end]
            lines.append(f"{i}. " + ", ".join(f"{w.start:.2f}–{w.end:.2f}" for w in timed))
        lines.append("")
    return lines


def _sheet_lines(an: Analysis) -> list[str]:
    if an.sheets is None:
        return []
    top, bottom = an.settings.caption_band
    return (["## Contact sheets", ""]
            + [f"- [{path}]({path}): S{a}–S{b}" for path, a, b in an.sheets.shots]
            + [f"- [{path}]({path}): caption band ({top:.0%}–{bottom:.0%} of the frame height) every "
               f"{an.settings.caption_step:g} s, {a:.2f}–{b:.2f} s" for path, a, b in an.sheets.captions] + [""])


def render_report(an: Analysis) -> str:
    lines = _header(an) + _skeleton(an)
    if an.picture is not None:
        lines += _pacing_table(an) + _cut_table(an)
    else:
        lines += _section_table(an)
    st = an.settings
    picture = (f"Cut threshold {st.cut_threshold:g}, minimum shot {st.min_shot} frames, "
               if an.picture is not None else "")
    lines += _audio_detail(an) + _sheet_lines(an) + [
        "## Settings", "",
        f"{picture}Whisper {st.whisper_model or 'skipped'}{f' ({st.lang})' if st.lang else ''}; versions: "
        f"picture {PICTURE_VERSION}, audio {AUDIO_VERSION}, words {WORDS_VERSION}.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media", type=Path, help="a video, or an audio file for the audio report only")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--cut-threshold", type=unit_float, default=CUT_THRESHOLD,
                    help=f"structure and edge distance needed for a hard cut (default {CUT_THRESHOLD})")
    ap.add_argument("--min-shot", type=positive_int, default=2, help="shortest shot in frames (default 2)")
    ap.add_argument("--caption-band", type=caption_band, default=(0.75, 1.0), metavar="TOP:BOTTOM",
                    help="frame rows to strip for captions, as fractions of the height (default 0.75:1)")
    ap.add_argument("--caption-step", type=positive_float, default=WINDOW,
                    help=f"seconds between caption strips (default {WINDOW})")
    ap.add_argument("--whisper-model", default="small.en",
                    help="faster-whisper model for word timings (default small.en)")
    ap.add_argument("--lang", help="language code for a multilingual Whisper model; default auto-detect")
    ap.add_argument("--no-words", action="store_true", help="skip Whisper word timings")
    a = ap.parse_args()
    media, out = a.media.resolve(), a.out_dir.resolve()
    if not media.is_file():
        ap.error(f"no such file: {media}")
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    stat = media.stat()
    source = {"source": str(media), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    info = probe(media)

    picture: Picture | None = None
    if info.video is not None:
        picture_sig = source | {"cut_threshold": a.cut_threshold, "min_shot": a.min_shot, "version": PICTURE_VERSION}
        data = cached(cache / "picture.json", picture_sig)
        if data is None:
            picture = analyze_picture(decode_frames(info), info.video.fps, cut_threshold=a.cut_threshold,
                                      min_shot=a.min_shot)
            save(cache / "picture.json", {"signature": picture_sig, "data": asdict(picture)})
        else:
            picture = picture_from_json(data)
        print(f"picture: {sum(s.kind == 'shot' for s in picture.segments)} shots, "
              f"{sum(s.kind == 'flash' for s in picture.segments)} flashes", flush=True)

    audio: AudioMap | None = None
    if info.has_audio:
        audio_sig = source | {"version": AUDIO_VERSION}
        data = cached(cache / "audio.json", audio_sig)
        if data is None:
            audio = analyze_audio(media, info.audio_offset)
            save(cache / "audio.json", {"signature": audio_sig, "data": asdict(audio)})
        else:
            audio = audio_from_json(data)
        drop = audio.drop
        print(f"audio: {audio.tempo:.1f} BPM, {_count(len(audio.drops), 'drop')}"
              f"{f', biggest at {drop.time:.2f}s' if drop else ''}", flush=True)

    words: list[Word] = []
    model = None if a.no_words or not info.has_audio else a.whisper_model
    if model is not None:
        words_sig = source | {"model": model, "lang": a.lang, "version": WORDS_VERSION}
        data = cached(cache / "words.json", words_sig)
        if data is None:
            words = transcribe_words(media, model, a.lang, info.audio_offset)
            save(cache / "words.json", {"signature": words_sig, "data": [asdict(w) for w in words]})
        else:
            words = [Word(**w) for w in data]
        print(f"words: {len(words)} timed", flush=True)

    phrases = phrases_from(words)
    sheets = write_sheets(info, picture, out, a.caption_band, a.caption_step) if picture is not None else None
    analysis = Analysis(info, picture, audio, words, phrases, build_sections(info, picture, audio, phrases), sheets,
                        Settings(a.cut_threshold, a.min_shot, a.caption_band, a.caption_step, model, a.lang))
    save(out / "timeline.json", timeline_json(analysis))
    if picture is not None:
        save(out / "frames.json", [{"frame": i, "time": analysis.time(i)} | asdict(s)
                                   for i, s in enumerate(picture.frames)])
    (out / "report.md").write_text(render_report(analysis))
    print(f"report -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
