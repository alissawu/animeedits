"""Draw frames from a source or an edit so you can see what happens across shots and cuts.

usage:
  python frames.py SOURCE IN OUT OUT.jpg [--crop-x F] [--index DIR | --no-index]           scene board
  python frames.py SOURCE IN OUT OUT.jpg --dense [--fps 6] [--crop-x F] [--index DIR]       shot detail
  python frames.py --edl EDL.json OUT.jpg [--cut N | A-B] [--song-map FILE] [--lyrics FILE.lrc] [--crop-x F]
                                                                                            cut pairs

IN and OUT are source seconds (1255.5) or clock times (20:55.5, 0:20:55.5). Every tile is
stamped with the source time of the frame it shows, to 0.01 s. Nothing here captions or
judges a picture: it measures how much the picture changes (the mean absolute RGB
difference between consecutive frames at analyze_edit.py's analysis size) and flags white
and black frames with analyze_edit.py's thresholds. Pans, zooms, camera shake and
lighting changes register as change as much as a character moving does.

Scene board (default): one row per shot inside IN..OUT, from the video's index
(out/index/NAME/shots.json, NAME = the file name without extension) or detected when there
is none. Each row holds the shot's first frame, its last frame and in-between frames spaced
evenly by accumulated change, not by time: a still shot gets one middle frame, a shot with
action gets more frames (up to MAX_MIDDLE), placed where the picture changes. A strip under
each row plots the change frame by frame, with the chosen frames and any flashes marked.
Dialogue words and subtitle cues from the index are printed under the shot they fall in.

Shot detail (--dense): a grid of frames every 1/--fps seconds. Consecutive tiles that barely
differ merge into one tile marked "held 0.8 s". Under the grid, the change curve on one time
axis with shot boundaries, flashes, the tiles' positions and dialogue words, then a list of
where change starts, peaks and settles. Refuses more than MAX_TILES tiles.

Cut pairs (--edl): one row per cut, numbered like edl_table.md and proof.jpg (cut N is where
clip #N enters): the last frames of the outgoing clip, a divider, the first frames of the
incoming clip. Frames are rendered with render.py's own segment pipeline, so speed,
keyframes, crop and color match the render. Composited layers (flash and crossfade
transitions, overlays, captions) are not drawn; the transition is printed instead. Under
each row: the song's beats and sung words within CUT_REACH seconds of the cut, from an
analyze_edit.py map of the song (found next to the EDL, or --song-map), plus lines from an
LRC file (--lyrics) and any audio layer playing. Recognized text and confidence are
shown separately from published reference lines. Intention notes and declared sync
claims are hidden by default; --show-intent includes them for comparison after review.

--crop-x F previews a 9:16 crop centred at F (0 left, 1 right) with render.py's crop math.
With --edl it crops the edit's 16:9 output, as a vertical cut of the same edit would.
"""
import argparse
from collections.abc import Sequence
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

import analyze_edit
import lyrics as lyric_reference
from edl import Clip, Crop, Edl, EdlError, Output, clip_frames, load_edl, song_pieces, source_offset
import render

Kind = Literal["shot", "flash", "black"]
Font = ImageFont.FreeTypeFont

NOISE_FLOOR = 1.5     # per-frame change (mean |ΔRGB|, 0-255) below this is grain or compression
CHANGE_UNIT = 12.0    # accumulated change above the floor that earns a shot one more in-between frame
MAX_MIDDLE = 6        # in-between frames per shot on the scene board
MAX_SHOTS = 30        # rows on one scene board
HOLD_DIFF = 2.0       # dense tiles closer than this to the first tile of their run merge into it
ACTIVE = 2.5          # a frame changing at least this much from the last counts as the picture moving
MERGE_GAP = 0.35      # seconds; moving frames closer than this are one stretch, bridging anime's held drawings
CHANGE_SCALE = 12.0   # change drawn at full height in the plots; larger values are clipped
MAX_TILES = 60        # dense tiles on one sheet
PAIR_FRAMES = 3       # frames either side of a cut
CUT_REACH = 1.0       # seconds either side of a cut for beats, words and lyrics
VERTICAL = (9, 16)

BOARD_TILE = 240      # long side of a tile, pixels
DENSE_TILE = 200
PAIR_TILE = 256
DENSE_WIDTH = 2000    # target sheet width of the dense grid
PAD = 8
BG, FG, DIM, ACCENT, CUT_RED = "#161616", "#eeeeee", "#9a9a9a", "#ffd400", "#ff5a5a"
TEXT_FONTS = ("Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf",
              "/System/Library/Fonts/Supplemental/Arial Unicode.ttf", "NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "DejaVuSans.ttf")
ROOT = Path(__file__).resolve().parents[1]


class FramesError(RuntimeError):
    pass


# ---- sources and decoding ----

@dataclass(frozen=True)
class Source:
    path: Path
    duration: float
    fps: float
    stream: int           # absolute index of the video stream
    width: int
    height: int
    sar: float


def open_source(path: Path) -> Source:
    if not path.is_file():
        raise FramesError(f"file not found: {path}")
    try:
        pr = render.probe(path)
        info = analyze_edit.probe(path)
    except (ValueError, RuntimeError) as e:
        raise FramesError(f"cannot read {path}: {e}") from None
    if pr.video_index is None or info.video is None or pr.image:
        raise FramesError(f"{path} has no video stream")
    duration = pr.duration or info.duration
    if not duration:
        raise FramesError(f"cannot tell how long {path} is")
    return Source(path, duration, info.video.fps, pr.video_index, info.video.width, info.video.height, pr.sar)


def crop_filter(crop_x: float, sar: float, fps: float) -> str:
    """render.py's static crop: the largest 9:16 window centred at crop_x, kept inside the frame."""
    clip = Clip(source="", src_in=0.0, timeline_in=0.0, timeline_out=1.0, crop=Crop(x=crop_x))
    return render.framing_filter(clip, 0, Output(VERTICAL[0] * 120, VERTICAL[1] * 120, max(1, round(fps))), sar)


def shown_aspect(src: Source, crop_x: float | None) -> float:
    """Width / height of the picture as shown."""
    return VERTICAL[0] / VERTICAL[1] if crop_x is not None else src.width * src.sar / src.height


def fit(aspect: float, long_side: int) -> tuple[int, int]:
    if aspect >= 1:
        return long_side, max(2, 2 * round(long_side / aspect / 2))
    return max(2, 2 * round(long_side * aspect / 2)), long_side


SHOWINFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:(-?[0-9.e+-]+)")


def decode(src: Source, start: float, end: float, size: tuple[int, int], *, crop_x: float | None = None,
           pick: Sequence[int] | None = None) -> tuple[list[float], NDArray[np.uint8]]:
    """Frames nearest to the times in [start, end), as source seconds and (n, h, w, 3) RGB.

    The seek lands half a source frame early, as render.py's does, so a time picks its nearest
    frame. Times are the frames' own timestamps relative to the start of the file, which is
    what src_in means in an EDL. With `pick`, only those frame numbers (counted from the first
    frame of the span) are returned, in order.
    """
    seek = max(0.0, start - 0.5 / src.fps)
    filters = []
    if pick is not None:
        filters.append(f"select='{_frame_set(pick)}'")
    if crop_x is not None:
        filters.append(crop_filter(crop_x, src.sar, src.fps))
    filters += [f"scale={size[0]}:{size[1]}:flags=area", "setsar=1", "showinfo"]
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info",
           "-ss", f"{seek:.6f}", "-t", f"{end - 0.5 / src.fps - seek:.6f}", "-i", str(src.path),
           "-map", f"0:{src.stream}", "-an", "-sn", "-dn", "-vf", ",".join(filters),
           "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, capture_output=True)
    err = proc.stderr.decode(errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(line for line in err.splitlines()[-8:] if "showinfo" not in line)
        raise FramesError(f"ffmpeg cannot decode {src.path.name} at {start:.2f} s:\n{tail}")
    times = [seek + float(m.group(3)) for m in SHOWINFO.finditer(err)]
    step = size[0] * size[1] * 3
    if len(proc.stdout) != step * len(times):
        raise FramesError(f"ffmpeg returned {len(proc.stdout)} bytes for {len(times)} frames of {src.path.name}")
    frames = np.frombuffer(proc.stdout, np.uint8).reshape(len(times), size[1], size[0], 3)
    return times, frames


def _frame_set(indices: Sequence[int]) -> str:
    """select expression true on the given frame numbers, as runs nested in balanced sums."""
    runs: list[list[int]] = []
    for n in sorted(set(indices)):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    terms = [f"between(n,{a},{b})" if a != b else f"eq(n,{a})" for a, b in runs]

    def join(items: list[str]) -> str:
        if len(items) == 1:
            return items[0]
        mid = len(items) // 2
        return f"({join(items[:mid])}+{join(items[mid:])})"
    return join(terms)


# ---- measuring change ----

@dataclass(frozen=True)
class Measured:
    src: Source
    start: float
    end: float
    crop_x: float | None
    times: list[float]                       # source seconds of every frame in the span
    small: NDArray[np.uint8]                 # the frames at analysis size
    change: list[float]                      # mean |ΔRGB| from the previous frame; 0 for the first
    labels: list[analyze_edit.Label]         # content, white or black
    segments: list[analyze_edit.Segment]     # detected shots, flashes and black, covering every frame

    def index_at(self, t: float) -> int:
        """The frame shown at source time t."""
        return int(np.argmin(np.abs(np.asarray(self.times) - t)))


def parse_time(text: str) -> float:
    """Seconds from '1255.5', '20:55.5' or '0:20:55.5'."""
    try:
        parts = [float(p) for p in text.split(":")]
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a time: {text!r}") from None
    if len(parts) > 3 or any(p < 0 or not math.isfinite(p) for p in parts):
        raise argparse.ArgumentTypeError(f"not a time: {text!r}")
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def check_span(src: Source, start: float, end: float) -> None:
    if start >= end:
        raise FramesError(f"IN ({start:.2f} s) must be before OUT ({end:.2f} s)")
    if start >= src.duration:
        raise FramesError(f"IN ({start:.2f} s) is past the end of {src.path.name} ({src.duration:.2f} s)")
    if end > src.duration + 1 / src.fps:
        raise FramesError(f"OUT ({end:.2f} s) is past the end of {src.path.name} ({src.duration:.2f} s)")


def measure(src: Source, start: float, end: float, crop_x: float | None) -> Measured:
    check_span(src, start, end)
    size = fit(shown_aspect(src, crop_x), analyze_edit.ANALYSIS_SIDE)
    times, small = decode(src, start, end, size, crop_x=crop_x)
    if not times:
        raise FramesError(f"no frames of {src.path.name} between {start:.2f} and {end:.2f} s")
    picture = analyze_edit.analyze_picture(small, src.fps)
    return Measured(src, start, end, crop_x, times, small, [f.pixel_diff for f in picture.frames],
                    [f.label for f in picture.frames], picture.segments)


def grab(m: Measured, indices: Sequence[int], long_side: int) -> dict[int, Image.Image]:
    """Decode the given frames of a measured span at tile size, checking they are the frames measured."""
    wanted = sorted(set(indices))
    if not wanted:
        return {}
    size = fit(shown_aspect(m.src, m.crop_x), long_side)
    times, frames = decode(m.src, m.start, m.end, size, crop_x=m.crop_x, pick=wanted)
    if len(times) != len(wanted) or any(abs(t - m.times[i]) > 1e-3 for t, i in zip(times, wanted)):
        raise FramesError(f"{m.src.path.name} decoded differently on a second pass; cannot place frames exactly")
    return {i: Image.fromarray(frame) for i, frame in zip(wanted, frames)}


def masked_change(m: Measured, boundaries: set[int]) -> list[float]:
    """Change with cuts and flash or black frames (and the frame after them) set to 0."""
    out = list(m.change)
    for i in range(len(out)):
        if i in boundaries or m.labels[i] != "content" or (i > 0 and m.labels[i - 1] != "content"):
            out[i] = 0.0
    return out


@dataclass(frozen=True)
class Stretch:
    start: int        # first frame that changed
    peak: int         # frame with the largest change
    last: int         # last frame that changed: the picture holds from here
    peak_change: float


def runs_of(mask: Sequence[bool]) -> list[tuple[int, int]]:
    """[start, end) runs of True."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, value in enumerate([*mask, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            out.append((start, i))
            start = None
    return out


def stretches(change: Sequence[float], fps: float) -> list[Stretch]:
    """Runs of frames changing by at least ACTIVE, joined across holds shorter than MERGE_GAP."""
    gap = max(1, round(MERGE_GAP * fps))
    spans: list[tuple[int, int]] = []
    for i, value in enumerate(change):
        if value < ACTIVE:
            continue
        if spans and i - spans[-1][1] < gap:
            spans[-1] = (spans[-1][0], i)
        else:
            spans.append((i, i))
    out = []
    for a, b in spans:
        peak = max(range(a, b + 1), key=lambda i: change[i])
        out.append(Stretch(a, peak, b, round(change[peak], 1)))
    return out


def pick_middles(change: Sequence[float], first: int, end: int) -> list[int]:
    """In-between frames of shot frames [first, end), spaced evenly by accumulated change."""
    candidates = list(range(first + 1, end - 1))
    if not candidates:
        return []
    work = [max(0.0, change[i] - NOISE_FLOOR) for i in candidates]
    total = sum(work)
    if total <= 0:
        return [candidates[len(candidates) // 2]]
    count = min(MAX_MIDDLE, len(candidates), 1 + int(total // CHANGE_UNIT))
    cumulative = np.cumsum(work)
    picks: list[int] = []
    for k in range(count):
        target = (k + 0.5) / count * total
        picks.append(candidates[int(np.searchsorted(cumulative, target))])
    chosen = sorted(set(picks))
    # Change concentrated in one jump lands several picks on one frame: show the frame before it too.
    for i in list(chosen):
        if len(chosen) >= count:
            break
        if i - 1 > first and i - 1 not in chosen:
            chosen.append(i - 1)
    return sorted(chosen)


# ---- the index ----

@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class IndexShot:
    id: int
    start: float
    end: float


@dataclass(frozen=True)
class Index:
    directory: Path
    shots: list[IndexShot]
    words: list[Word]         # Whisper dialogue words, deduplicated across shots
    cues: list[Word]          # subtitle cues


def _number(value: object, where: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    raise FramesError(f"{where}: expected a number, got {value!r}")


def load_index(directory: Path) -> Index:
    file = directory / "shots.json"
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise FramesError(f"cannot read {file}: {e}") from None
    if not isinstance(data, list):
        raise FramesError(f"{file}: expected an array of shots")
    shots: list[IndexShot] = []
    words: dict[tuple[float, str], Word] = {}
    cues: dict[tuple[float, str], Word] = {}
    for k, raw in enumerate(data):
        where = f"{file.name}[{k}]"
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), int):
            raise FramesError(f"{where}: expected a shot with an id")
        shots.append(IndexShot(raw["id"], _number(raw.get("start"), f"{where}.start"),
                               _number(raw.get("end"), f"{where}.end")))
        for line in raw.get("dialogue") or []:
            for w in line.get("words") or []:
                text = str(w.get("word", "")).strip()
                if text:
                    start = _number(w.get("start"), f"{where}.dialogue")
                    words[(round(start, 3), text)] = Word(start, _number(w.get("end"), f"{where}.dialogue"), text)
        for cue in raw.get("subtitles") or []:
            text = " ".join(str(cue.get("text", "")).split())
            if text:
                start = _number(cue.get("start"), f"{where}.subtitles")
                cues[(round(start, 3), text)] = Word(start, _number(cue.get("end"), f"{where}.subtitles"), text)
    shots.sort(key=lambda s: s.start)
    return Index(directory, shots, sorted(words.values(), key=lambda w: w.start),
                 sorted(cues.values(), key=lambda w: w.start))


def find_index(source: Path) -> Path | None:
    """out/index/NAME beside the working directory or any folder above this script (a worktree's main checkout)."""
    name = source.stem
    for base in [Path.cwd(), *Path(__file__).resolve().parents]:
        candidate = base / "out" / "index" / name
        if (candidate / "shots.json").is_file():
            return candidate
    return None


# ---- drawing ----

_fonts: dict[int, Font] = {}


def font(size: int) -> Font:
    if size not in _fonts:
        for name in TEXT_FONTS:
            try:
                _fonts[size] = ImageFont.truetype(name, size)
                break
            except OSError:
                continue
        else:
            _fonts[size] = render.load_font(None, size, ROOT)
    return _fonts[size]


def wrap_text(text: str, f: Font, width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        lines += [" ".join(t for _, t in line) for line in render.wrap(words, f, width)] if words else [""]
    return lines


def stamp(tile: Image.Image, text: str, *, top: str | None = None, flag: str | None = None) -> Image.Image:
    """The source time in the bottom-left corner, an optional note top-left and flag top-right."""
    tile = tile.copy()
    draw = ImageDraw.Draw(tile)
    f = font(13)
    for label, corner in ((text, "bl"), (top, "tl"), (flag, "tr")):
        if not label:
            continue
        w = f.getlength(label) + 8
        x = 0 if corner in ("bl", "tl") else tile.width - w
        y = tile.height - 18 if corner == "bl" else 0
        fill = (255, 212, 0) if corner == "tr" else (0, 0, 0)
        draw.rectangle((x, y, x + w, y + 18), fill=fill)
        draw.text((x + 4, y + 1), label, font=f, fill=(0, 0, 0) if corner == "tr" else (255, 255, 255))
    return tile


def flash_name(label: analyze_edit.Label) -> str | None:
    return {"white": "WHITE", "black": "BLACK"}.get(label)


class Canvas:
    """A sheet built top to bottom: append rows, then save."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.parts: list[Image.Image] = []

    def text(self, text: str, *, size: int = 15, fill: str = FG, indent: int = PAD) -> None:
        f = font(size)
        lines = wrap_text(text, f, self.width - indent - PAD)
        line_h = round(size * 1.35)
        img = Image.new("RGB", (self.width, line_h * len(lines) + 2), BG)
        draw = ImageDraw.Draw(img)
        for k, line in enumerate(lines):
            draw.text((indent, k * line_h), line, font=f, fill=fill)
        self.parts.append(img)

    def image(self, img: Image.Image) -> None:
        self.parts.append(img)

    def gap(self, height: int = PAD) -> None:
        self.parts.append(Image.new("RGB", (self.width, height), BG))

    def rule(self) -> None:
        img = Image.new("RGB", (self.width, 9), BG)
        ImageDraw.Draw(img).line((PAD, 4, self.width - PAD, 4), fill="#3a3a3a")
        self.parts.append(img)

    def save(self, file: Path) -> None:
        sheet = Image.new("RGB", (self.width, sum(p.height for p in self.parts) + PAD), BG)
        y = 0
        for part in self.parts:
            sheet.paste(part, (0, y))
            y += part.height
        file.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(file, quality=88)


def change_strip(width: int, height: int, values: Sequence[float], *, marks: Sequence[int] = (),
                 flashes: Sequence[tuple[int, analyze_edit.Label]] = (), cuts: Sequence[int] = ()) -> Image.Image:
    """Bars of per-frame change across `width`, frame k at x = k * width / len(values)."""
    img = Image.new("RGB", (width, height), "#202020")
    draw = ImageDraw.Draw(img)
    n = max(1, len(values))
    step = width / n
    for k, v in enumerate(values):
        h = min(1.0, v / CHANGE_SCALE) * (height - 4)
        if h >= 1:
            draw.rectangle((k * step, height - 2 - h, max(k * step, (k + 1) * step - 1), height - 2), fill="#6fb3ff")
    for k, label in flashes:
        draw.rectangle((k * step, 0, (k + 1) * step, height), fill="#ffffff" if label == "white" else "#000000",
                       outline="#888888")
    for k in cuts:
        draw.line((k * step, 0, k * step, height), fill=CUT_RED, width=2)
    for k in marks:
        x = (k + 0.5) * step
        draw.polygon([(x - 4, 0), (x + 4, 0), (x, 6)], fill=ACCENT)
    return img


# ---- mode 1: scene board ----

@dataclass(frozen=True)
class Row:
    kind: Kind
    first: int        # frame numbers in the measured span, [first, end)
    end: int
    title: str
    words: list[Word]
    cues: list[Word]


def board_rows(m: Measured, index: Index | None) -> list[Row]:
    half = 0.5 / m.src.fps
    rows: list[Row] = []
    if index is None:
        number = 0
        for seg in m.segments:
            t0, t1 = m.times[seg.start], m.times[seg.end - 1] + 1 / m.src.fps
            if seg.kind == "shot":
                number += 1
                title = f"shot {number} (detected) · {t0:.2f}–{t1:.2f} s"
            else:
                title = f"{seg.kind} · {seg.end - seg.start} frame(s) · {t0:.2f}–{t1:.2f} s"
            rows.append(Row(seg.kind, seg.start, seg.end, title, [], []))
        return rows
    owner: list[IndexShot | None] = []
    for t in m.times:
        owner.append(next((s for s in index.shots if s.start - half <= t < s.end - half), None))
    k = 0
    while k < len(owner):
        shot = owner[k]
        e = k + 1
        while e < len(owner) and owner[e] is shot:
            e += 1
        if shot is not None:
            shown = ""
            if m.times[k] > shot.start + half or m.times[e - 1] + 1 / m.src.fps < shot.end - half:
                shown = f" · shown {m.times[k]:.2f}–{m.times[e - 1] + 1 / m.src.fps:.2f}"
            title = f"shot {shot.id} · {shot.start:.2f}–{shot.end:.2f} s ({shot.end - shot.start:.2f} s){shown}"
            lo, hi = m.times[k] - half, m.times[e - 1] + half
            words = [w for w in index.words if lo <= w.start < hi]
            cues = [c for c in index.cues if c.start < hi and c.end > lo]
            rows.append(Row("shot", k, e, title, words, cues))
        k = e
    return rows


def scene_board(m: Measured, index: Index | None, out: Path) -> str:
    rows = board_rows(m, index)
    if len(rows) > MAX_SHOTS:
        raise FramesError(f"{m.start:.2f}–{m.end:.2f} s holds {len(rows)} shots; one board shows up to {MAX_SHOTS}. "
                          "Narrow the span.")
    boundaries = {r.first for r in rows}
    change = masked_change(m, boundaries)
    picks: list[list[int]] = []
    for r in rows:
        if r.kind != "shot" or r.end - r.first <= 2:
            picks.append(list(range(r.first, r.end))[:2])
        else:
            picks.append([r.first, *pick_middles(change, r.first, r.end), r.end - 1])
    tiles = grab(m, [i for p in picks for i in p], BOARD_TILE)
    tw, th = fit(shown_aspect(m.src, m.crop_x), BOARD_TILE)
    width = PAD + (MAX_MIDDLE + 2) * (tw + PAD)

    canvas = Canvas(width)
    where = f"index {index.directory}" if index else "no index; shots detected"
    crop = f" · 9:16 crop at x={m.crop_x:g}" if m.crop_x is not None else ""
    canvas.text(f"{m.src.path.name} · {m.start:.2f}–{m.end:.2f} s · scene board · {where}{crop}", size=17, fill=ACCENT)
    canvas.text("Tiles: first frame, in-between frames spaced by how much the picture changes, last frame. "
                "Stamps are source seconds. Strip: change per frame (cuts and flashes zeroed), ▼ = tile shown.",
                size=13, fill=DIM)
    canvas.gap()
    for r, frames in zip(rows, picks):
        flashes = [(i, m.labels[i]) for i in range(r.first, r.end) if m.labels[i] != "content"]
        title = r.title
        if flashes and r.kind == "shot":
            runs = runs_of([m.labels[i] != "content" for i in range(r.first, r.end)])
            title += " · " + ", ".join(f"{m.labels[r.first + a]} frames {m.times[r.first + a]:.2f} ({b - a}f)"
                                       for a, b in runs)
        canvas.text(title, size=15, fill=ACCENT)
        strip_w = len(frames) * (tw + PAD) - PAD
        row_img = Image.new("RGB", (width, th + 4 + 28), BG)
        for k, i in enumerate(frames):
            note = "first" if i == r.first else "last" if i == r.end - 1 else None
            row_img.paste(stamp(tiles[i], f"{m.times[i]:.2f}", top=note, flag=flash_name(m.labels[i])),
                          (PAD + k * (tw + PAD), 0))
        values = change[r.first:r.end]
        strip = change_strip(strip_w, 24, values, marks=[i - r.first for i in frames],
                             flashes=[(i - r.first, label) for i, label in flashes])
        row_img.paste(strip, (PAD, th + 4))
        canvas.image(row_img)
        if r.words:
            canvas.text("words: " + "   ".join(f"{w.start:.2f} {w.text}" for w in r.words), size=14)
        for cue in r.cues:
            canvas.text(f"subtitle {cue.start:.2f}–{cue.end:.2f}: {cue.text}", size=14, fill=DIM)
        canvas.rule()
    canvas.save(out)
    return f"{out}: {len(rows)} rows, {sum(len(p) for p in picks)} tiles ({where})"


# ---- mode 2: shot detail ----

@dataclass(frozen=True)
class DenseTile:
    frame: int            # frame number in the measured span
    until: float          # source time the tile stands for until (the next tile's time, or OUT)
    samples: int          # sample times merged into this tile


def dense_tiles(m: Measured, fps: float, boundaries: set[int]) -> list[DenseTile]:
    times = [m.start + k / fps for k in range(math.ceil((m.end - m.start) * fps - 1e-9))]
    frames: list[int] = []
    for t in times:
        i = m.index_at(t)
        if not frames or i != frames[-1]:
            frames.append(i)
    small = m.small.astype(np.float32)
    runs: list[list[int]] = []
    for i in frames:
        if runs:
            head = runs[-1][0]
            crosses = any(head < b <= i for b in boundaries) or any(
                m.labels[j] != "content" for j in range(head, i + 1))
            if not crosses and float(np.abs(small[i] - small[head]).mean()) < HOLD_DIFF:
                runs[-1].append(i)
                continue
        runs.append([i])
    tiles = []
    for k, run in enumerate(runs):
        until = m.times[runs[k + 1][0]] if k + 1 < len(runs) else m.end
        tiles.append(DenseTile(run[0], until, len(run)))
    return tiles


def boundaries_of(m: Measured, index: Index | None) -> dict[int, str]:
    """Frame numbers where a new shot starts inside the span, with a label."""
    if index is None:
        out: dict[int, str] = {}
        number = 0
        for seg in m.segments:
            number += seg.kind == "shot"
            if seg.start > 0:
                out[seg.start] = f"shot {number}" if seg.kind == "shot" else seg.kind
        return out
    half = 0.5 / m.src.fps
    return {m.index_at(s.start): f"shot {s.id}" for s in index.shots
            if m.times[0] + half < s.start < m.times[-1] + half}


def shot_detail(m: Measured, index: Index | None, fps: float, out: Path) -> str:
    cuts = boundaries_of(m, index)
    tiles = dense_tiles(m, fps, set(cuts))
    if len(tiles) > MAX_TILES:
        raise FramesError(f"{m.start:.2f}–{m.end:.2f} s at {fps:g} fps makes {len(tiles)} tiles after merging holds; "
                          f"one sheet shows up to {MAX_TILES}. Narrow the span or lower --fps.")
    images = grab(m, [t.frame for t in tiles], DENSE_TILE)
    tw, th = fit(shown_aspect(m.src, m.crop_x), DENSE_TILE)
    columns = max(1, min(len(tiles), (DENSE_WIDTH - PAD) // (tw + PAD)))
    width = PAD + columns * (tw + PAD)
    change = masked_change(m, set(cuts))
    moving = stretches(change, m.src.fps)

    canvas = Canvas(width)
    where = f"index {index.directory}" if index else "no index; shots detected"
    crop = f" · 9:16 crop at x={m.crop_x:g}" if m.crop_x is not None else ""
    canvas.text(f"{m.src.path.name} · {m.start:.2f}–{m.end:.2f} s · detail at {fps:g} fps · {where}{crop}",
                size=17, fill=ACCENT)
    canvas.text("Stamps are source seconds. Tiles that barely differ are merged and marked held. Plot: change per "
                "frame (cuts and flashes zeroed); shaded where frames keep changing past the move line. Any pixel "
                "change counts: pans, zooms, shake and lighting register as much as action.", size=13, fill=DIM)
    canvas.gap()
    grid_rows = math.ceil(len(tiles) / columns)
    grid = Image.new("RGB", (width, grid_rows * (th + PAD)), BG)
    draw = ImageDraw.Draw(grid)
    for k, tile in enumerate(tiles):
        x, y = PAD + (k % columns) * (tw + PAD), (k // columns) * (th + PAD)
        held = tile.until - m.times[tile.frame]
        top = f"#{k + 1}" + (f" held {held:.1f} s" if tile.samples > 1 else "")
        cut = next((label for b, label in cuts.items() if b == tile.frame), None)
        grid.paste(stamp(images[tile.frame], f"{m.times[tile.frame]:.2f}", top=top,
                         flag=flash_name(m.labels[tile.frame]) or cut), (x, y))
        if cut is not None:
            draw.line((x - PAD // 2, y, x - PAD // 2, y + th), fill=CUT_RED, width=3)
    canvas.image(grid)
    canvas.gap()
    canvas.image(time_axis(m, width, change, cuts, tiles, moving, index))
    canvas.gap()

    events: list[tuple[float, str]] = []
    for b, name in cuts.items():
        events.append((m.times[b], f"cut {m.times[b]:.2f}: {name} begins"))
    for s in moving:
        if s.start == s.last:
            text = f"change {m.times[s.start]:.2f} ({s.peak_change:g}), one frame"
        else:
            text = (f"change starts {m.times[s.start]:.2f} · peaks {m.times[s.peak]:.2f} ({s.peak_change:g}) · "
                    f"settles {m.times[s.last]:.2f} ({m.times[s.last] - m.times[s.start]:.2f} s)")
        events.append((m.times[s.start], text))
    for a, b in runs_of([label != "content" for label in m.labels]):
        events.append((m.times[a], f"{m.labels[a]} frames {m.times[a]:.2f}–{m.times[b - 1] + 1 / m.src.fps:.2f} "
                                   f"({b - a}f)"))
    canvas.text("\n".join(text for _, text in sorted(events)) or "no change past the move line, no cuts, no flashes",
                size=15)
    if index is not None:
        words = [w for w in index.words if m.start <= w.start < m.end]
        cues = [c for c in index.cues if c.start < m.end and c.end > m.start]
        if words:
            canvas.text("words: " + "   ".join(f"{w.start:.2f} {w.text}" for w in words), size=14)
        for cue in cues:
            canvas.text(f"subtitle {cue.start:.2f}–{cue.end:.2f}: {cue.text}", size=14, fill=DIM)
    canvas.save(out)
    return f"{out}: {len(tiles)} tiles, {len(moving)} stretches of change, {len(cuts)} cuts"


def time_axis(m: Measured, width: int, change: Sequence[float], cuts: dict[int, str],
              tiles: Sequence[DenseTile], moving: Sequence[Stretch], index: Index | None) -> Image.Image:
    """Change per frame on a source-time axis, with cuts, flashes, tile positions and words."""
    left, right = PAD + 40, width - PAD
    words = [w for w in index.words if m.start <= w.start < m.end] if index is not None else []
    plot_h, word_h = 90, 40 if words else 0
    img = Image.new("RGB", (width, 26 + plot_h + 22 + word_h), BG)
    draw = ImageDraw.Draw(img)
    span = m.end - m.start

    def x_of(t: float) -> float:
        return left + (t - m.start) / span * (right - left)

    top = 26
    draw.rectangle((left, top, right, top + plot_h), fill="#202020")
    dt = 1 / m.src.fps
    for s in moving:
        draw.rectangle((x_of(m.times[s.start]), top, x_of(m.times[s.last] + dt), top + plot_h), fill="#2c3340")
    for i, v in enumerate(change):
        h = min(1.0, v / CHANGE_SCALE) * plot_h
        if h >= 1:
            draw.rectangle((x_of(m.times[i]), top + plot_h - h, max(x_of(m.times[i]), x_of(m.times[i] + dt) - 1),
                            top + plot_h), fill="#3d6a99")
    y_active = top + plot_h - ACTIVE / CHANGE_SCALE * plot_h
    draw.line((left, y_active, right, y_active), fill="#555555")
    f = font(12)
    draw.text((PAD, y_active - 7), "move", font=f, fill=DIM)
    for i, label in enumerate(m.labels):
        if label != "content":
            draw.rectangle((x_of(m.times[i]), top, x_of(m.times[i] + dt), top + plot_h),
                           fill="#ffffff" if label == "white" else "#000000", outline="#888888")
    for b, name in cuts.items():
        x = x_of(m.times[b])
        draw.line((x, top - 4, x, top + plot_h), fill=CUT_RED, width=2)
        draw.text((x + 3, top - 16), name, font=f, fill=CUT_RED)
    for k, tile in enumerate(tiles):
        x = x_of(m.times[tile.frame])
        draw.polygon([(x - 4, top + plot_h + 1), (x + 4, top + plot_h + 1), (x, top + plot_h + 7)], fill=ACCENT)
        if k % max(1, len(tiles) // 30) == 0:
            draw.text((x - 6, top + plot_h + 7), str(k + 1), font=f, fill=ACCENT)
    step = next(s for s in (0.25, 0.5, 1, 2, 5, 10, 30, 60) if span / s <= 16)
    t = math.ceil(m.start / step) * step
    while t <= m.end + 1e-9:
        draw.line((x_of(t), top, x_of(t), top + 4), fill=DIM)
        draw.text((min(x_of(t) - 18, right - 44), 4), f"{t:.2f}", font=f, fill=DIM)
        t += step
    if words:
        y0 = top + plot_h + 24
        ends = [-1e9, -1e9]
        for w in words:
            x = x_of(w.start)
            lane = 0 if x > ends[0] else 1 if x > ends[1] else 0
            draw.line((x, top + plot_h, x, y0 + lane * 16), fill="#c792ea")
            draw.text((x + 2, y0 + lane * 16), w.text, font=font(13), fill="#c792ea")
            ends[lane] = x + font(13).getlength(w.text) + 6
    return img


# ---- mode 3: cut pairs ----

@dataclass(frozen=True)
class SongMap:
    path: Path
    of_song: bool                 # a map of the song file itself, not of a render
    beats: list[float]            # timeline seconds
    words: list[analyze_edit.Word]
    drops: list[float]
    lyrics: lyric_reference.Reference | None = None
    word_source: str = "recognized"


def _song_time(edl: Edl, t: float) -> float | None:
    """Song-file seconds to edit timeline seconds. Moments before the first piece or after the last map to
    negative or past-the-end times, so a lyric line already under way at the start still shows; moments in a
    stretch the splices cut out are None."""
    pieces = song_pieces(edl.song)
    speed = edl.song.speed
    if t < pieces[0].src_in:
        return (t - pieces[0].src_in) / speed
    start = 0.0
    for piece in pieces:
        if t <= piece.src_out:
            return start + (t - piece.src_in) / speed if t >= piece.src_in else None
        start += (piece.src_out - piece.src_in) / speed
    last = pieces[-1]
    return start + (t - last.src_out) / speed


def load_song_map(file: Path, edl: Edl, edl_dir: Path) -> SongMap | None:
    """An analyze_edit.py timeline.json of the song file or of a render of this EDL, on the edit's timeline."""
    try:
        data = json.loads(file.read_text())
        media = Path(str(data["media"]["path"])).resolve()
        audio = data["audio"]
        if audio is None:
            return None
        if media == render.resolve(edl_dir, edl.song.source):
            of_song = True
        elif media.parent == edl_dir and data["media"].get("video") is not None:
            of_song = False
        else:
            return None

        def to_tl(t: float) -> float | None:
            return _song_time(edl, t) if of_song else t

        def played(times: Sequence[float]) -> list[float]:
            return [tl for tl in map(to_tl, times) if tl is not None]
        words = [analyze_edit.Word(start, end, float(w.get("probability", 0)), w.get("text")) for w in data.get("words", [])
                 if (start := to_tl(float(w["start"]))) is not None and (end := to_tl(float(w["end"]))) is not None]
        ref = data.get("lyrics")
        reference = (lyric_reference.Reference(str(ref["path"]), str(ref["source"]),
                     [lyric_reference.Line(float(line["start"]) if line.get("start") is not None else None,
                                           str(line["text"])) for line in ref["lines"]]) if ref else None)
        return SongMap(file, of_song, played([float(b) for b in audio["beats"]]), words,
                       played([float(d["time"]) for d in audio.get("drops", [])]), reference)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None



def with_alignment(song_map: SongMap | None, file: Path, edl: Edl, edl_dir: Path) -> SongMap:
    """Explicitly overlay reference-conditioned words; never rewrite raw Whisper caches."""
    import lyricsync
    try:
        data = json.loads(file.read_text())
        if Path(data["media"]).resolve() != render.resolve(edl_dir, edl.song.source):
            raise ValueError("alignment must be of this EDL's song file, not a different mix/render")
        aligned = [lyricsync.Word(**w) for w in data["words"]]
        lines = [str(line["text"]) for line in data["reference"]["lines"]]
        lyricsync.validate(lines, aligned, float(data["duration"]))
        words = [analyze_edit.Word(start, end, w.ctc_score, w.text) for w in aligned
                 if (start := _song_time(edl, w.start)) is not None
                 and (end := _song_time(edl, w.end)) is not None and end > start]
        base = song_map if song_map is not None else SongMap(file, True, [], [], [])
        if not base.of_song:
            raise ValueError("use a song map, not a render map, with --alignment")
        return replace(base, words=words, word_source="aligned reference")
    except (OSError, KeyError, TypeError, ValueError) as e:
        raise FramesError(f"cannot use alignment {file}: {e}") from None


def find_song_map(edl: Edl, edl_dir: Path) -> SongMap | None:
    """A map in the EDL's folder or its parent, one level down at most; a map of the song beats one of a render."""
    found: list[SongMap] = []
    for base in (edl_dir, edl_dir.parent):
        for file in sorted([base / "timeline.json", *base.glob("*/timeline.json")]):
            if file.is_file() and (song_map := load_song_map(file, edl, edl_dir)) is not None:
                found.append(song_map)
    return max(found, key=lambda sm: sm.of_song, default=None)


def load_lyrics(file: Path, edl: Edl) -> list[tuple[float, str]]:
    """LRC lines as (timeline seconds, text); word tags such as <00:12.34> are dropped."""
    try:
        ref = lyric_reference.read(file)
    except (OSError, UnicodeError) as e:
        raise FramesError(f"cannot read {file}: {e}") from None
    return reference_lines(ref, edl, True)


def reference_lines(ref: lyric_reference.Reference, edl: Edl, of_song: bool) -> list[tuple[float, str]]:
    return sorted((at, line.text) for line in ref.lines if line.start is not None
                  and (at := _song_time(edl, line.start) if of_song else line.start) is not None)


def parse_cuts(spec: str | None, clips: int) -> list[int]:
    if clips < 2 and spec is None:
        raise FramesError("the EDL has one clip and so no cuts")
    if spec is None:
        return list(range(2, clips + 1))
    picked: list[int] = []
    for part in spec.split(","):
        bounds = part.strip().split("-")
        try:
            lo, hi = int(bounds[0]), int(bounds[-1])
        except ValueError:
            raise FramesError(f"--cut takes N, A-B or a comma list, got {spec!r}") from None
        if len(bounds) > 2 or not 1 <= lo <= hi <= clips:
            raise FramesError(f"--cut {part.strip()}: cuts are numbered 1 to {clips} (cut N is where clip #N enters)")
        picked += range(lo, hi + 1)
    return sorted(set(picked))


@dataclass(frozen=True)
class PairFrame:
    image: Image.Image
    timeline: float
    source: float


def clip_frames_rendered(edl: Edl, edl_dir: Path, probes: dict[Path, render.Probe], index: int, first: int,
                         count: int, crop_x: float | None, work: Path) -> list[PairFrame]:
    """Output frames [first, first + count) of clip `index`, through render.py's segment pipeline."""
    clip = edl.clips[index]
    fps = edl.output.fps
    file = render.resolve(edl_dir, clip.source)
    out_aspect = edl.output.width / edl.output.height
    long_side = PAIR_TILE * 3 if crop_x is not None else PAIR_TILE
    w, h = fit(out_aspect, long_side)
    seg = render.Segment(count, clip, file, probes[file], first=first)
    target = work / f"clip{index}_{first}.mp4"
    try:
        render.run(render.segment_command(seg, Output(w, h, fps), target), f"rendering clip #{index + 1}", False)
    except render.RenderError as e:
        raise FramesError(str(e)) from None
    small = fit(VERTICAL[0] / VERTICAL[1] if crop_x is not None else out_aspect, PAIR_TILE)
    vf = [crop_filter(crop_x, 1.0, fps)] if crop_x is not None else []
    vf += [f"scale={small[0]}:{small[1]}:flags=area"]
    proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(target), "-vf", ",".join(vf),
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    step = small[0] * small[1] * 3
    if proc.returncode != 0 or len(proc.stdout) != step * count:
        raise FramesError(f"cannot read rendered frames of clip #{index + 1}: {proc.stderr.decode(errors='replace')[-300:]}")
    start = clip_frames(clip, fps)[0]
    return [PairFrame(Image.frombytes("RGB", small, proc.stdout[k * step:(k + 1) * step]),
                      (start + first + k) / fps, clip.src_in + source_offset(clip, fps, first + k))
            for k in range(count)]


def _transition(clip: Clip) -> str:
    t = clip.transition
    return {"cut": "hard cut", "flash": f"{t.color} flash, {t.frames} frames",
            "crossfade": f"crossfade, {t.frames} frames"}[t.type]


def _rel(values: Sequence[float], at: float) -> list[float]:
    return [v - at for v in values if abs(v - at) <= CUT_REACH]


def cut_pairs(edl_path: Path, out: Path, cut_spec: str | None, song_map_file: Path | None, lyrics_file: Path | None,
              crop_x: float | None, *, show_intent: bool = False, lyrics_source: str | None = None,
              alignment_file: Path | None = None) -> str:
    try:
        edl = load_edl(edl_path)
    except EdlError as e:
        raise FramesError(f"{edl_path}: " + "; ".join(e.problems)) from None
    except OSError as e:
        raise FramesError(f"cannot read {edl_path}: {e}") from None
    edl_dir = edl_path.resolve().parent
    probes, problems = render.check_sources(edl, edl_dir)
    if problems:
        raise FramesError(f"{edl_path}: " + "; ".join(problems))
    if crop_x is not None and edl.output.width * VERTICAL[1] <= edl.output.height * VERTICAL[0] * 1.01:
        raise FramesError("--crop-x previews a 9:16 cut of a wider edit; this EDL's output is already vertical")
    fps = edl.output.fps
    cuts = parse_cuts(cut_spec, len(edl.clips))
    if song_map_file is not None:
        song_map = load_song_map(song_map_file, edl, edl_dir)
        if song_map is None:
            raise FramesError(f"{song_map_file} is not an analyze_edit.py map of this EDL's song or of its render")
    else:
        song_map = find_song_map(edl, edl_dir)
    if alignment_file is not None:
        song_map = with_alignment(song_map, alignment_file, edl, edl_dir)
    reference = song_map.lyrics if song_map is not None else None
    if lyrics_file is not None:
        try:
            reference = lyric_reference.read(lyrics_file, lyrics_source)
        except (OSError, UnicodeError) as e:
            raise FramesError(f"cannot read {lyrics_file}: {e}") from None
    lyrics = (reference_lines(reference, edl, lyrics_file is not None or song_map is None or song_map.of_song)
              if reference is not None else [])

    tw, th = fit(VERTICAL[0] / VERTICAL[1] if crop_x is not None else edl.output.width / edl.output.height, PAIR_TILE)
    divider = 46
    width = max(PAD * 2 + 2 * PAIR_FRAMES * (tw + PAD) + divider, 900)
    canvas = Canvas(width)
    crop = f" · 9:16 crop at x={crop_x:g}" if crop_x is not None else ""
    title = (edl.title or edl_path.name) if show_intent else edl_path.name
    canvas.text(f"{title} · cut pairs {cuts[0]}–{cuts[-1]}{crop}", size=17, fill=ACCENT)
    source = (f"song map {song_map.path}" if song_map else "no song map found (run analyze_edit.py on the song, "
              "or pass --song-map)")
    canvas.text(f"Left: the outgoing clip's last {PAIR_FRAMES} frames. Right: the incoming clip's first "
                f"{PAIR_FRAMES}. Stamps: timeline t / source s. Frames are the clip picture rendered by render.py; "
                f"flash and crossfade layers are listed, not drawn. Strip: ±{CUT_REACH:g} s of song around the cut, "
                f"beats as ticks, sung words as bars. {source}.", size=13, fill=DIM)
    canvas.text("Intention notes shown." if show_intent else "Intention notes hidden; use --show-intent to compare afterward.",
                size=13, fill=DIM)
    canvas.text("Aligned reference words use uncalibrated CTC scores, not recognition confidence. Timing needs review."
                if song_map is not None and song_map.word_source == "aligned reference" else
                "Cue proximity describes timing, not editing quality. Recognized words are unverified Whisper text; "
                "p is recognition confidence, not timing accuracy. Legacy maps may have no text.", size=13, fill=DIM)
    if reference is not None:
        canvas.text(f"Published lyric reference: {reference.source}. {reference.alignment}. "
                    "Untimed lines have unknown placement and are not attached to cuts.", size=13, fill=DIM)
    canvas.gap()
    with tempfile.TemporaryDirectory(prefix="frames_") as tmp:
        work = Path(tmp)
        for n in cuts:
            canvas.image(pair_row(edl, edl_dir, probes, n, crop_x, work, width, (tw, th), divider, show_intent=show_intent))
            canvas.image(cut_strip(edl, n, song_map, width))
            canvas.text(cut_notes(edl, n, song_map, lyrics, show_intent=show_intent), size=14)
            if show_intent:
                clip = edl.clips[n - 1]
                a, b = clip_frames(clip, fps)
                for point in clip.sync_points:
                    frame = round(point.t * fps)
                    if frame == 0:
                        continue
                    at = (a + frame) / fps
                    canvas.text(f'In-shot action: {point.label} · picture {at:.3f} s', size=14, fill=ACCENT)
                    first, end = max(0, frame - 1), min(b - a, frame + 2)
                    pictures = clip_frames_rendered(edl, edl_dir, probes, n - 1, first, end - first, crop_x, work)
                    row = Image.new("RGB", (width, th + 4), BG)
                    for k, picture in enumerate(pictures):
                        row.paste(stamp(picture.image, f"t {picture.timeline:.2f} / s {picture.source:.2f}"),
                                  (PAD + k * (tw + PAD), 0))
                    canvas.image(row)
                    canvas.image(cut_strip(edl, n, song_map, width, at=at))
                    canvas.text(cut_notes(edl, n, song_map, lyrics, at=at), size=14)
            canvas.rule()
    canvas.save(out)
    return f"{out}: {len(cuts)} cuts" + (f", song map {song_map.path}" if song_map else ", no song map")


def pair_row(edl: Edl, edl_dir: Path, probes: dict[Path, render.Probe], n: int, crop_x: float | None, work: Path,
             width: int, tile: tuple[int, int], divider: int, *, show_intent: bool = False) -> Image.Image:
    fps = edl.output.fps
    tw, th = tile
    incoming = edl.clips[n - 1]
    a_in, b_in = clip_frames(incoming, fps)
    right = clip_frames_rendered(edl, edl_dir, probes, n - 1, 0, min(PAIR_FRAMES, b_in - a_in), crop_x, work)
    left: list[PairFrame] = []
    gap = a_in
    if n >= 2:
        outgoing = edl.clips[n - 2]
        a_out, b_out = clip_frames(outgoing, fps)
        k = min(PAIR_FRAMES, b_out - a_out)
        left = clip_frames_rendered(edl, edl_dir, probes, n - 2, b_out - a_out - k, k, crop_x, work)
        gap = a_in - b_out
    head = 40
    img = Image.new("RGB", (width, head + th + 4), BG)
    draw = ImageDraw.Draw(img)
    f = font(15)
    out_name = f"#{n - 1} {Path(edl.clips[n - 2].source).name}" if n >= 2 else "start of the edit"
    draw.text((PAD, 2), f"cut {n} · {a_in / fps:.3f} s (frame {a_in}) · {_transition(incoming)}", font=font(16),
              fill=ACCENT)
    draw.text((PAD, 21), f"out: {out_name}", font=f, fill=DIM)
    rx = PAD + PAIR_FRAMES * (tw + PAD) + divider
    draw.text((rx, 21), f"in: #{n} {Path(incoming.source).name}" +
              (f" · declared sync {incoming.sync}" if show_intent and incoming.sync else ""),
              font=f, fill=DIM)
    for k, fr in enumerate(left):
        x = PAD + (PAIR_FRAMES - len(left) + k) * (tw + PAD)
        img.paste(stamp(fr.image, f"t {fr.timeline:.2f} / s {fr.source:.2f}"), (x, head))
    cx = PAD + PAIR_FRAMES * (tw + PAD) + divider // 2 - PAD // 2
    draw.line((cx, head - 4, cx, head + th + 2), fill=CUT_RED, width=3)
    if gap > 0:
        draw.text((cx - 20, head + th // 2 - 8), f"black\n{gap / fps:.2f}s", font=font(12), fill=FG)
    t = incoming.transition
    for k, fr in enumerate(right):
        note = None
        if t.type == "flash" and k < t.frames:
            note = f"under {t.color} flash"
        elif t.type == "crossfade" and k < t.frames:
            note = f"fading in {k + 1}/{t.frames}"
        img.paste(stamp(fr.image, f"t {fr.timeline:.2f} / s {fr.source:.2f}", top=note), (rx + k * (tw + PAD), head))
    return img


def cut_strip(edl: Edl, n: int, song_map: SongMap | None, width: int, *, at: float | None = None) -> Image.Image:
    """±CUT_REACH seconds of song around cut n: beats, sung words and the frames shown in the row."""
    fps = edl.output.fps
    if at is None:
        at = clip_frames(edl.clips[n - 1], fps)[0] / fps
    center = at
    left, right = PAD + 40, width - PAD - 40
    img = Image.new("RGB", (width, 36), BG)
    draw = ImageDraw.Draw(img)

    def x_of(t: float) -> float:
        return left + (t - center + CUT_REACH) / (2 * CUT_REACH) * (right - left)

    draw.rectangle((left, 4, right, 32), fill="#202020")
    f = font(11)
    draw.text((PAD, 12), f"−{CUT_REACH:g} s", font=f, fill=DIM)
    draw.text((right + 6, 12), f"+{CUT_REACH:g} s", font=f, fill=DIM)
    for k in range(-PAIR_FRAMES, PAIR_FRAMES):
        draw.rectangle((x_of(at + k / fps) + 1, 28, x_of(at + (k + 1) / fps) - 1, 32), fill="#777777")
    if song_map is not None:
        shade = ("#c792ea", "#9d6fc0")
        for k, word in enumerate(song_map.words):
            s, e = word.start, word.end
            if e >= at - CUT_REACH and s <= at + CUT_REACH:
                draw.rectangle((max(left, x_of(s)) + 1, 18, min(right, x_of(e)) - 1, 25), fill=shade[k % 2])
        for b in _rel(song_map.beats, at):
            draw.line((x_of(at + b), 5, x_of(at + b), 16), fill="#9fd0ff", width=2)
        for d in _rel(song_map.drops, at):
            draw.text((x_of(at + d) + 3, 4), "DROP", font=f, fill=ACCENT)
    draw.line((x_of(at), 0, x_of(at), 35), fill=CUT_RED, width=2)
    return img


def cut_notes(edl: Edl, n: int, song_map: SongMap | None, lyrics: Sequence[tuple[float, str]], *,
              show_intent: bool = False, at: float | None = None) -> str:
    fps = edl.output.fps
    clip = edl.clips[n - 1]
    if at is None:
        at = clip_frames(clip, fps)[0] / fps
    parts = []
    if song_map is not None:
        beats = _rel(song_map.beats, at)
        starts = _rel([w.start for w in song_map.words], at)
        # A word end only matters where silence follows it: the end of a phrase or a breath.
        gaps = _rel([w.end for w, following in zip(song_map.words, [*song_map.words[1:],
                                                                  analyze_edit.Word(math.inf, math.inf, 0)])
                     if following.start - w.end > 0.05], at)
        parts.append("beats " + (" ".join(f"{b:+.2f}" for b in beats) if beats else "none") + " s")
        parts.append("word starts " + (" ".join(f"{w:+.2f}" for w in starts) if starts else "none") + " s")
        if gaps:
            parts.append("singing pauses at " + " ".join(f"{w:+.2f}" for w in gaps) + " s")
        for w in song_map.words:
            if w.end >= at - CUT_REACH and w.start <= at + CUT_REACH:
                text = json.dumps(w.text, ensure_ascii=False) if w.text is not None else "[text unknown]"
                label = "CTC" if song_map.word_source == "aligned reference" else "p"
                parts.append(f"{song_map.word_source} {text} {w.start-at:+.2f}–{w.end-at:+.2f} s ({label}={w.probability:.2f})")
    sung = [line for line in lyrics if line[0] <= at]
    if sung and sung[-1][1]:
        parts.append(f"reference lyric at the cut (began {sung[-1][0] - at:+.2f} s): {sung[-1][1]}")
    for t, text in lyrics:
        if text and at < t <= at + CUT_REACH:
            parts.append(f"next reference lyric {t - at:+.2f} s: {text}")
    for k, layer in enumerate(edl.audio):
        start = layer.timeline_in
        end = None if layer.src_out is None else start + layer.src_out - layer.src_in  # None: to the file's end
        if start - CUT_REACH <= at and (end is None or at <= end + CUT_REACH):
            span = f"{start:.2f}–{end:.2f} s" if end is not None else f"from {start:.2f} s"
            parts.append(f"audio layer {k + 1} {span}" + (f": {layer.why}" if show_intent and layer.why else ""))
    if show_intent and clip.sync:
        parts.append(f"declared sync: {clip.sync}")
    if show_intent:
        parts.extend(render.sync_point_notes(clip, fps))
    if show_intent and clip.why:
        parts.append(f"why: {clip.why}")
    return " · ".join(parts) if parts else "no song map, lyrics or notes for this cut"


# ---- driver ----

def unit_interval(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be from 0 to 1")
    return value


def positive(text: str) -> float:
    value = float(text)
    if not value > 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a positive number")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("args", nargs="*", metavar="ARG", help="SOURCE IN OUT OUT.jpg, or OUT.jpg with --edl")
    ap.add_argument("--dense", action="store_true", help="shot detail: frames every 1/--fps seconds")
    ap.add_argument("--fps", type=positive, default=6.0, help="dense sampling rate (default 6)")
    ap.add_argument("--edl", type=Path, help="cut pairs of this EDL")
    ap.add_argument("--cut", help="cut pairs: N, A-B or a comma list; cut N is where clip #N enters (default: all)")
    ap.add_argument("--song-map", type=Path, help="cut pairs: analyze_edit.py timeline.json of the song or the render")
    ap.add_argument("--alignment", type=Path, help="cut pairs: lyricsync alignment.json of the song")
    ap.add_argument("--lyrics", type=Path, help="cut pairs: an LRC file of the song")
    ap.add_argument("--lyrics-source", help="attribution for --lyrics (defaults to its local path)")
    ap.add_argument("--show-intent", action="store_true", help="cut pairs: show why notes and declared sync cues")
    ap.add_argument("--crop-x", type=unit_interval, help="preview a 9:16 crop centred at this fraction of the width")
    ap.add_argument("--index", type=Path, help="index directory (default: out/index/NAME)")
    ap.add_argument("--no-index", action="store_true", help="ignore any index and detect shots")
    a = ap.parse_args(argv)
    if a.lyrics_source and a.lyrics is None:
        ap.error("--lyrics-source needs --lyrics")
    try:
        if a.edl is not None:
            if len(a.args) != 1 or a.dense:
                ap.error("--edl takes one positional argument, OUT.jpg, and no --dense")
            print(cut_pairs(a.edl, Path(a.args[0]), a.cut, a.song_map, a.lyrics, a.crop_x,
                            show_intent=a.show_intent, lyrics_source=a.lyrics_source, alignment_file=a.alignment))
            return 0
        if len(a.args) != 4:
            ap.error("expected SOURCE IN OUT OUT.jpg")
        if a.cut or a.song_map or a.lyrics or a.show_intent or a.alignment:
            ap.error("--cut, --song-map, --lyrics and --show-intent go with --edl")
        source, out = Path(a.args[0]), Path(a.args[3])
        try:
            start, end = parse_time(a.args[1]), parse_time(a.args[2])
        except argparse.ArgumentTypeError as e:
            ap.error(str(e))
        src = open_source(source)
        index: Index | None = None
        if not a.no_index:
            directory = a.index if a.index is not None else find_index(source)
            if directory is not None:
                index = load_index(directory)
        m = measure(src, start, end, a.crop_x)
        print(shot_detail(m, index, a.fps, out) if a.dense else scene_board(m, index, out))
    except FramesError as e:
        print(f"frames.py: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
