"""Render an edit decision list to a frame-accurate MP4.

usage: python render.py EDL.json OUT.mp4|OUT.wav [--check] [--jobs N] [--crf N] [--preset P] [--verbose]

Each clip is normalized (seek, speed, reverse, crop, color) into a cached segment of
exactly its frame count, and each audio layer with stem "voice" into a cached voice stem.
One ffmpeg filtergraph then lays the segments end to end and composites crossfades,
overlays, flashes, captions, the ending fade and the audio mix. Writes OUT.mp4, proof.jpg
and edl_table.md, and keeps segments and stems in render_cache/ next to OUT.mp4 so
re-renders only redo changed clips and lines. OUT.wav renders the audio mix alone, with
edl_table.md, for quick listening while shaping the song. The schema is documented in edl.py.
"""
import argparse
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from edl import (BASS_HZ, DUCK_RAMP, ECHO_DELAY, ECHO_FEEDBACK, JOIN, OPEN_HIGHPASS, OPEN_LOWPASS, REVERB_DECAY,
                 TAIL_FADE, Caption, Clip, Color, Edl, EdlError, Output, Tail, check_layer_span, clip_frames,
                 crossfade_handle, join_seconds, layer_end, layer_times, load_edl, output_frames, reveal_frames,
                 song_bass_db, song_compress, song_gain_db, song_highpass, song_lowpass, song_pieces, song_window,
                 source_end, source_offset, span_frames, speed_range, to_frame, track)
import separate
from separate import SeparationError, VoiceSeparator

SAMPLE_RATE = 48000
CACHE_DIR = "render_cache"
CACHE_VERSION = 2
LIMIT = 0.95  # master peak limit, about -0.45 dBFS
# Tag frames instead of passing -colorspace to the encoder, which makes ffmpeg convert untagged input.
TAG_BT709 = "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
SEGMENT_CODEC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-bf", "0", "-pix_fmt", "yuv420p"]
RGBA_TO_YUVA = "scale=out_color_matrix=bt709:out_range=tv,format=yuva420p"
DEFAULT_FONTS = ("Arial Bold.ttf", "DejaVuSans-Bold.ttf", "Helvetica.ttc")
PRESETS = ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow")
Font = ImageFont.FreeTypeFont


class RenderError(RuntimeError):
    pass


# ---- sources ----

@dataclass(frozen=True)
class Probe:
    duration: float | None
    video_index: int | None  # absolute stream index of the picture, None for audio-only files
    sar: float
    fps: float | None
    audio_streams: int
    image: bool
    matrix: str = "bt709"  # colorspace-filter name of the picture's YUV matrix; untagged SD is BT.601


def _ratio(text: object) -> float | None:
    if not isinstance(text, str):
        return None
    num, _, den = text.replace(":", "/").partition("/")
    try:
        value = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def probe(file: Path) -> Probe:
    proc = subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(file)],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        reason = (proc.stderr.strip().splitlines() or ["unknown error"])[-1]
        raise ValueError(f"ffprobe cannot read {file.name}: {reason}")
    info = json.loads(proc.stdout)
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    fmt = info.get("format", {})
    name = str(fmt.get("format_name", ""))
    duration = _ratio(fmt.get("duration")) or (_ratio(video.get("duration")) if video else None)
    space = str(video.get("color_space", "unknown")) if video else "unknown"
    if space in ("smpte170m", "bt470bg") or (space == "unknown" and video and int(video.get("height", 0)) < 720):
        matrix = "bt601-6-525"
    else:
        matrix = "bt2020" if space.startswith("bt2020") else "bt709"
    return Probe(
        duration=duration,
        video_index=int(video["index"]) if video else None,
        sar=(_ratio(video.get("sample_aspect_ratio")) or 1.0) if video else 1.0,
        fps=(_ratio(video.get("avg_frame_rate")) or _ratio(video.get("r_frame_rate"))) if video else None,
        audio_streams=sum(1 for s in streams if s.get("codec_type") == "audio"),
        image=video is not None and (name == "image2" or name.endswith("_pipe")),
        matrix=matrix,
    )


def load_font(name: str | None, size: int, edl_dir: Path) -> Font:
    """A font file path (relative to the EDL), an installed font file name, or the default."""
    if name is not None:
        local = edl_dir / name
        return ImageFont.truetype(str(local) if local.exists() else name, size)
    for candidate in DEFAULT_FONTS:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    font = ImageFont.load_default(size)
    if not isinstance(font, ImageFont.FreeTypeFont):
        raise RenderError("Pillow was built without FreeType; set captions[].style.font")
    return font


def resolve(edl_dir: Path, source: str) -> Path:
    return (edl_dir / source).resolve()


def check_sources(edl: Edl, edl_dir: Path) -> tuple[dict[Path, Probe], list[str]]:
    """Probe every source once and report problems that depend on the files themselves."""
    probes: dict[Path, Probe] = {}
    unreadable: dict[Path, str] = {}
    problems: list[str] = []
    fps = edl.output.fps

    def get(source: str, path: str) -> Probe | None:
        file = resolve(edl_dir, source)
        if file not in probes and file not in unreadable:
            if not file.is_file():
                unreadable[file] = f"file not found: {file}"
            else:
                try:
                    probes[file] = probe(file)
                except ValueError as e:
                    unreadable[file] = str(e)
        if file in unreadable:
            problems.append(f"{path}.source: {unreadable[file]}")
            return None
        return probes[file]

    def overrun(pr: Probe, end: float, source: str) -> str | None:
        tolerance = 1 / pr.fps if pr.fps else 0.01
        if pr.duration is None or end <= pr.duration + tolerance:
            return None
        return f"{end:.3f} s is past the end of {source} ({pr.duration:.3f} s)"

    song = get(edl.song.source, "song")
    if song is not None:
        if song.audio_streams == 0:
            problems.append(f"song.source: {edl.song.source} has no audio stream")
        for k, piece in enumerate(song_pieces(edl.song)):
            # A crossfade out of the piece plays half its length past src_out.
            handle = join_seconds(edl.song, k + 1) / 2 * edl.song.speed
            if note := overrun(song, piece.src_out + handle, edl.song.source):
                extra = f" (with the {join_seconds(edl.song, k + 1):g} s crossfade out of it)" if handle else ""
                problems.append(f"{f'song.pieces[{k}]' if edl.song.pieces else 'song'}.src_out: {note}{extra}")

    for i, clip in enumerate(edl.clips):
        path = f"clips[{i}]"
        pr = get(clip.source, path)
        if pr is None:
            continue
        if pr.video_index is None or pr.image:
            kind = "is a still image (use an overlay)" if pr.image else "has no video stream"
            problems.append(f"{path}.source: {clip.source} {kind}")
            continue
        handle = crossfade_handle(edl, i)
        start, end = clip_frames(clip, fps)
        # Offsets are monotonic, so the furthest source moment is at the first or the last frame.
        reach = clip.src_in + max(source_offset(clip, fps, 0), source_offset(clip, fps, end - start + handle - 1))
        if pr.duration is not None and reach > pr.duration + 1e-6:
            extra = f" (with the next clip's {handle}-frame crossfade)" if handle else ""
            problems.append(f"{path}: shows {clip.source} at {reach:.3f} s{extra}, past its end ({pr.duration:.3f} s)")

    for i, layer in enumerate(edl.audio):
        path = f"audio[{i}]"
        pr = get(layer.source, path)
        if pr is None:
            continue
        if layer.track >= pr.audio_streams:
            problems.append(f"{path}.track: {layer.source} has {pr.audio_streams} audio stream(s), got track {layer.track}")
        if layer.src_out is not None and (note := overrun(pr, layer.src_out, layer.source)):
            problems.append(f"{path}.src_out: {note}")
        elif layer.src_out is None and pr.duration is not None and layer.src_in >= pr.duration:
            problems.append(f"{path}.src_in: {layer.src_in:.3f} s is past the end of {layer.source} ({pr.duration:.3f} s)")

    for i, overlay in enumerate(edl.overlays):
        path = f"overlays[{i}]"
        pr = get(overlay.source, path)
        if pr is None:
            continue
        start, end = span_frames(overlay.timeline_in, overlay.timeline_out, fps)
        if pr.video_index is None:
            problems.append(f"{path}.source: {overlay.source} has no picture")
        elif not pr.image and (note := overrun(pr, overlay.src_in + (end - start) / fps, overlay.source)):
            problems.append(f"{path}: the overlay plays its source for {(end - start) / fps:.3f} s, but {note}")

    for i, caption in enumerate(edl.captions):
        try:
            load_font(caption.style.font, 12, edl_dir)
        except OSError:
            problems.append(f"captions[{i}].style.font: cannot load {caption.style.font!r}")
    return probes, problems


def resolve_layers(edl: Edl, edl_dir: Path, probes: dict[Path, Probe]) -> tuple[Edl, list[str]]:
    """The EDL with every audio layer's src_out filled in (the end of its file when not given), and the
    problems with where the filled-in layers play. Run after check_sources found no problems."""
    layers = list(edl.audio)
    filled = []
    problems: list[str] = []
    for i, layer in enumerate(layers):
        if layer.src_out is not None:
            continue
        duration = probes[resolve(edl_dir, layer.source)].duration
        if duration is None:
            problems.append(f"audio[{i}].src_out: the length of {layer.source} is unknown; give src_out")
            continue
        layers[i] = dataclasses.replace(layer, src_out=duration)
        filled.append(i)
    resolved = dataclasses.replace(edl, audio=tuple(layers))
    for i in filled:
        problems += check_layer_span(resolved, i)
    return resolved, problems


# ---- picture segments ----

@dataclass(frozen=True)
class Segment:
    """Frames [first, first + frames) of a clip, counted from the clip's start (past its end for a
    crossfade handle), rendered to one cached file. No clip means black."""
    frames: int
    clip: Clip | None = None
    source: Path | None = None
    probe: Probe | None = None
    first: int = 0

    def key(self, out: Output) -> str:
        data: dict[str, object] = {"v": CACHE_VERSION, "frames": self.frames, "out": dataclasses.asdict(out)}
        if self.clip is not None and self.source is not None and self.probe is not None:
            stat = self.source.stat()
            start, end = clip_frames(self.clip, out.fps)
            # Placement, reasoning and transition do not change the pixels of the segment.
            look = {k: v for k, v in dataclasses.asdict(self.clip).items()
                    if k not in ("timeline_in", "timeline_out", "src_out", "sync", "why", "transition")}
            data.update(source=str(self.source), size=stat.st_size, mtime=stat.st_mtime_ns, clip=look,
                        length=end - start, first=self.first, probe=dataclasses.asdict(self.probe))
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:20]


def plan_picture(edl: Edl, edl_dir: Path, probes: dict[Path, Probe]) -> tuple[list[Segment], dict[int, Segment]]:
    """Picture track segments in order (clips and black gaps), plus crossfade handles by clip index."""
    fps = edl.output.fps
    picture: list[Segment] = []
    handles: dict[int, Segment] = {}
    cursor = 0
    for i, clip in enumerate(edl.clips):
        start, end = clip_frames(clip, fps)
        if start > cursor:
            picture.append(Segment(start - cursor))
        file = resolve(edl_dir, clip.source)
        picture.append(Segment(end - start, clip, file, probes[file]))
        handle = crossfade_handle(edl, i)
        if handle:
            handles[i] = Segment(handle, clip, file, probes[file], first=end - start)
        cursor = end
    total = output_frames(edl)
    if cursor < total:
        picture.append(Segment(total - cursor))
    return picture, handles


TRANSFORM = ("zoom", "x", "y", "shake")
EASE_EXPR = {"linear": "{u}", "in": "{u}*{u}", "out": "(2*{u}-{u}*{u})", "in_out": "{u}*{u}*(3-2*{u})", "hold": "0"}
SHAKE_Y_RATIO, SHAKE_Y_PHASE = 1.37, 1.3  # the vertical wobble runs off-beat from the horizontal one


def track_expr(points: Sequence[tuple[float, float, str]], base: float, t: str) -> str:
    """ffmpeg expression for a keyframed property at time `t`, matching edl.py's interpolation."""
    if not points:
        return repr(base)
    expr = repr(points[-1][1])
    for (t0, v0, _), (t1, v1, ease) in reversed(list(zip(points, points[1:]))):
        u = f"(({t}-{t0!r})/{t1 - t0!r})"
        expr = f"if(lt({t},{t1!r}),{v0!r}+{v1 - v0!r}*{EASE_EXPR[ease].format(u=u)},{expr})"
    return f"if(lt({t},{points[0][0]!r}),{points[0][1]!r},{expr})"


def framing_filter(clip: Clip, first: int, out: Output, sar: float) -> str:
    """Crop the largest output-shaped window / zoom around (x, y), following keyframes and shake."""
    ar = out.width / out.height / sar  # window aspect in stored source pixels
    crop = clip.crop
    if not any(track(clip, name) for name in TRANSFORM):
        return (f"crop=w='min(iw,ih*{ar!r})/{crop.zoom!r}':h='ow/{ar!r}'"
                f":x='clip({crop.x!r}*iw-ow/2,0,iw-ow)':y='clip({crop.y!r}*ih-oh/2,0,ih-oh)'")
    # perspective maps a sub-pixel source quad onto the frame, so slow moves do not jitter. Its
    # `in` counts frames from 1. Every corner expression recomputes the window into st() slots:
    # 0 time, 1 width, 2 height, 3 shake in output pixels, 4 left, 5 top.
    hz = clip.shake_hz
    prelude = ";".join([
        f"st(0,(in-1+{first})/{out.fps})",
        f"st(1,min(W,H*{ar!r})/({track_expr(track(clip, 'zoom'), crop.zoom, 'ld(0)')}))",
        f"st(2,ld(1)/{ar!r})",
        f"st(3,({track_expr(track(clip, 'shake'), 0.0, 'ld(0)')})*{min(out.width, out.height)})",
        f"st(4,clip(({track_expr(track(clip, 'x'), crop.x, 'ld(0)')})*W-ld(1)/2"
        f"+ld(3)*ld(1)/{out.width}*sin(2*PI*{hz!r}*ld(0)),0,W-ld(1)))",
        f"st(5,clip(({track_expr(track(clip, 'y'), crop.y, 'ld(0)')})*H-ld(2)/2"
        f"+ld(3)*ld(2)/{out.height}*sin(2*PI*{hz * SHAKE_Y_RATIO!r}*ld(0)+{SHAKE_Y_PHASE}),0,H-ld(2)))",
    ])
    corners = {"x0": "ld(4)", "y0": "ld(5)", "x1": "ld(4)+ld(1)", "y1": "ld(5)",
               "x2": "ld(4)", "y2": "ld(5)+ld(2)", "x3": "ld(4)+ld(1)", "y3": "ld(5)+ld(2)"}
    return ("perspective=" + ":".join(f"{name}='{prelude};{value}'" for name, value in corners.items())
            + ":interpolation=cubic:eval=frame")


def _balanced_sum(terms: Sequence[str]) -> str:
    """terms joined with + as a balanced tree of parentheses.

    ffmpeg's expression parser rejects a flat chain of 100 or more terms, which a clip of
    about four seconds already needs. Nesting halves keeps the depth at log2 of the count.
    """
    if not terms:
        raise ValueError("a sum needs at least one term")
    if len(terms) == 1:
        return terms[0]
    mid = len(terms) // 2
    return f"({_balanced_sum(terms[:mid])}+{_balanced_sum(terms[mid:])})"


def _index_set(indices: list[int]) -> str:
    """select expression true for the given frame numbers, as runs."""
    runs: list[list[int]] = []
    for n in sorted(set(indices)):
        if runs and n == runs[-1][1] + 1:
            runs[-1][1] = n
        else:
            runs.append([n, n])
    return _balanced_sum([f"between(n,{a},{b})" if a != b else f"eq(n,{a})" for a, b in runs])


def retime(seg: Segment, out: Output) -> tuple[float, float, list[str]]:
    """Seek time, read length and filters that turn the source into exactly seg.frames output frames.

    Python decides which source frame every output frame shows (from edl.source_offset). ffmpeg
    lays the source on a constant-rate grid, keeps the grid frames used, stamps each with the
    first output frame that shows it and lets fps repeat it until the next one.
    """
    if seg.clip is None or seg.probe is None:
        raise ValueError("a black segment has no source to retime")
    clip, fps = seg.clip, out.fps
    src_fps = seg.probe.fps or 24.0
    offsets = [source_offset(clip, fps, seg.first + j) for j in range(seg.frames)]
    lowest = min(offsets)
    filters = ["setpts=PTS-STARTPTS", f"fps={src_fps!r}"]
    grid = src_fps
    speeds = [v for _, v, _ in track(clip, "speed")] or [clip.speed]
    slowest = min((v for v in speeds if v > 0), default=1.0)
    if clip.slowmo != "repeat":
        # Enough in-between frames that the slowest speed still gets a new one each output frame.
        grid = float(min(8 * src_fps, max(src_fps, math.ceil(fps / slowest))))
        mode = "blend" if clip.slowmo == "blend" else "mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
        filters.append(f"minterpolate=fps={grid!r}:mi_mode={mode}")
    picks = [math.floor((o - lowest) * grid + 1e-6) for o in offsets]
    order = list(dict.fromkeys(picks))  # grid frames in the order they first appear
    first_use = {n: j for j, n in reversed(list(enumerate(picks)))}
    starts = [first_use[n] for n in order]
    # setpts also stamps end-of-stream by evaluating at N = number of kept frames, so the last
    # term puts EOF at the segment's end and fps repeats the final frame up to it.
    steps = [*starts, seg.frames]
    table = _balanced_sum([str(steps[0])] + [f"gte(N,{m})*{steps[m] - steps[m - 1]}" for m in range(1, len(steps))])
    filters.append(f"select='{_index_set(picks)}'")
    if clip.reverse:
        filters.append("reverse")
    # tpad covers a source that runs a frame or two short; render_segment rejects anything worse.
    filters += [f"settb=1/{fps}", f"setpts='{table}'", f"fps={fps}",
                "tpad=stop_mode=clone:stop=2", f"trim=end_frame={seg.frames}"]
    seek = max(0.0, clip.src_in + lowest - 0.5 / src_fps)
    return seek, (max(picks) + 1) / grid + 1.0, filters


def color_filters(color: Color) -> list[str]:
    filters = []
    if (color.brightness, color.contrast, color.saturation) != (0.0, 1.0, 1.0):
        filters.append(f"eq=brightness={color.brightness!r}:contrast={color.contrast!r}:saturation={color.saturation!r}")
    if color.tint is not None and color.tint.amount > 0:
        r, g, b = (int(color.tint.color[k:k + 2], 16) / 255 for k in (1, 3, 5))
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b  # BT.709, limited range below
        target = {"y": 16 + 219 * luma, "u": 128 + 224 * (b - luma) / 1.8556, "v": 128 + 224 * (r - luma) / 1.5748}
        a = color.tint.amount
        # A linear mix toward a color is the same mix in YUV, so no RGB round trip is needed.
        filters.append("lutyuv=" + ":".join(f"{c}='val+({t:.3f}-val)*{a!r}'" for c, t in target.items()))
    return filters


def segment_command(seg: Segment, out: Output, file: Path) -> list[str]:
    head = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y"]
    tail = ["-frames:v", str(seg.frames), *SEGMENT_CODEC, "-f", "mp4", str(file)]
    if seg.clip is None or seg.source is None or seg.probe is None:
        black = f"color=c=black:s={out.width}x{out.height}:r={out.fps},format=yuv420p,{TAG_BT709}"
        return [*head, "-f", "lavfi", "-i", black, *tail]
    clip, pr = seg.clip, seg.probe
    # The seek lands half a source frame early so src_in picks the nearest frame despite rounding.
    seek, read, vf = retime(seg, out)
    vf += [framing_filter(clip, seg.first, out, pr.sar),
           f"scale={out.width}:{out.height}:flags=lanczos:out_range=tv", "setsar=1", "format=yuv420p"]
    if pr.matrix != "bt709":  # the colorspace filter is exact on neutrals; swscale's matrix change is not
        vf.append(f"colorspace=all=bt709:iall={pr.matrix}:fast=1:format=yuv420p")
    vf += [*color_filters(clip.color), TAG_BT709]
    return [*head, "-ss", f"{seek:.6f}", "-t", f"{read:.6f}", "-i", str(seg.source), "-map", f"0:{pr.video_index}",
            "-an", "-sn", "-dn", "-vf", ",".join(vf), *tail]


def count_frames(file: Path) -> int:
    proc = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                           "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(file)],
                          capture_output=True, text=True)
    return int(proc.stdout.strip() or 0)


# ---- captions ----

def wrap(words: list[str], font: Font, width: float) -> list[list[tuple[int, str]]]:
    """Greedy pixel-width wrap; each line holds (word index, text) pieces. Over-long words split by character."""
    lines: list[list[tuple[int, str]]] = []
    line: list[tuple[int, str]] = []
    for index, word in enumerate(words):
        candidate = " ".join([t for _, t in line] + [word])
        if line and font.getlength(candidate) <= width:
            line.append((index, word))
            continue
        if line:
            lines.append(line)
        line = []
        piece = ""
        for ch in word:
            if piece and font.getlength(piece + ch) > width:
                lines.append([(index, piece)])
                piece = ""
            piece += ch
        line = [(index, piece)]
    if line:
        lines.append(line)
    return lines


@dataclass(frozen=True)
class CaptionLine:
    text: str
    x: float
    y: float
    words: tuple[int, ...]  # per character: index of its word in the caption, -1 for spaces
    chars: tuple[int, ...]  # per character: index among the caption's non-space characters, -1 for spaces


def layout_caption(caption: Caption, font: Font, out: Output) -> list[CaptionLine]:
    """Wrap and centre a caption. Lines are laid out for the full text so a typewriter reveal never reflows."""
    style = caption.style
    width = style.max_width * out.width - 2 * round(style.stroke_width * font.size)
    rows: list[list[tuple[int, str]]] = []
    offset = 0
    for paragraph in caption.text.split("\n"):
        words = paragraph.split()
        rows += [[(offset + i, t) for i, t in line] for line in wrap(words, font, width)]
        offset += len(words)
    ascent, descent = font.getmetrics()
    height = round((ascent + descent) * 1.1)
    top = style.y * out.height - height * len(rows) / 2
    lines = []
    count = 0
    for row, pieces in enumerate(rows):
        text = " ".join(t for _, t in pieces)
        words_of: list[int] = []
        for k, (word, piece) in enumerate(pieces):
            words_of += ([-1] if k else []) + [word] * len(piece)
        chars_of = []
        for word in words_of:
            chars_of.append(count if word >= 0 else -1)
            count += word >= 0
        lines.append(CaptionLine(text, (out.width - font.getlength(text)) / 2, top + row * height,
                                 tuple(words_of), tuple(chars_of)))
    return lines


def draw_caption(draw: ImageDraw.ImageDraw, caption: Caption, lines: list[CaptionLine], font: Font,
                 visible: int | None) -> None:
    """Draw the first `visible` reveal units of a caption (None: all of it)."""
    for line in lines:
        end = len(line.text)
        if visible is not None and caption.reveal is not None:
            units = line.words if caption.reveal.unit == "word" else line.chars
            end = max((k + 1 for k, unit in enumerate(units) if 0 <= unit < visible), default=0)
        if end:
            draw.text((line.x, line.y), line.text[:end], font=font, fill=caption.style.color,
                      stroke_width=round(caption.style.stroke_width * font.size), stroke_fill=caption.style.stroke_color)


def _concat_line(file: Path) -> str:
    return "file '" + str(file).replace("'", "'\\''") + "'\n"


def write_caption_track(edl: Edl, edl_dir: Path, directory: Path) -> tuple[Path, str] | None:
    """Render every caption state to a full-frame PNG and list them with durations for ffmpeg's concat demuxer.

    Returns the list file and an `enable` expression covering every caption.
    """
    if not edl.captions:
        return None
    out, fps, total = edl.output, edl.output.fps, output_frames(edl)
    directory.mkdir(parents=True, exist_ok=True)
    spans = [span_frames(c.timeline_in, c.timeline_out, fps) for c in edl.captions]
    reveals = [reveal_frames(c, fps) for c in edl.captions]
    fonts = [load_font(c.style.font, max(8, round(c.style.size * min(out.width, out.height))), edl_dir)
             for c in edl.captions]
    layouts = [layout_caption(c, font, out) for c, font in zip(edl.captions, fonts)]
    cuts = sorted({0, total, *(f for s in spans for f in s), *(f for r in reveals for f in r)})
    images: dict[tuple[int, ...], Path] = {}
    entries: list[tuple[Path, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        # -1: caption hidden; otherwise the number of revealed units (all units without a reveal).
        state = tuple(-1 if not s <= a < e else (sum(f <= a for f in r) if r else 1 << 30)
                      for (s, e), r in zip(spans, reveals))
        if state not in images:
            canvas = Image.new("RGBA", (out.width, out.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            for caption, lines, font, visible, r in zip(edl.captions, layouts, fonts, state, reveals):
                if visible >= 0:
                    draw_caption(draw, caption, lines, font, visible if r else None)
            images[state] = directory / f"caption_{len(images):04d}.png"
            canvas.save(images[state], compress_level=1)
        if entries and entries[-1][0] == images[state]:
            entries[-1] = (images[state], entries[-1][1] + b - a)
        else:
            entries.append((images[state], b - a))
    listing = directory / "captions.ffconcat"
    body = "".join(_concat_line(img) + f"duration {n / fps:.6f}\n" for img, n in entries)
    listing.write_text("ffconcat version 1.0\n" + body + _concat_line(entries[-1][0]))
    return listing, _balanced_sum([window(s, e, fps) for s, e in spans])


# ---- voice stems ----

STEM_CONTEXT = 1.0  # seconds of soundtrack the separator hears either side of a span; the stem is cut to the span
STEM_VERSION = 1


@dataclass(frozen=True)
class Stem:
    """The voice in one audio layer's source span, separated once and cached as a WAV file."""
    source: Path
    probe: Probe
    track: int
    src_in: float
    src_out: float

    def key(self) -> str:
        stat = self.source.stat()
        # Gain, fades, ducking and placement are applied in the mix, so they do not change the stem.
        data = {"v": STEM_VERSION, "model": separate.MODEL, "overlap": separate.OVERLAP, "context": STEM_CONTEXT,
                "source": str(self.source), "size": stat.st_size, "mtime": stat.st_mtime_ns, "track": self.track,
                "src_in": self.src_in, "src_out": self.src_out}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:20]


def plan_stems(edl: Edl, edl_dir: Path, probes: dict[Path, Probe]) -> dict[int, Stem]:
    """Voice stems by audio layer index."""
    stems: dict[int, Stem] = {}
    for k, layer in enumerate(edl.audio):
        if layer.stem == "voice":
            file = resolve(edl_dir, layer.source)
            stems[k] = Stem(file, probes[file], layer.track, layer.src_in, layer_end(layer))
    return stems


def render_stem(stem: Stem, separator: VoiceSeparator, file: Path, verbose: bool) -> None:
    """Separate the span with STEM_CONTEXT either side (clamped to the source), keep the span."""
    rate = separate.RATE
    start = max(0.0, stem.src_in - STEM_CONTEXT)
    end = stem.src_out + STEM_CONTEXT
    if stem.probe.duration is not None:
        end = min(end, stem.probe.duration)
    what = f"separating the voice in {stem.source.name} from {stem.src_in:.3f} s"
    spec, trim = audio_source(stem.source, stem.probe, start, end)
    raw = run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", *spec, "-map", f"0:a:{stem.track}", "-af",
               f"{trim},aresample={rate},aformat=sample_fmts=flt:channel_layouts=stereo", "-f", "f32le", "-"],
              what, verbose)
    mix = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).T
    first = round((stem.src_in - start) * rate)
    count = round((stem.src_out - stem.src_in) * rate)
    if mix.shape[1] == 0 or mix.shape[1] < first + count - rate // 100:
        raise RenderError(f"{what}: read {mix.shape[1] / rate:.3f} s of audio, expected {end - start:.3f} s; "
                          "the source ends early")
    try:
        voice = separator.separate(mix)[:, first:first + count]
    except SeparationError as e:
        raise RenderError(f"{what}: {e}") from e
    voice = np.pad(voice, ((0, 0), (0, count - voice.shape[1])))
    partial = file.with_name(file.stem + ".partial.wav")
    try:
        run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", "2",
             "-i", "-", "-c:a", "pcm_f32le", str(partial)], what, verbose,
            data=np.ascontiguousarray(voice.T, dtype="<f4").tobytes())
    except RenderError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(file)


# ---- song ----

BLOCK = 256  # samples per audio frame where the song's gain and filters can change (5.3 ms)
ECHO_FLOOR = 3e-5  # echo repeats quieter than this (-90 dB) are dropped
BASS_Q = 0.7071  # Q of the bass shelf: the flattest that does not overshoot


def butterworth_q(slope: int) -> tuple[float, ...]:
    """Qs of the cascaded biquads that make a Butterworth filter falling `slope` dB per octave (12 per biquad)."""
    order = slope // 6
    return tuple(1 / (2 * math.cos((2 * k - 1) * math.pi / (2 * order))) for k in range(1, order // 2 + 1))


def filter_arg(text: str) -> str:
    """`text` quoted as one filter option value inside a filtergraph."""
    quoted = "'" + text.replace("'", "'\\''") + "'"
    return "".join("\\" + c if c in "\\'[],;" else c for c in quoted)


def song_controls(edl: Edl) -> tuple[bool, bool, bool, bool, bool]:
    """Whether the song has a bass shelf and a compressor, and whether its gain, low-pass and high-pass
    change over time."""
    song = edl.song
    return (bool(track(song, "bass_db")),
            song.compress is not None,
            bool(track(song, "gain_db")),
            bool(track(song, "lowpass")) or any(layer.muffle_hz is not None for layer in edl.audio),
            bool(track(song, "highpass")))


def control_values(edl: Edl, t: float) -> dict[str, tuple[float, int]]:
    """(value, on) of each song control in use at output time `t`, by filter name, in processing order: the
    bass shelf's gain in dB, the compressor's mix, the linear gain, and each filter's cutoff, switched off (0)
    at its open cutoff."""
    bass, compress, gain, lowpass, highpass = song_controls(edl)
    values: dict[str, tuple[float, int]] = {}
    if bass:
        values["lowshelf"] = (song_bass_db(edl.song, t), 1)
    if compress:
        values["acompressor"] = (song_compress(edl.song, t), 1)
    if gain:
        values["volume"] = (10 ** (song_gain_db(edl.song, t) / 20), 1)
    if lowpass:
        cutoff = song_lowpass(edl, t)
        values["lowpass"] = (cutoff, int(cutoff < OPEN_LOWPASS * (1 - 1e-9)))
    if highpass:
        cutoff = song_highpass(edl.song, t)
        values["highpass"] = (cutoff, int(cutoff > OPEN_HIGHPASS * (1 + 1e-9)))
    return values


def song_command_text(edl: Edl) -> str | None:
    """asendcmd commands that move the song's gain and filters, one block at a time; None when nothing moves.

    Values are taken at the middle of each BLOCK-sample frame and sent only when they change, so a value
    changes at the start of the block nearest its time. A filter at its open cutoff is switched off (mix 0)
    and keeps running underneath, so switching it back on is smooth.
    """
    if not any(song_controls(edl)):
        return None
    stages = len(butterworth_q(edl.song.filter_slope))
    samples = round(output_frames(edl) * SAMPLE_RATE / edl.output.fps)
    last = control_values(edl, BLOCK / 2 / SAMPLE_RATE)
    lines = []
    for k in range(1, samples // BLOCK + 1):
        commands = []
        for name, (value, on) in control_values(edl, (k * BLOCK + BLOCK / 2) / SAMPLE_RATE).items():
            before, was_on = last[name]
            if abs(value - before) <= 1e-4 * abs(before) and on == was_on:
                continue
            last[name] = (value, on)
            if name == "lowshelf":
                commands.append(f"lowshelf@song_bass gain {value:.4f}")
            elif name == "acompressor":
                commands.append(f"acompressor@song_compress mix {value:.6f}")
            elif name == "volume":
                commands.append(f"volume@song_gain volume {value:.6f}")
            else:
                for stage in range(stages):
                    commands += [f"{name}@song_{name}{stage} frequency {value:.3f}", f"{name}@song_{name}{stage} mix {on}"]
        if commands:  # just before the block starts, so float rounding cannot push it a block late
            lines.append(f"{k * BLOCK / SAMPLE_RATE - 1e-7:.7f} {', '.join(commands)};")
    return "\n".join(lines) + "\n" if lines else None


def song_control_filters(edl: Edl, commands: Path | None) -> list[str]:
    """The song's gain and filters, set to their values at the start of the output and driven by the
    `commands` file of song_command_text when they move."""
    filters = [f"asetnsamples=n={BLOCK}:p=0"]
    if commands is not None:
        filters.append(f"asendcmd=f={filter_arg(str(commands))}")
    for name, (value, on) in control_values(edl, BLOCK / 2 / SAMPLE_RATE).items():
        if name == "lowshelf":
            filters.append(f"lowshelf@song_bass=f={BASS_HZ:g}:width_type=q:width={BASS_Q}:gain={value:.4f}")
        elif name == "acompressor":
            filters.append(compressor_filter(edl, value))
        elif name == "volume":
            filters.append(f"volume@song_gain=volume={value:.6f}:eval=once")
        else:
            filters += [f"{name}@song_{name}{stage}=f={value:.3f}:width_type=q:width={q:.4f}:mix={on}"
                        for stage, q in enumerate(butterworth_q(edl.song.filter_slope))]
    return filters


def compressor_filter(edl: Edl, mix: float) -> str:
    """acompressor with song.compress's settings, `mix` of the compressed song in its output."""
    c = edl.song.compress
    if c is None:
        raise ValueError("the song has no compressor")
    return (f"acompressor@song_compress=threshold={10 ** (c.threshold_db / 20):.6f}:ratio={c.ratio:g}"
            f":attack={c.attack * 1000:g}:release={c.release * 1000:g}:makeup={10 ** (c.makeup_db / 20):.6f}"
            f":mix={mix:.6f}")


def echo_filter(tail: Tail) -> str:
    """aecho of only the repeats: one every delay seconds, each feedback times the last, through the tail."""
    delay = ECHO_DELAY if tail.delay is None else tail.delay
    feedback = ECHO_FEEDBACK if tail.feedback is None else tail.feedback
    delays, decays = [], []
    k = 1
    while k * delay <= tail.input + tail.seconds and feedback ** (k - 1) >= ECHO_FLOOR:
        delays.append(f"{k * delay * 1000:.3f}")
        decays.append(f"{feedback ** (k - 1):.6f}")
        k += 1
    return f"aecho=in_gain=0:out_gain=1:delays={'|'.join(delays)}:decays={'|'.join(decays)}"


def reverb_response(tail: Tail) -> np.ndarray:
    """Stereo impulse response of the reverb tail: decorrelated noise falling 60 dB over `decay` seconds,
    each channel scaled to unit energy so the reverb is about as loud as what goes in."""
    decay = REVERB_DECAY if tail.decay is None else tail.decay
    length = round(min(decay, tail.input + tail.seconds) * SAMPLE_RATE)
    t = np.arange(length) / SAMPLE_RATE
    noise = np.random.default_rng(0).standard_normal((2, length)) * np.exp(-math.log(1000) * t / decay)
    return noise / np.sqrt((noise ** 2).sum(axis=1, keepdims=True))


def write_wav(samples: np.ndarray, file: Path, what: str, verbose: bool) -> None:
    """Write (channels, n) float samples at SAMPLE_RATE as a float WAV."""
    run(["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", "-f", "f32le", "-ar", str(SAMPLE_RATE),
         "-ac", str(samples.shape[0]), "-i", "-", "-c:a", "pcm_f32le", str(file)], what, verbose,
        data=np.ascontiguousarray(samples.T, dtype="<f4").tobytes())


def song_chains(edl: Edl, edl_dir: Path, probes: dict[Path, Probe], add: Callable[..., int], stereo: str,
                commands: Path | None, reverb_ir: Path | None) -> list[str]:
    """Filter chains from the song's source to [song]: pieces, speed, gain, fades, tail, keyframes and ducks."""
    out, song = edl.output, edl.song
    fps, total = out.fps, output_frames(edl)
    samples = round(total * SAMPLE_RATE / fps)
    file = resolve(edl_dir, song.source)
    chains: list[str] = []
    if not song.pieces:
        window = song_pieces(song)[0]
        spec, trim = audio_source(file, probes[file], window.src_in, window.src_out)
        head = f"[{add(*spec)}:a:0]"
        filters = [trim, stereo]
    else:
        for k, piece in enumerate(song.pieces):
            before = join_seconds(song, k) / 2 * song.speed
            after = join_seconds(song, k + 1) / 2 * song.speed
            spec, trim = audio_source(file, probes[file], piece.src_in - before, piece.src_out + after)
            chains.append(f"[{add(*spec)}:a:0]{trim},{stereo}[piece{k}]")
        joined = "[piece0]"
        for k in range(1, len(song.pieces)):
            overlap = round(join_seconds(song, k) * song.speed * SAMPLE_RATE)
            join = (f"acrossfade=ns={overlap}:c1=tri:c2=tri" if overlap else "concat=n=2:v=0:a=1")
            chains.append(f"{joined}[piece{k}]{join}[joined{k}]")
            joined = f"[joined{k}]"
        head, filters = joined, ["asetpts=N/SR/TB"]
    if song.speed != 1:
        filters += [f"asetrate={round(SAMPLE_RATE * song.speed)}", f"aresample={SAMPLE_RATE}", "asetpts=N/SR/TB"]
    if song.gain_db:
        filters.append(f"volume={song.gain_db!r}dB")
    if song.fade_in:
        filters.append(f"afade=t=in:st=0:d={song.fade_in!r}")
    end = total / fps if song.tail is None else song_window(song)  # where the song itself stops
    if song.fade_out:
        filters.append(f"afade=t=out:st={end - song.fade_out:.6f}:d={song.fade_out!r}")
    tail = song.tail
    if tail is not None:
        # The song stops at its end; its last `input` seconds, delayed back into place, feed the effect.
        filters += [f"afade=t=out:st={end - JOIN:.6f}:d={JOIN}", "asplit[dry][send]"]
        chains.append(f"{head}{','.join(filters)}")
        start = round((end - tail.input) * SAMPLE_RATE)
        length = round((tail.input + tail.seconds) * SAMPLE_RATE)
        send = [f"atrim=start_sample={start}", "asetpts=N/SR/TB", f"afade=t=in:d={JOIN}", f"apad=whole_len={length}"]
        if tail.type == "echo":
            chains.append(f"[send]{','.join(send)},{echo_filter(tail)}[wet]")
        else:
            if reverb_ir is None:
                raise ValueError("a reverb tail needs its impulse response file")
            ir = add("-i", str(reverb_ir))
            chains.append(f"[send]{','.join(send)}[sent]")
            chains.append(f"[sent][{ir}:a:0]afir=gtype=none:irnorm=-1:irgain=1[wet]")
        fade = tail.seconds * TAIL_FADE
        chains.append(f"[wet]volume={tail.wet_db!r}dB,atrim=end_sample={length},adelay=delays={start}S:all=1[delayed]")
        head = "[dry][delayed]"
        # amix stamps no times on what it outputs after the song ends; afade and asendcmd need them.
        filters = ["amix=inputs=2:normalize=0:duration=longest", "asetpts=N/SR/TB",
                   f"afade=t=out:st={total / fps - fade:.6f}:d={fade:.6f}"]
    controlled = any(song_controls(edl))
    if controlled:
        filters += song_control_filters(edl, commands)
    duck = duck_expression(edl)
    if duck:
        if not controlled:
            filters.append(f"asetnsamples=n={BLOCK}:p=0")
        filters.append(f"volume='{duck}':eval=frame")
    filters += [f"apad=whole_len={samples}", f"atrim=end_sample={samples}"]
    chains.append(f"{head}{','.join(filters)}[song]")
    return chains


# ---- composite ----

def window(start: int, end: int, fps: int) -> str:
    """Timeline expression true on output frames [start, end)."""
    return f"between(t,{(start - 0.5) / fps:.6f},{(end - 0.5) / fps:.6f})"


SEEK_FROM = 1.0  # an input seek to within one audio frame of the start (21 ms of AAC) drops that frame


def audio_source(file: Path, pr: Probe, src_in: float, src_out: float) -> tuple[list[str], str]:
    """Input args and leading filter for an audio window: seek in video containers, trim exactly in audio
    files and near the start of a video."""
    if pr.video_index is not None and src_in >= SEEK_FROM:
        return ["-ss", f"{src_in:.6f}", "-t", f"{src_out - src_in:.6f}", "-i", str(file)], "asetpts=PTS-STARTPTS"
    return ["-i", str(file)], f"atrim=start={src_in:.6f}:end={src_out:.6f},asetpts=PTS-STARTPTS"


def duck_expression(edl: Edl) -> str | None:
    """Song gain over time: the deepest duck among the layers playing, with linear ramps."""
    fps, r = edl.output.fps, DUCK_RAMP
    terms = []
    for layer in edl.audio:
        if layer.duck_db < 0:
            start, end = layer_times(layer, fps)
            depth = 1 - 10 ** (layer.duck_db / 20)
            terms.append(f"(1-{depth:.6f}*clip(min((t-{start - r:.6f})/{r},({end + r:.6f}-t)/{r}),0,1))")
    if not terms:
        return None
    expr = terms[0]
    for term in terms[1:]:
        expr = f"min({expr},{term})"
    return expr


def mix_chains(edl: Edl, edl_dir: Path, probes: dict[Path, Probe], add: Callable[..., int], stems: dict[int, Path],
               song_commands: Path | None, reverb_ir: Path | None) -> list[str]:
    """Filter chains from the song and the audio layers to the limited mix, [aout]."""
    fps = edl.output.fps
    samples = round(output_frames(edl) * SAMPLE_RATE / fps)
    stereo = f"aresample={SAMPLE_RATE},aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=N/SR/TB"
    chains = song_chains(edl, edl_dir, probes, add, stereo, song_commands, reverb_ir)

    mix = ["[song]"]
    for k, audio in enumerate(edl.audio):
        if k in stems:  # already cut to the span
            spec, trim, stream = ["-i", str(stems[k])], "asetpts=PTS-STARTPTS", 0
        else:
            source = resolve(edl_dir, audio.source)
            spec, trim = audio_source(source, probes[source], audio.src_in, layer_end(audio))
            stream = audio.track
        length = layer_end(audio) - audio.src_in
        filters = [trim, stereo]
        if audio.gain_db:
            filters.append(f"volume={audio.gain_db!r}dB")
        if audio.fade_in:
            filters.append(f"afade=t=in:st=0:d={audio.fade_in!r}")
        if audio.fade_out:
            filters.append(f"afade=t=out:st={length - audio.fade_out:.6f}:d={audio.fade_out!r}")
        delay = round(to_frame(audio.timeline_in, fps) * SAMPLE_RATE / fps)
        filters.append(f"adelay=delays={delay}S:all=1")
        chains.append(f"[{add(*spec)}:a:{stream}]{','.join(filters)}[a{k}]")
        mix.append(f"[a{k}]")
    master = "[song]"
    if len(mix) > 1:
        chains.append(f"{''.join(mix)}amix=inputs={len(mix)}:normalize=0:duration=first[mix]")
        master = "[mix]"
    chains.append(f"{master}alimiter=limit={LIMIT}:level=0:latency=1,"
                  f"apad=whole_len={samples},atrim=end_sample={samples}[aout]")

    return chains


def audio_command(edl: Edl, edl_dir: Path, probes: dict[Path, Probe], stems: dict[int, Path], file: Path,
                  song_commands: Path | None = None, reverb_ir: Path | None = None) -> list[str]:
    """ffmpeg command for the mix alone, as a 24-bit WAV."""
    args: list[str] = []
    inputs = itertools.count()

    def add(*spec: str) -> int:
        args.extend(spec)
        return next(inputs)

    chains = mix_chains(edl, edl_dir, probes, add, stems, song_commands, reverb_ir)
    return ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args, "-filter_complex", ";".join(chains),
            "-map", "[aout]", "-c:a", "pcm_s24le", "-ar", str(SAMPLE_RATE), "-f", "wav", str(file)]


def composite_command(edl: Edl, edl_dir: Path, probes: dict[Path, Probe], picture_list: Path,
                      handles: dict[int, Path], stems: dict[int, Path], captions: tuple[Path, str] | None,
                      file: Path, crf: int, preset: str, song_commands: Path | None = None,
                      reverb_ir: Path | None = None) -> list[str]:
    """The final ffmpeg command. handles are crossfade handles by clip index, stems voice WAVs by audio layer;
    song_commands is the file of song_command_text and reverb_ir the impulse response of a reverb tail."""
    out = edl.output
    fps, total = out.fps, output_frames(edl)
    args: list[str] = []
    chains: list[str] = []
    inputs = itertools.count()
    labels = itertools.count(1)
    cur = "v0"

    def add(*spec: str) -> int:
        args.extend(spec)
        return next(inputs)

    def layer(filters: str, other: str | None = None) -> None:
        nonlocal cur
        nxt = f"v{next(labels)}"
        chains.append(f"[{cur}]" + (f"[{other}]" if other else "") + f"{filters}[{nxt}]")
        cur = nxt

    pic = add("-f", "concat", "-safe", "0", "-i", str(picture_list))
    chains.append(f"[{pic}:v]settb=1/{fps},setpts=N,format=yuv420p[v0]")

    # Crossfades: the outgoing picture (handle or black) on top, fading out over the incoming clip.
    for i, clip in enumerate(edl.clips):
        t = clip.transition
        if t.type != "crossfade":
            continue
        start = clip_frames(clip, fps)[0]
        if i - 1 in handles:
            under = f"[{add('-i', str(handles[i - 1]))}:v]"
        else:
            under = f"color=c=black:s={out.width}x{out.height}:r={fps},trim=end_frame={t.frames},"
        # Timestamp-based fade (frame-count fades depend on graph scheduling). Frame j sits at
        # (j + 1) / fps here, so its alpha is (frames - j) / (frames + 1) and the incoming clip
        # shows through at (j + 1) / (frames + 1).
        chains.append(f"{under}format=yuva420p,settb=1/{fps},setpts=N+1,"
                      f"fade=t=out:st=0:d={(t.frames + 1) / fps:.6f}:alpha=1,setpts=N+{start}[x{i}]")
        layer(f"overlay=eof_action=pass:enable='{window(start, start + t.frames, fps)}'", f"x{i}")

    for k, overlay in enumerate(edl.overlays):
        start, end = span_frames(overlay.timeline_in, overlay.timeline_out, fps)
        source = resolve(edl_dir, overlay.source)
        pr = probes[source]
        if pr.image:
            idx = add("-i", str(source))
            timing, eof = "", "repeat"
        else:
            idx = add("-ss", f"{overlay.src_in:.6f}", "-t", f"{(end - start) / fps + 0.5:.6f}", "-i", str(source))
            timing, eof = f"fps={fps},trim=end_frame={end - start},settb=1/{fps},setpts=N+{start},", "pass"
        chains.append(f"[{idx}:{pr.video_index}]{timing}scale={max(2, round(overlay.width * out.width))}:-1,"
                      f"format=rgba,colorchannelmixer=aa={overlay.opacity!r},{RGBA_TO_YUVA}[o{k}]")
        layer(f"overlay=x='{overlay.x!r}*main_w-overlay_w/2':y='{overlay.y!r}*main_h-overlay_h/2'"
              f":eof_action={eof}:enable='{window(start, end, fps)}'", f"o{k}")

    for clip in edl.clips:
        t = clip.transition
        if t.type == "flash":
            start = clip_frames(clip, fps)[0]
            layer(f"drawbox=x=0:y=0:w=iw:h=ih:color={t.color}:t=fill:enable='{window(start, start + t.frames, fps)}'")

    if captions is not None:
        listing, enable = captions
        idx = add("-f", "concat", "-safe", "0", "-i", str(listing))
        chains.append(f"[{idx}:v]fps={fps},tpad=stop_mode=clone:stop={total},trim=end_frame={total},"
                      f"settb=1/{fps},setpts=N,{RGBA_TO_YUVA}[captions]")
        layer(f"overlay=eof_action=pass:enable='{enable}'", "captions")

    fade = to_frame(edl.ending.fade_to_black, fps)
    if fade:  # the last `fade` frames dim to black; the final frame is black
        layer(f"fade=t=out:st={(total - fade - 1) / fps:.6f}:d={fade / fps:.6f}")
    chains.append(f"[{cur}]format=yuv420p,{TAG_BT709}[vout]")

    chains += mix_chains(edl, edl_dir, probes, add, stems, song_commands, reverb_ir)

    return ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-y", *args,
            "-filter_complex", ";".join(chains), "-map", "[vout]", "-map", "[aout]", "-frames:v", str(total),
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "256k", "-ar", str(SAMPLE_RATE), "-movflags", "+faststart", "-f", "mp4", str(file)]


# ---- edl_table.md and proof.jpg ----

def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "<br>")


def _transition(clip: Clip) -> str:
    t = clip.transition
    return {"cut": "cut", "flash": f"{t.color} flash {t.frames}f", "crossfade": f"crossfade {t.frames}f"}[t.type]


def _motion(clip: Clip) -> str:
    """Speed, direction and animation of a clip in a few words."""
    lo, hi = speed_range(clip)
    parts = ["freeze" if hi == 0 else f"{lo:g}×" if lo == hi else f"ramp {lo:g}–{hi:g}×"]
    if clip.reverse:
        parts.append("reversed")
    keyed = [name for name in ("zoom", "x", "y", "shake") if track(clip, name)]
    if keyed:
        parts.append("keys " + "/".join(keyed))
    if clip.slowmo != "repeat":
        parts.append(clip.slowmo)
    return ", ".join(parts)


def write_table(edl: Edl, name: str, file: Path) -> None:
    fps, total = edl.output.fps, output_frames(edl)
    song, out = edl.song, edl.output

    def span(a: int, b: int) -> str:
        return f"{a / fps:.3f}–{b / fps:.3f}"

    windows = ", ".join(f"{piece.src_in:.3f}–{piece.src_out:.3f}" for piece in song_pieces(song))
    tail, comp = song.tail, song.compress
    compressed = (f", compressed {comp.ratio:g}:1 over {comp.threshold_db:g} dB (attack {comp.attack * 1000:g} ms, "
                  f"release {comp.release * 1000:g} ms, makeup {comp.makeup_db:+g} dB)" if comp else "")
    lines = [f"# {edl.title or name}", "",
             f"{out.width}×{out.height} at {fps} fps, {total / fps:.3f} s ({total} frames). "
             f"Song `{song.source}` {windows} s at {song.speed:g}×"
             f"{f', fade in {song.fade_in:g} s' if song.fade_in else ''}"
             f"{f', fade out {song.fade_out:g} s' if song.fade_out else ''}{compressed}"
             f"{f', {tail.type} tail {tail.seconds:g} s from the last {tail.input:g} s' if tail else ''}"
             f"{f', ending fade to black {edl.ending.fade_to_black:g} s' if edl.ending.fade_to_black else ''}.", ""]
    if edl.notes:
        lines += [edl.notes, ""]
    lines += ["| # | Time (s) | Frames | Source | Source time (s) | Motion | Enter | Sync | Why |",
              "|---|---|---|---|---|---|---|---|---|"]
    cursor = 0
    for i, clip in enumerate(edl.clips):
        a, b = clip_frames(clip, fps)
        if a > cursor:
            lines.append(f"| | {span(cursor, a)} | {a - cursor} | black | | | | | |")
        lines.append(f"| {i + 1} | {span(a, b)} | {b - a} | {_cell(clip.source)} | "
                     f"{clip.src_in:.3f}–{source_end(clip, fps):.3f} | {_motion(clip)} | {_transition(clip)} | "
                     f"{clip.sync or ''} | {_cell(clip.why or '')} |")
        cursor = b
    if cursor < total:
        lines.append(f"| | {span(cursor, total)} | {total - cursor} | black | | | | | |")
    if song.keyframes:
        bass, compress = bool(track(song, "bass_db")), bool(track(song, "compress"))
        headers = ["Song key", "Time (s)", "Ease", "Gain", "Low-pass", "High-pass"]
        headers += ["Bass"] * bass + ["Compress"] * compress
        lines += ["", f"| {' | '.join(headers)} |", "|" + "---|" * len(headers)]
        for i, key in enumerate(song.keyframes):
            cells = [f"{key.gain_db:+g} dB" if key.gain_db is not None else "",
                     f"{key.lowpass:g} Hz" if key.lowpass is not None else "",
                     f"{key.highpass:g} Hz" if key.highpass is not None else ""]
            if bass:
                cells.append(f"{key.bass_db:+g} dB" if key.bass_db is not None else "")
            if compress:
                cells.append(f"{key.compress:.0%}" if key.compress is not None else "")
            lines.append(f"| {i + 1} | {key.t:.3f} | {key.ease} | {' | '.join(cells)} |")
    if edl.audio:
        lines += ["", "| Audio layer | Time (s) | Source | Source time (s) | Gain | Song duck | Why |",
                  "|---|---|---|---|---|---|---|"]
        for i, layer in enumerate(edl.audio):
            start, end = layer_times(layer, fps)
            muffle = f", muffled to {layer.muffle_hz:g} Hz" if layer.muffle_hz is not None else ""
            lines.append(f"| {i + 1} | {start:.3f}–{end:.3f} | "
                         f"{_cell(layer.source)} (track {layer.track}{', voice' if layer.stem == 'voice' else ''}) | "
                         f"{layer.src_in:.3f}–{layer_end(layer):.3f} | "
                         f"{layer.gain_db:+g} dB | {layer.duck_db:g} dB{muffle} | {_cell(layer.why or '')} |")
    if edl.overlays:
        lines += ["", "| Overlay | Time (s) | Source | Opacity | Why |", "|---|---|---|---|---|"]
        for i, overlay in enumerate(edl.overlays):
            a, b = span_frames(overlay.timeline_in, overlay.timeline_out, fps)
            lines.append(f"| {i + 1} | {span(a, b)} | {_cell(overlay.source)} | {overlay.opacity:g} | "
                         f"{_cell(overlay.why or '')} |")
    if edl.captions:
        lines += ["", "| Caption | Time (s) | Text | Reveal |", "|---|---|---|---|"]
        for i, caption in enumerate(edl.captions):
            a, b = span_frames(caption.timeline_in, caption.timeline_out, fps)
            reveal = f"by {caption.reveal.unit}" if caption.reveal else ""
            lines.append(f"| {i + 1} | {span(a, b)} | {_cell(caption.text)} | {reveal} |")
    file.write_text("\n".join(lines) + "\n")


def grab_frames(video: Path, frames: list[int], size: tuple[int, int]) -> dict[int, Image.Image]:
    """Decode the given output frames, scaled to `size`."""
    wanted = sorted(set(frames))
    expr = _balanced_sum([f"eq(n,{f})" for f in wanted])
    proc = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vf",
                           f"select='{expr}',scale={size[0]}:{size[1]}", "-fps_mode", "passthrough",
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True)
    step = size[0] * size[1] * 3
    if proc.returncode != 0 or len(proc.stdout) != step * len(wanted):
        raise RenderError(f"cannot read proof frames from {video}: {proc.stderr.decode(errors='replace')[-500:]}")
    return {f: Image.frombytes("RGB", size, proc.stdout[k * step:(k + 1) * step]) for k, f in enumerate(wanted)}


def write_proof(video: Path, edl: Edl, edl_dir: Path, file: Path) -> None:
    """One row per cut: the frames either side (and the first clean frame after a flash or crossfade)."""
    out, fps, total = edl.output, edl.output.fps, output_frames(edl)
    long_side = 320
    tw, th = ((long_side, round(long_side * out.height / out.width / 2) * 2) if out.width >= out.height
              else (round(long_side * out.width / out.height / 2) * 2, long_side))
    rows: list[tuple[list[tuple[int | None, str]], str, str]] = []
    for i, clip in enumerate(edl.clips):
        a, b = clip_frames(clip, fps)
        t = clip.transition
        shots: list[tuple[int | None, str]] = [(a - 1 if a > 0 else None, "before"), (a, "cut")]
        if t.type != "cut":
            shots.append((a + t.frames, f"after {t.type}"))
        sync = f"sync {clip.sync} · " if clip.sync else ""
        title = (f"#{i + 1} · {a / fps:.3f} s · frame {a} · {sync}{_transition(clip)} · "
                 f"{Path(clip.source).name} {clip.src_in:.3f}–{source_end(clip, fps):.3f} s · {_motion(clip)}")
        rows.append((shots, title, clip.why or ""))
        nxt = clip_frames(edl.clips[i + 1], fps)[0] if i + 1 < len(edl.clips) else total
        if b < nxt:
            rows.append(([(b - 1, "before"), (b, "black")], f"black · {b / fps:.3f}–{nxt / fps:.3f} s · frame {b}",
                         "No clip: the picture cuts to black."))
    images = grab_frames(video, [f for shots, _, _ in rows for f, _ in shots if f is not None], (tw, th))

    font = load_font(None, 18, edl_dir)
    small = load_font(None, 14, edl_dir)
    pad, label, text_w = 12, 22, 620

    def lines_of(text: str, f: Font) -> list[str]:
        return [" ".join(t for _, t in line) for line in wrap(text.split(), f, text_w)]

    texts = [(lines_of(title, small), lines_of(why, font)) for _, title, why in rows]
    heights = [max(th + label, 20 * len(t) + 8 + 24 * len(w)) + pad for t, w in texts]
    sheet = Image.new("RGB", (pad + 3 * (tw + pad) + text_w + pad, pad + sum(heights)), "#181818")
    draw = ImageDraw.Draw(sheet)
    y = pad
    for (shots, _, _), (title_lines, why_lines), height in zip(rows, texts, heights):
        for k, (frame, name) in enumerate(shots):
            x = pad + k * (tw + pad)
            if frame is None:
                draw.rectangle((x, y, x + tw - 1, y + th - 1), outline="#555555")
                draw.text((x + 8, y + 8), "start", font=small, fill="#888888")
            else:
                sheet.paste(images[frame], (x, y))
                draw.text((x, y + th + 3), f"{name}  f{frame}  {frame / fps:.3f}s", font=small, fill="#bbbbbb")
        tx = pad + 3 * (tw + pad)
        for n, line in enumerate(title_lines):
            draw.text((tx, y + 20 * n), line, font=small, fill="#ffd400")
        for n, line in enumerate(why_lines):
            draw.text((tx, y + 20 * len(title_lines) + 8 + 24 * n), line, font=font, fill="#f0f0f0")
        y += height
    sheet.save(file, quality=90)


# ---- driver ----

def run(cmd: list[str], what: str, verbose: bool, data: bytes | None = None) -> bytes:
    """Run ffmpeg with `data` on stdin; returns its stdout."""
    if verbose:
        print(shlex.join(cmd), file=sys.stderr)
    proc = subprocess.run(cmd, input=data, capture_output=True)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.decode(errors="replace").strip().splitlines()[-15:])
        raise RenderError(f"ffmpeg failed while {what}:\n{tail}")
    return proc.stdout


def render_segment(seg: Segment, out: Output, file: Path, verbose: bool) -> None:
    partial = file.with_name(file.stem + ".partial.mp4")
    what = (f"rendering {seg.source.name} from {seg.clip.src_in:.3f} s" if seg.source and seg.clip
            else "rendering black")
    run(segment_command(seg, out, partial), what, verbose)
    got = count_frames(partial)
    if got != seg.frames:
        partial.unlink(missing_ok=True)
        raise RenderError(f"{what}: got {got} frames, expected {seg.frames}; the source ends early")
    partial.replace(file)


def render(edl_path: Path, out_path: Path | None, *, jobs: int = 4, crf: int = 16, preset: str = "medium",
           verbose: bool = False) -> str:
    """Validate the EDL; with out_path, render it. Returns a one-line summary."""
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RenderError(f"{tool} is not on PATH")
    edl = load_edl(edl_path)
    edl_dir = edl_path.resolve().parent
    probes, problems = check_sources(edl, edl_dir)
    if not problems:
        edl, problems = resolve_layers(edl, edl_dir, probes)
    if problems:
        raise EdlError(problems)
    out, fps, total = edl.output, edl.output.fps, output_frames(edl)
    summary = (f"{total / fps:.3f} s, {total} frames at {fps} fps, {out.width}x{out.height}, "
               f"{len(edl.clips)} clips, {len(edl.audio)} audio layers, {len(edl.overlays)} overlays, "
               f"{len(edl.captions)} captions")
    if out_path is None:
        derived = [f"  clips[{i}].src_out = {source_end(clip, fps):.4f} (derived from timeline_out and speed)"
                   for i, clip in enumerate(edl.clips) if clip.src_out is None]
        return "\n".join([f"{edl_path}: OK, {summary}", *derived])

    out_path = out_path.resolve()
    cache = out_path.parent / CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    stems = plan_stems(edl, edl_dir, probes)
    stem_files = {k: cache / f"stem_{stem.key()}.wav" for k, stem in stems.items()}
    separating = {stem_files[k]: stem for k, stem in stems.items() if not stem_files[k].exists()}
    if separating:  # one at a time: the model already uses the whole GPU
        with VoiceSeparator(verbose=verbose) as separator:
            for file, stem in separating.items():
                render_stem(stem, separator, file, verbose)
    # Files the song's filters read, rewritten (or removed) on every render.
    song_commands: Path | None = None
    reverb_ir: Path | None = None
    for stale in (cache / "song_commands.txt", cache / "reverb.wav"):
        stale.unlink(missing_ok=True)
    if (text := song_command_text(edl)) is not None:
        song_commands = cache / "song_commands.txt"
        song_commands.write_text(text)
    if edl.song.tail is not None and edl.song.tail.type == "reverb":
        reverb_ir = cache / "reverb.wav"
        write_wav(reverb_response(edl.song.tail), reverb_ir, "writing the reverb's impulse response", verbose)
    stemmed = f"; {len(separating)} voice stems separated, {len(set(stem_files.values())) - len(separating)} cached"

    if out_path.suffix.lower() == ".wav":  # the mix alone; the picture and its cache are left as they are
        partial = out_path.with_name(out_path.stem + ".partial.wav")
        try:
            run(audio_command(edl, edl_dir, probes, stem_files, partial, song_commands, reverb_ir), "mixing", verbose)
        except RenderError:
            partial.unlink(missing_ok=True)
            raise
        partial.replace(out_path)
        write_table(edl, edl_path.name, out_path.parent / "edl_table.md")
        return f"{out_path}: audio only, {summary}{stemmed if stem_files else ''}"

    picture, handles = plan_picture(edl, edl_dir, probes)
    files = {seg: cache / f"seg_{seg.key(out)}.mp4" for seg in [*picture, *handles.values()]}
    todo = {files[seg]: seg for seg in files if not files[seg].exists()}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(lambda item: render_segment(item[1], out, item[0], verbose), todo.items()))
    picture_list = cache / "picture.ffconcat"
    picture_list.write_text("ffconcat version 1.0\n" + "".join(_concat_line(files[seg]) for seg in picture))
    caption_dir = cache / "captions"
    shutil.rmtree(caption_dir, ignore_errors=True)
    captions = write_caption_track(edl, edl_dir, caption_dir)
    partial = out_path.with_name(out_path.stem + ".partial.mp4")
    try:
        run(composite_command(edl, edl_dir, probes, picture_list, {i: files[s] for i, s in handles.items()},
                              stem_files, captions, partial, crf, preset, song_commands, reverb_ir), "compositing", verbose)
    except RenderError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(out_path)

    write_table(edl, edl_path.name, out_path.parent / "edl_table.md")
    write_proof(out_path, edl, edl_dir, out_path.parent / "proof.jpg")
    keep = set(files.values()) | set(stem_files.values())
    for stale in [*cache.glob("seg_*.mp4"), *cache.glob("stem_*.wav")]:
        if stale not in keep:
            stale.unlink()
    done = f"{len(todo)} segments rendered, {len(set(files.values())) - len(todo)} cached"
    if stem_files:
        done += stemmed
    return f"{out_path}: {summary}; {done}"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edl", type=Path)
    ap.add_argument("out", type=Path, nargs="?", help="output MP4, or WAV for the audio alone (omit with --check)")
    ap.add_argument("--check", action="store_true", help="validate the EDL and its sources without rendering")
    ap.add_argument("--jobs", type=positive_int, default=min(4, os.cpu_count() or 1), help="segments rendered at once")
    ap.add_argument("--crf", type=int, choices=range(0, 52), default=16, metavar="0-51", help="x264 quality (default 16)")
    ap.add_argument("--preset", choices=PRESETS, default="medium", help="x264 preset (default medium)")
    ap.add_argument("--verbose", action="store_true", help="print every ffmpeg command")
    a = ap.parse_args()
    if a.out is None and not a.check:
        ap.error("OUT is required unless --check is given")
    try:
        print(render(a.edl, None if a.check else a.out, jobs=a.jobs, crf=a.crf, preset=a.preset, verbose=a.verbose))
    except EdlError as e:
        print(f"{a.edl}: {len(e.problems)} problem(s)", *[f"  - {p}" for p in e.problems], sep="\n", file=sys.stderr)
        return 2
    except RenderError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
