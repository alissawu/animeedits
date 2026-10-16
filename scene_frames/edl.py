"""Edit decision list (EDL): schema, loader and validation. render.py renders it.

An EDL is a JSON object. Times are seconds. `src_*` times are positions in a source
file; `timeline_*` times are positions on the output timeline and snap to the nearest
output frame. Relative paths resolve against the EDL file's directory. Unknown fields
are errors, so typos never pass silently.

  output    {width, height, fps}: even pixel sizes and a whole-number frame rate.
            9:16 output from 16:9 sources crops each clip (see clip.crop).
  song      {source, src_in, src_out, speed=1, gain_db=0, fade_in=0, fade_out=0}
            One continuous window of the song. Output length = (src_out - src_in) / speed;
            the picture is cut to fit the audio, never the reverse. speed resamples like a
            turntable, so sped-up audio is also pitched up. Fades are seconds of output.
            The final mix goes through a peak limiter at -0.45 dBFS.
  clips     [{source, src_in, timeline_in, timeline_out, src_out, sync, why, speed=1,
              reverse=false, slowmo="repeat", crop, keyframes, shake_hz=12, color,
              transition}], in timeline order, never overlapping. Time not covered by a
            clip is black. src_in picks the source frame nearest to that time.
            Length: timeline_out - timeline_in, or (src_out - src_in) / speed when there is
            no timeline_out (constant speed only). With both, src_out must match where the
            speed has carried the source by timeline_out; leave it out and it is derived
            (--check prints it, edl_table.md shows it). speed 0 freezes on src_in.
            sync and why are optional notes for critique: sync is what the cut lands on
            (beat, drop, word, phrase_end, motion or free), why is free text. The renderer
            does not interpret them; edl_table.md and proof.jpg show them.
            crop {zoom=1, x=0.5, y=0.5}: the largest window of the output's aspect ratio,
            divided by zoom (>= 1), centred on the source point (x, y) as fractions of
            width and height, kept inside the frame.
            keyframes [{t, ease="linear", speed, zoom, x, y, shake}]: t is seconds from the
            clip's start on the output timeline, strictly increasing. A key sets any of
            the properties. Each property moves from its previous key to this one with
            `ease`: linear, in, out, in_out, or hold (keep the old value until t, then
            jump). Before its first key a property has that key's value, after its last
            key the last value, and with no keys the clip's speed / crop / no shake.
            Speed keys make ramps (slow, fast, slow around a hit) and freezes (speed 0,
            entered with hold); the source span is the integral of the speed curve.
            shake is a two-axis wobble at shake_hz, its amplitude a fraction of the
            output's shorter side; key it down to 0 to decay it. The window stays inside
            the frame, so zoom in a little to give a shake room.
            slowmo: where in-between frames of slowed footage come from. repeat (default)
            holds the latest source frame; blend crossfades neighbours; motion uses
            motion-compensated interpolation (ffmpeg minterpolate). Anime is usually drawn
            on 2s or 3s (8-12 drawings a second), so blend and motion can smear line art.
            color {brightness=0 (-1..1), contrast=1 (0..3), saturation=1 (0..3),
            tint={color: "#rrggbb", amount: 0..1}}: tint mixes every pixel toward color.
            transition {type="cut", frames, color="white"}: how the clip enters.
              flash: the clip's first `frames` frames are solid white or black; the clip
                keeps playing underneath, so its picture appears frames later.
              crossfade: the clip fades in over its first `frames` frames. If the
                previous clip ends exactly here it keeps playing past its src_out
                underneath (a handle), otherwise the clip fades in from black.
  audio     [{source, src_in, src_out, timeline_in, track=0, stem="mix", gain_db=0,
              duck_db=0, fade_in=0.01, fade_out=0.01, why}]: extra layers, e.g. film dialogue.
            track is the source's audio stream number. stem "mix" plays that track as it
            is; "voice" keeps only the voices (lines, breaths, screams) and drops the
            score and sound effects under them, so a line can sit on the song without the
            show's music. The voice is separated by a model (separate.py) from the span
            plus a second of context either side, takes about real time on Apple silicon
            and is cached in render_cache/. duck_db (<= 0) lowers the song
            while the layer plays, ramping over DUCK_RAMP (0.08) seconds on either side.
            Overlapping layers do not stack their ducks; the deepest one applies.
  overlays  [{source, timeline_in, timeline_out, src_in=0, opacity=1, x=0.5, y=0.5,
              width=1, why}]: an image or video composited over the picture. (x, y) is
            its centre as fractions of the output; width is a fraction of output width.
            Overlays sit above clips and crossfades, below flashes and captions.
  captions  [{text, timeline_in, timeline_out, style, reveal}], drawn above everything
            else. style {font, size=0.05, color="#ffffff", stroke_color="#000000",
            stroke_width=0.08, y=0.8, max_width=0.9}: font is a file path or an installed
            font file name (default: a bold sans-serif); size is a fraction of the
            output's shorter side; stroke_width a fraction of size; y the vertical centre
            of the text block. Lines wrap at max_width of the output width.
            reveal {unit: "word" | "char", seconds | times}: typewriter reveal. With
            seconds, units appear evenly and the last one appears `seconds` after
            timeline_in. times lists the output time of every unit.
  ending    {fade_to_black=0}: seconds over which the whole picture, captions included,
            fades to black; the last frame is black.
  title, notes: optional free text.
"""
from collections.abc import Callable
import dataclasses
from dataclasses import dataclass, field
import difflib
import json
import math
from pathlib import Path
import re
import types
from typing import Literal, Union, cast, get_args, get_origin, get_type_hints

Sync = Literal["beat", "drop", "word", "phrase_end", "motion", "free"]
Ease = Literal["linear", "in", "out", "in_out", "hold"]
SlowMo = Literal["repeat", "blend", "motion"]
AudioStem = Literal["mix", "voice"]

ANIMATED = ("speed", "zoom", "x", "y", "shake")
DUCK_RAMP = 0.08
HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}")


@dataclass(frozen=True)
class Output:
    width: int
    height: int
    fps: int


@dataclass(frozen=True)
class Song:
    source: str
    src_in: float
    src_out: float
    speed: float = 1.0
    gain_db: float = 0.0
    fade_in: float = 0.0
    fade_out: float = 0.0


@dataclass(frozen=True)
class AudioLayer:
    source: str
    src_in: float
    src_out: float
    timeline_in: float
    track: int = 0
    stem: AudioStem = "mix"
    gain_db: float = 0.0
    duck_db: float = 0.0
    fade_in: float = 0.01
    fade_out: float = 0.01
    why: str | None = None


@dataclass(frozen=True)
class Crop:
    zoom: float = 1.0
    x: float = 0.5
    y: float = 0.5


@dataclass(frozen=True)
class Keyframe:
    t: float
    ease: Ease = "linear"
    speed: float | None = None
    zoom: float | None = None
    x: float | None = None
    y: float | None = None
    shake: float | None = None


@dataclass(frozen=True)
class Tint:
    color: str
    amount: float


@dataclass(frozen=True)
class Color:
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    tint: Tint | None = None


@dataclass(frozen=True)
class Transition:
    type: Literal["cut", "flash", "crossfade"] = "cut"
    frames: int = 0
    color: Literal["white", "black"] = "white"


@dataclass(frozen=True)
class Clip:
    source: str
    src_in: float
    timeline_in: float
    timeline_out: float | None = None
    src_out: float | None = None
    sync: Sync | None = None
    why: str | None = None
    speed: float = 1.0
    reverse: bool = False
    slowmo: SlowMo = "repeat"
    crop: Crop = field(default_factory=Crop)
    keyframes: tuple[Keyframe, ...] = ()
    shake_hz: float = 12.0
    color: Color = field(default_factory=Color)
    transition: Transition = field(default_factory=Transition)


@dataclass(frozen=True)
class Overlay:
    source: str
    timeline_in: float
    timeline_out: float
    src_in: float = 0.0
    opacity: float = 1.0
    x: float = 0.5
    y: float = 0.5
    width: float = 1.0
    why: str | None = None


@dataclass(frozen=True)
class CaptionStyle:
    font: str | None = None
    size: float = 0.05
    color: str = "#ffffff"
    stroke_color: str = "#000000"
    stroke_width: float = 0.08
    y: float = 0.8
    max_width: float = 0.9


@dataclass(frozen=True)
class Reveal:
    unit: Literal["word", "char"]
    seconds: float | None = None
    times: tuple[float, ...] | None = None


@dataclass(frozen=True)
class Caption:
    text: str
    timeline_in: float
    timeline_out: float
    style: CaptionStyle = field(default_factory=CaptionStyle)
    reveal: Reveal | None = None


@dataclass(frozen=True)
class Ending:
    fade_to_black: float = 0.0


@dataclass(frozen=True)
class Edl:
    output: Output
    song: Song
    clips: tuple[Clip, ...]
    audio: tuple[AudioLayer, ...] = ()
    overlays: tuple[Overlay, ...] = ()
    captions: tuple[Caption, ...] = ()
    ending: Ending = field(default_factory=Ending)
    title: str | None = None
    notes: str | None = None


class EdlError(ValueError):
    """An EDL that cannot be rendered; `problems` lists every reason, one per line."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = problems


# ---- frame math shared by validation and rendering ----

def to_frame(seconds: float, fps: int) -> int:
    """Nearest output frame to `seconds`; exact halves round up."""
    return math.floor(seconds * fps + 0.5 + 1e-9)


def output_frames(edl: Edl) -> int:
    song = edl.song
    return to_frame((song.src_out - song.src_in) / song.speed, edl.output.fps)


def clip_frames(clip: Clip, fps: int) -> tuple[int, int]:
    """[start, end) output frames of a clip."""
    if clip.timeline_out is not None:
        end = clip.timeline_out
    elif clip.src_out is not None and clip.speed > 0:
        end = clip.timeline_in + (clip.src_out - clip.src_in) / clip.speed
    else:
        raise ValueError("a clip needs timeline_out, or src_out with a speed above 0")
    return to_frame(clip.timeline_in, fps), to_frame(end, fps)


def track(clip: Clip, name: str) -> list[tuple[float, float, Ease]]:
    """(t, value, ease) of the keyframes that set property `name`."""
    points = []
    for key in clip.keyframes:
        value = getattr(key, name)
        if value is not None:
            points.append((key.t, float(value), key.ease))
    return points


def speed_range(clip: Clip) -> tuple[float, float]:
    values = [v for _, v, _ in track(clip, "speed")] or [clip.speed]
    return min(values), max(values)


# Integrals from 0 to u of the ease curves on u in [0, 1), turning a speed curve into source time.
_EASE_INTEGRAL: dict[str, Callable[[float], float]] = {
    "linear": lambda u: u * u / 2,
    "in": lambda u: u ** 3 / 3,
    "out": lambda u: u * u - u ** 3 / 3,
    "in_out": lambda u: u ** 3 - u ** 4 / 2,
    "hold": lambda u: 0.0,
}


def advance(clip: Clip, t: float) -> float:
    """Source seconds played in the first `t` output seconds of the clip: the integral of its speed."""
    points = track(clip, "speed")
    if not points:
        return clip.speed * t
    first_t, first_v, _ = points[0]
    total = first_v * min(t, first_t)
    for (a, va, _), (b, vb, ease) in zip(points, points[1:]):
        if t <= a:
            break
        u = (min(t, b) - a) / (b - a)
        total += (b - a) * (va * u + (vb - va) * _EASE_INTEGRAL[ease](u))
    last_t, last_v, _ = points[-1]
    return total + last_v * max(0.0, t - last_t)


def source_offset(clip: Clip, fps: int, frame: int) -> float:
    """Seconds after src_in of the source moment shown at output `frame` of the clip.

    Frames past the clip's end continue at its final speed (crossfade handles). A reversed
    clip plays the same source span backwards, ending on src_in.
    """
    if not clip.reverse:
        return advance(clip, frame / fps)
    start, end = clip_frames(clip, fps)
    return advance(clip, (end - start - 1) / fps) - advance(clip, frame / fps)


def source_end(clip: Clip, fps: int) -> float:
    """The clip's effective src_out: where its speed has carried the source by its last frame's end."""
    start, end = clip_frames(clip, fps)
    return clip.src_in + advance(clip, (end - start) / fps)


def span_frames(timeline_in: float, timeline_out: float, fps: int) -> tuple[int, int]:
    return to_frame(timeline_in, fps), to_frame(timeline_out, fps)


def crossfade_handle(edl: Edl, index: int) -> int:
    """Frames clip `index` must keep playing past its end under the next clip's crossfade."""
    if index + 1 >= len(edl.clips):
        return 0
    nxt = edl.clips[index + 1]
    fps = edl.output.fps
    if nxt.transition.type != "crossfade" or clip_frames(edl.clips[index], fps)[1] != clip_frames(nxt, fps)[0]:
        return 0
    return nxt.transition.frames


def reveal_units(text: str, unit: str) -> int:
    return len(text.split()) if unit == "word" else sum(not c.isspace() for c in text)


def reveal_frames(caption: Caption, fps: int) -> list[int]:
    """Output frame at which each reveal unit appears."""
    reveal = caption.reveal
    if reveal is None:
        return []
    if reveal.times is not None:
        return [to_frame(t, fps) for t in reveal.times]
    count = reveal_units(caption.text, reveal.unit)
    seconds = reveal.seconds or 0.0
    step = seconds / (count - 1) if count > 1 else 0.0
    return [to_frame(caption.timeline_in + k * step, fps) for k in range(count)]


# ---- JSON -> dataclasses ----

class _Invalid:
    pass


_INVALID = _Invalid()


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _parse(value: object, tp: object, path: str, problems: list[str]) -> object:
    """Convert a JSON value to schema type `tp`, appending any problems; _INVALID on failure."""
    origin = get_origin(tp)
    if origin is Union or origin is types.UnionType:
        if value is None:
            return None
        (inner,) = [a for a in get_args(tp) if a is not type(None)]
        return _parse(value, inner, path, problems)
    if origin is Literal:
        choices = get_args(tp)
        if isinstance(value, str) and value in choices:
            return value
        problems.append(f"{path}: must be one of {', '.join(choices)}, got {json.dumps(value)}")
        return _INVALID
    if origin is tuple:
        if not isinstance(value, list):
            problems.append(f"{path}: must be a list")
            return _INVALID
        items = [_parse(v, get_args(tp)[0], f"{path}[{i}]", problems) for i, v in enumerate(value)]
        return _INVALID if any(item is _INVALID for item in items) else tuple(items)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return _parse_object(value, tp, path, problems)
    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
        kind = "a number"
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        kind = "a whole number"
    elif tp is bool:
        if isinstance(value, bool):
            return value
        kind = "true or false"
    elif tp is str:
        if isinstance(value, str):
            return value
        kind = "a string"
    else:
        raise TypeError(f"unsupported schema type {tp!r}")
    problems.append(f"{path}: must be {kind}, got {json.dumps(value)}")
    return _INVALID


def _parse_object(value: object, cls: type, path: str, problems: list[str]) -> object:
    if not isinstance(value, dict):
        problems.append(f"{path or 'EDL'}: must be an object, got {json.dumps(value)[:60]}")
        return _INVALID
    hints = get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    ok = True
    for key in value:
        if key not in fields:
            near = difflib.get_close_matches(key, list(fields), n=1)
            hint = f"did you mean {near[0]!r}?" if near else f"allowed: {', '.join(fields)}"
            problems.append(f"{_join(path, key)}: unknown field ({hint})")
            ok = False
    kwargs: dict[str, object] = {}
    for name, f in fields.items():
        sub = _join(path, name)
        if name not in value:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                problems.append(f"{sub}: required")
                ok = False
            continue
        parsed = _parse(value[name], hints[name], sub, problems)
        if parsed is _INVALID:
            ok = False
        else:
            kwargs[name] = parsed
    return cls(**kwargs) if ok else _INVALID


# ---- semantic checks ----

def _s(seconds: float) -> str:
    return f"{seconds:.3f} s"


def check(edl: Edl) -> list[str]:
    """Problems that make the EDL unrenderable regardless of its source files."""
    p: list[str] = []

    def need(ok: bool, path: str, message: str) -> None:
        if not ok:
            p.append(f"{path}: {message}")

    def within(value: float, lo: float, hi: float, path: str) -> None:
        need(lo <= value <= hi, path, f"must be from {lo:g} to {hi:g}, got {value:g}")

    def color(value: str, path: str) -> None:
        need(HEX_COLOR.fullmatch(value) is not None, path, f'must look like "#rrggbb", got {value!r}')

    def framing(zoom: float | None, x: float | None, y: float | None, path: str) -> None:
        for name, value, lo, hi in (("zoom", zoom, 1, 8), ("x", x, 0, 1), ("y", y, 0, 1)):
            if value is not None:
                within(value, lo, hi, f"{path}.{name}")

    def source_range(src_in: float, src_out: float, path: str) -> None:
        need(src_in >= 0, f"{path}.src_in", f"must be >= 0, got {src_in:g}")
        need(src_out > src_in, f"{path}.src_out", f"must be after src_in ({src_in:g}), got {src_out:g}")

    out = edl.output
    for name in ("width", "height"):
        size = getattr(out, name)
        need(16 <= size <= 8192 and size % 2 == 0, f"output.{name}", f"must be an even number from 16 to 8192, got {size}")
    within(out.fps, 1, 120, "output.fps")
    if p:
        return p  # every later check depends on the output format
    fps = out.fps

    song = edl.song
    source_range(song.src_in, song.src_out, "song")
    within(song.speed, 0.25, 4, "song.speed")
    within(song.gain_db, -60, 24, "song.gain_db")
    if p:
        return p
    total = output_frames(edl)
    duration = total / fps
    need(total >= 1, "song", "the window is shorter than one output frame")
    need(song.fade_in >= 0 and song.fade_out >= 0 and song.fade_in + song.fade_out <= duration + 1e-9,
         "song", f"fade_in and fade_out must be >= 0 and fit in the {_s(duration)} output")

    need(len(edl.clips) > 0, "clips", "at least one clip is required")
    prev_end = 0
    for i, clip in enumerate(edl.clips):
        path = f"clips[{i}]"
        before_clip = len(p)
        need(clip.src_in >= 0, f"{path}.src_in", f"must be >= 0, got {clip.src_in:g}")
        need(clip.timeline_in >= 0, f"{path}.timeline_in", f"must be >= 0, got {clip.timeline_in:g}")
        within(clip.speed, 0, 20, f"{path}.speed")
        within(clip.shake_hz, 0.1, 60, f"{path}.shake_hz")
        framing(clip.crop.zoom, clip.crop.x, clip.crop.y, f"{path}.crop")
        within(clip.color.brightness, -1, 1, f"{path}.color.brightness")
        within(clip.color.contrast, 0, 3, f"{path}.color.contrast")
        within(clip.color.saturation, 0, 3, f"{path}.color.saturation")
        if clip.color.tint is not None:
            color(clip.color.tint.color, f"{path}.color.tint.color")
            within(clip.color.tint.amount, 0, 1, f"{path}.color.tint.amount")
        for k, key in enumerate(clip.keyframes):
            kp = f"{path}.keyframes[{k}]"
            if k == 0:
                need(key.t >= 0, f"{kp}.t", f"must be >= 0, got {key.t:g}")
            else:
                need(key.t > clip.keyframes[k - 1].t, f"{kp}.t", f"must be after the previous keyframe, got {key.t:g}")
            need(any(getattr(key, name) is not None for name in ANIMATED), kp, f"sets none of {', '.join(ANIMATED)}")
            if key.speed is not None:
                within(key.speed, 0, 20, f"{kp}.speed")
            framing(key.zoom, key.x, key.y, kp)
            if key.shake is not None:
                within(key.shake, 0, 0.25, f"{kp}.shake")
        if clip.timeline_out is None:
            if clip.src_out is None:
                p.append(f"{path}: give timeline_out, src_out or both")
            else:
                need(clip.src_out > clip.src_in, f"{path}.src_out", f"must be after src_in ({clip.src_in:g}), got {clip.src_out:g}")
            need(clip.speed > 0, f"{path}.speed", "a freeze (speed 0) needs timeline_out")
            need(not track(clip, "speed"), f"{path}.keyframes", "a speed curve needs timeline_out; src_out is derived from it")
        else:
            need(clip.timeline_out > clip.timeline_in, f"{path}.timeline_out",
                 f"must be after timeline_in ({clip.timeline_in:g}), got {clip.timeline_out:g}")
            if clip.src_out is not None:
                need(clip.src_out >= clip.src_in, f"{path}.src_out", f"must not be before src_in ({clip.src_in:g})")
        if len(p) > before_clip:
            continue
        start, end = clip_frames(clip, fps)
        length = end - start
        if length < 1:
            p.append(f"{path}: lasts {_s((end - start) / fps)} after snapping, under one frame at {fps} fps")
            continue
        if clip.keyframes:
            # One frame of slack: keys written at the unsnapped length should not fail.
            need(clip.keyframes[-1].t <= (length + 1) / fps + 1e-6, f"{path}.keyframes",
                 f"the last keyframe ({_s(clip.keyframes[-1].t)}) is past the clip's {_s(length / fps)} length")
        if clip.timeline_out is not None and clip.src_out is not None:
            derived = source_end(clip, fps)
            need(abs(derived - clip.src_out) <= 0.5 * speed_range(clip)[1] / fps + 1e-6, f"{path}.src_out",
                 f"{clip.src_out:g} s does not match the speed, which reaches source {derived:.4f} s by timeline_out; "
                 f"set src_out to {derived:.4f} or leave it out")
        need(end <= total, path,
             f"ends at {_s(end / fps)} (frame {end}), after the song window ends at {_s(duration)} (frame {total}); "
             "the picture is cut to fit the audio")
        if i > 0 and start < prev_end:
            before = edl.clips[i - 1]
            if before.timeline_out is not None:
                fix = f"clips[{i - 1}].timeline_out = {start / fps:.4f}"
            else:
                fix = f"clips[{i - 1}].src_out = {before.src_in + (start / fps - before.timeline_in) * before.speed:.4f}"
            p.append(f"{path}: starts at {_s(start / fps)} (frame {start}), {prev_end - start} frame(s) before "
                     f"clips[{i - 1}] ends at {_s(prev_end / fps)}; clips must be in timeline order without overlap "
                     f"(e.g. {fix})")
        t = clip.transition
        if t.type == "cut":
            need(t.frames == 0, f"{path}.transition.frames", "a cut has no frames; use type flash or crossfade")
        else:
            need(1 <= t.frames < length, f"{path}.transition.frames",
                 f"a {t.type} needs 1 to {length - 1} frames for this {length}-frame clip, got {t.frames}")
        prev_end = max(prev_end, end)
    if not p:
        for i, clip in enumerate(edl.clips):
            handle = crossfade_handle(edl, i)
            start, end = clip_frames(clip, fps)
            need(not clip.reverse or clip.src_in + source_offset(clip, fps, end - start + handle - 1) >= -1e-9,
                 f"clips[{i}]", f"the next clip's {handle}-frame crossfade plays this reversed clip past source time 0")

    for i, layer in enumerate(edl.audio):
        path = f"audio[{i}]"
        source_range(layer.src_in, layer.src_out, path)
        need(layer.timeline_in >= 0, f"{path}.timeline_in", f"must be >= 0, got {layer.timeline_in:g}")
        need(layer.track >= 0, f"{path}.track", f"must be >= 0, got {layer.track}")
        within(layer.gain_db, -60, 24, f"{path}.gain_db")
        within(layer.duck_db, -60, 0, f"{path}.duck_db")
        seconds = layer.src_out - layer.src_in
        need(layer.fade_in >= 0 and layer.fade_out >= 0 and layer.fade_in + layer.fade_out <= seconds + 1e-9,
             path, "fade_in and fade_out must be >= 0 and fit in the layer")
        ends = to_frame(layer.timeline_in, fps) / fps + seconds
        need(ends <= duration + 1e-6, path, f"ends at {_s(ends)}, after the output ends at {_s(duration)}")

    for i, overlay in enumerate(edl.overlays):
        path = f"overlays[{i}]"
        start, end = span_frames(overlay.timeline_in, overlay.timeline_out, fps)
        need(0 <= start < end <= total, path,
             f"timeline_in..timeline_out must cover at least one frame inside 0..{_s(duration)}")
        need(overlay.src_in >= 0, f"{path}.src_in", f"must be >= 0, got {overlay.src_in:g}")
        within(overlay.opacity, 0, 1, f"{path}.opacity")
        within(overlay.width, 0.01, 10, f"{path}.width")
        within(overlay.x, -1, 2, f"{path}.x")
        within(overlay.y, -1, 2, f"{path}.y")

    for i, caption in enumerate(edl.captions):
        path = f"captions[{i}]"
        need(caption.text.strip() != "", f"{path}.text", "must not be empty")
        start, end = span_frames(caption.timeline_in, caption.timeline_out, fps)
        need(0 <= start < end <= total, path,
             f"timeline_in..timeline_out must cover at least one frame inside 0..{_s(duration)}")
        style = caption.style
        within(style.size, 0.005, 0.5, f"{path}.style.size")
        color(style.color, f"{path}.style.color")
        color(style.stroke_color, f"{path}.style.stroke_color")
        within(style.stroke_width, 0, 0.5, f"{path}.style.stroke_width")
        within(style.y, 0, 1, f"{path}.style.y")
        within(style.max_width, 0.05, 1, f"{path}.style.max_width")
        reveal = caption.reveal
        if reveal is None or not caption.text.strip():
            continue
        if (reveal.seconds is None) == (reveal.times is None):
            p.append(f"{path}.reveal: give exactly one of seconds or times")
            continue
        units = reveal_units(caption.text, reveal.unit)
        if reveal.times is not None and len(reveal.times) != units:
            p.append(f"{path}.reveal.times: needs one time per {reveal.unit} ({units}), got {len(reveal.times)}")
            continue
        if reveal.seconds is not None and reveal.seconds <= 0:
            p.append(f"{path}.reveal.seconds: must be > 0, got {reveal.seconds:g}")
            continue
        frames = reveal_frames(caption, fps)
        need(all(a <= b for a, b in zip(frames, frames[1:])), f"{path}.reveal.times", "must not decrease")
        need(start <= frames[0] and frames[-1] < end, f"{path}.reveal",
             f"every {reveal.unit} must appear between timeline_in and timeline_out")

    fade = edl.ending.fade_to_black
    need(fade >= 0 and to_frame(fade, fps) < total, "ending.fade_to_black",
         f"must be >= 0 and shorter than the {_s(duration)} output, got {fade:g}")
    return p


def parse_edl(raw: object) -> Edl:
    problems: list[str] = []
    edl = _parse(raw, Edl, "", problems)
    if isinstance(edl, _Invalid):
        raise EdlError(problems)
    edl = cast(Edl, edl)
    problems = check(edl)
    if problems:
        raise EdlError(problems)
    return edl


def load_edl(path: Path) -> Edl:
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise EdlError([f"{path.name}: invalid JSON at line {e.lineno} column {e.colno}: {e.msg}"]) from None
    return parse_edl(raw)
