"""Split a video into shots and record what an editor needs from each one.

Per shot: the best keyframe, the first and last frames (for match cuts), a
per-frame motion curve with its peaks, bright flashes, and look stats.

usage: python detect.py INPUT OUT_DIR [--faces live|anime] [--sample N] [--proxy WIDTH]
See docs/frame_selection.md.
"""
from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Sequence
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Literal, NotRequired, TypedDict, cast

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

HERE = Path(__file__).resolve().parent
YUNET = HERE / "models" / "face_detection_yunet_2023mar.onnx"
ANIME = HERE / "models" / "lbpcascade_animeface.xml"
N_CHUNKS = 3
ANALYSIS_WIDTH = 160    # motion and flashes are measured on a downscale about this wide
LOOK_WIDTH = 80         # look stats pool pixels from a smaller downscale about this wide
LOOK_FRAMES = 8         # evenly spaced usable frames per shot pooled for look stats
PALETTE_SIZE = 5
GRAY_SATURATION = 0.05  # below this mean saturation a shot has no meaningful hue
FLASH_RISE = 0.25       # luma jump (0-1) that starts a flash
FLASH_MAX_S = 0.25      # longer bright stretches are lighting changes, not flashes
PEAK_WINDOW_S = 0.25    # motion is smoothed over this window; peaks are further apart than it
PEAK_RATIO = 1.5        # a peak must exceed this multiple of the shot's median motion...
PEAK_FLOOR = 0.05       # ...and this many frame widths per second

Frame = cv2.typing.MatLike
FaceMode = Literal["live", "anime"]
FaceDetector = cv2.FaceDetectorYN | cv2.CascadeClassifier


class Still(TypedDict):
    time: float
    file: str


class Flash(TypedDict):
    time: float  # first bright frame
    frames: int


class Motion(TypedDict):
    fps: float
    # Per shot frame: optical-flow speed from the previous frame, in frame widths per second.
    # None where flow means nothing: the shot's first frame, and into, during and out of a flash.
    curve: list[float | None]
    peaks: list[float]  # times of motion bursts
    mean: float
    max: float


class Swatch(TypedDict):
    hex: str
    share: float


class Look(TypedDict):
    luma: float         # 0-1
    hue: float | None   # degrees, saturation-weighted; None for near-gray shots
    saturation: float   # 0-1
    palette: list[Swatch]


class Shot(TypedDict):
    i: int
    start: float
    end: float
    best: float
    file: str
    first: Still
    last: Still
    flashes: list[Flash]
    motion: Motion
    look: Look
    face: NotRequired[bool]
    face_checks: NotRequired[int]


def score_frame(fr: Frame) -> float:
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(g, cv2.CV_64F).var()
    contrast = g.std()
    mean = g.mean()
    expo = 1 - abs(mean - 115) / 115  # prefer mid exposure
    return float(np.log1p(sharp) * 0.5 + np.log1p(contrast) * 0.35 + max(expo, 0) * 0.15 * 5)


def make_detector(kind: FaceMode | None) -> FaceDetector | None:
    if kind == "live":
        return cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.7)
    if kind == "anime":
        return cv2.CascadeClassifier(str(ANIME))
    return None


def has_face(det: FaceDetector, img: Frame) -> bool:
    if isinstance(det, cv2.CascadeClassifier):
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        m = min(g.shape) // 12
        return len(det.detectMultiScale(g, 1.1, 5, minSize=(m, m))) > 0
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    return faces is not None and len(faces) > 0


def usable_range(a0: int, b: int) -> tuple[int, int]:
    """skip transition-blend frames at shot edges"""
    if b - a0 <= 4:
        return a0, b
    pad = max(1, (b - a0) // 6)
    return a0 + pad, b - pad


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def add_detection_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--faces", choices=["live", "anime"])
    ap.add_argument("--sample", type=positive_int, help="score N evenly spaced frames per shot (default: all)")
    ap.add_argument("--threshold", type=float, default=23)
    ap.add_argument("--min-len", type=positive_int, default=8)
    ap.add_argument("--proxy", type=positive_int, help="detect cuts on a temporary proxy of this width")


def flash_runs(luma: Sequence[float], max_len: int) -> list[tuple[int, int]]:
    """[start, end) runs of at most max_len frames that are FLASH_RISE brighter than
    the frames on both sides and than the shot's median frame. A run touching the
    start or end of the shot has only one side to compare with; the median keeps
    the frames around a dip to black from passing as flashes there."""
    n = len(luma)
    if n == 0:
        return []
    bright = float(np.median(luma)) + FLASH_RISE
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if i > 0 and luma[i] - luma[i - 1] < FLASH_RISE:
            i += 1
            continue
        low = luma[i]
        end = None
        for j in range(i + 1, min(n, i + max_len) + 1):
            if j == n:
                end = n if i > 0 else None
                break
            if luma[j] <= low - FLASH_RISE:
                end = j
                break
            low = min(low, luma[j])
        if end is not None and low >= bright and (i == 0 or low - luma[i - 1] >= FLASH_RISE):
            runs.append((i, end))
            i = end
        else:
            i += 1
    return runs


def motion_peaks(curve: Sequence[float | None], fps: float) -> list[int]:
    """Frame indices of motion bursts: local maxima of the curve averaged over
    PEAK_WINDOW_S, each moved to the fastest measured frame under its window.
    Averaging first keeps animation on twos or threes, where every other frame
    is a held drawing, from producing a peak per drawing. None means unmeasured."""
    measured = np.array([v is not None for v in curve])
    if measured.sum() < 2:
        return []
    values = np.array([v if v is not None else 0.0 for v in curve], dtype=np.float64)
    n = len(values)
    w = max(1, round(fps * PEAK_WINDOW_S))
    smooth = np.convolve(values, np.ones(w) / w, mode="same")
    floor = max(PEAK_FLOOR, PEAK_RATIO * float(np.median(smooth[measured])))
    picked: list[int] = []
    for k in map(int, np.argsort(-smooth, kind="stable")):
        if smooth[k] < floor:
            break
        if smooth[k] < np.max(smooth[max(0, k - w):k + w + 1]) or any(abs(k - p) <= w for p in picked):
            continue
        picked.append(k)
    # mode="same" puts values[k - w//2 : k + (w-1)//2 + 1] under smooth[k]
    peaks: set[int] = set()
    for k in picked:
        lo, hi = max(0, k - w // 2), min(n, k + (w - 1) // 2 + 1)
        window = np.where(measured[lo:hi], values[lo:hi], -1.0)
        if window.max() >= 0:
            peaks.add(lo + int(np.argmax(window)))
    return sorted(peaks)


def flow_speed(prev: Frame, cur: Frame) -> float:
    """Mean optical-flow magnitude in pixels between two grayscale frames."""
    flow = cv2.calcOpticalFlowFarneback(prev, cur, np.empty((*prev.shape[:2], 2), np.float32),
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.hypot(flow[..., 0], flow[..., 1]).mean())


def look_stats(frames: Sequence[Frame]) -> Look:
    """Mean luma, saturation-weighted hue, mean saturation and a k-means Lab palette."""
    bgr = np.concatenate([f.reshape(-1, 3) for f in frames]).reshape(-1, 1, 3).astype(np.float32) / 255
    luma = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)  # float input: H in degrees, S in 0-1
    sat = hsv[:, 1].astype(np.float64)
    angle = np.radians(hsv[:, 0].astype(np.float64))
    hue = None
    if sat.mean() >= GRAY_SATURATION:
        hue = round(math.degrees(math.atan2(float((sat * np.sin(angle)).sum()),
                                            float((sat * np.cos(angle)).sum()))) % 360, 1)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).reshape(-1, 3)
    k = min(PALETTE_SIZE, len(lab))
    cv2.setRNGSeed(0)  # deterministic palettes for identical input
    _, labels, centers = cv2.kmeans(lab, k, np.zeros((len(lab), 1), np.int32),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5),
                                    1, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel().astype(np.int64), minlength=k)
    colors = cv2.cvtColor(centers.reshape(-1, 1, 3), cv2.COLOR_Lab2BGR).reshape(-1, 3)
    shares: dict[str, int] = {}
    for (b, g, r), count in zip(np.clip(np.rint(colors * 255), 0, 255).astype(int), counts):
        if count:
            name = f"#{r:02x}{g:02x}{b:02x}"
            shares[name] = shares.get(name, 0) + int(count)
    palette = [Swatch(hex=name, share=round(count / len(lab), 3))
               for name, count in sorted(shares.items(), key=lambda item: -item[1])]
    return Look(luma=round(float(luma.mean()), 3), hue=hue, saturation=round(float(sat.mean()), 3), palette=palette)


def find_cuts(input: str | Path, threshold: float, min_len: int, proxy: int | None) -> list[tuple[int, int]]:
    with tempfile.TemporaryDirectory(prefix="animeedits-") as tmp:
        source = str(input)
        if proxy:
            source = str(Path(tmp) / "proxy.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(input), "-an",
                            "-vf", f"scale={proxy}:-2", "-c:v", "libx264",
                            "-preset", "ultrafast", "-crf", "27", "-threads", "4",
                            source], check=True)
        return [(s.frame_num, e.frame_num) for s, e in
                detect(source, ContentDetector(threshold=threshold, min_scene_len=min_len))]


def downscale(img: Frame, factor: int) -> Frame:
    """Shrink by a whole-number factor. OpenCV area-averages that on a fast path; an arbitrary
    target size such as 1920x800 -> 160x67 takes over ten times longer per frame."""
    return cv2.resize(img, None, fx=1 / factor, fy=1 / factor, interpolation=cv2.INTER_AREA)


def write_jpeg(path: Path, img: Frame) -> None:
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"cannot write {path}")


def extract_frames(input: str | Path, out_dir: str | Path, *, faces: FaceMode | None = None,
                   sample: int | None = None, threshold: float = 23, min_len: int = 8,
                   proxy: int | None = None) -> list[Shot]:
    """Detect cuts, then stream original frames; keep only one shot in memory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    det = make_detector(faces)
    n_chunks = N_CHUNKS if det is not None else 1
    cap = cv2.VideoCapture(str(input))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    if fps <= 0 or total <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("video has no readable frames or frame rate")
    small_factor = max(1, round(width / ANALYSIS_WIDTH))
    look_factor = max(1, round(width / LOOK_WIDTH))
    max_flash = max(1, round(fps * FLASH_MAX_S))
    try:
        scenes = find_cuts(input, threshold, min_len, proxy) or [(0, total)]
        shots: list[Shot] = []
        f = 0
        for i, (a0, b) in enumerate(scenes, 1):
            n = b - a0
            lo, hi = usable_range(a0, b)
            selected = (set(np.linspace(lo, hi - 1, min(sample, hi - lo), dtype=int).tolist())
                        if sample else None)
            look_at = set(np.linspace(lo, hi - 1, min(LOOK_FRAMES, hi - lo), dtype=int).tolist())
            # Each chunk keeps alternates so a flash can be dropped once the shot's flashes are known.
            chunks: list[list[tuple[float, int, Frame]]] = [[] for _ in range(n_chunks)]
            flow: list[float] = []  # flow[k - 1]: speed into shot frame k
            luma: list[float] = []
            look_frames: list[tuple[int, Frame]] = []
            head: list[Frame] = []  # enough frames to step past a leading flash
            tail: deque[Frame] = deque(maxlen=max_flash + 1)  # ...and back from a trailing one
            prev: Frame | None = None
            while f < b:
                if f < a0:
                    if not cap.grab():
                        raise RuntimeError(f"video ended unexpectedly at frame {f}")
                    f += 1
                    continue
                ok, fr = cap.read()
                if not ok:
                    raise RuntimeError(f"video ended unexpectedly at frame {f}")
                gray = downscale(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), small_factor)
                luma.append(float(gray.mean()) / 255)
                if prev is not None:
                    flow.append(round(flow_speed(prev, gray) / gray.shape[1] * fps, 4))
                prev = gray
                if len(head) <= max_flash:
                    head.append(fr)
                tail.append(fr)
                if f in look_at:
                    look_frames.append((f, downscale(fr, look_factor)))
                if lo <= f < hi and (selected is None or f in selected):
                    chunk = chunks[min(n_chunks - 1, (f - lo) * n_chunks // (hi - lo))]
                    chunk.append((score_frame(fr), f, fr))
                    if len(chunk) > max_flash + 1:
                        del chunk[min(range(len(chunk)), key=lambda j: chunk[j][0])]
                f += 1
            runs = flash_runs(luma, max_flash)
            in_flash = {a0 + k for s, e in runs for k in range(s, e)}
            cands = sorted((max([c for c in chunk if c[1] not in in_flash] or chunk, key=lambda c: c[0])
                            for chunk in chunks if chunk), key=lambda c: -c[0])
            if not cands:
                raise RuntimeError(f"no candidate for shot {i}")
            pick, found, checks = cands[0], False, 0
            if det is not None:
                for cand in cands:
                    checks += 1
                    if has_face(det, cand[2]):
                        pick, found = cand, True
                        break
            name = f"scene_{i:04d}_{pick[1] / fps:06.2f}s.jpg"
            write_jpeg(out / name, pick[2])

            # Edge frames skip a flash at either end: a white frame says nothing about a match cut.
            first_k = runs[0][1] if runs and runs[0][0] == 0 else 0
            last_k = runs[-1][0] - 1 if runs and runs[-1][1] == n else n - 1
            if first_k > last_k:
                first_k, last_k = 0, n - 1
            edges: dict[str, Still] = {}
            for edge, k, img in (("first", first_k, head[first_k]),
                                 ("last", last_k, tail[last_k - (n - len(tail))])):
                t = (a0 + k) / fps
                edges[edge] = Still(time=round(t, 3), file=f"scene_{i:04d}_{edge}_{t:06.2f}s.jpg")
                write_jpeg(out / edges[edge]["file"], img)

            # Optical flow assumes constant brightness, so it means nothing across a flash.
            curve: list[float | None] = [None, *flow]
            for s, e in runs:
                for k in range(s, min(e, n - 1) + 1):
                    curve[k] = None
            speeds = [v for v in curve if v is not None]
            looks = [fr for k, fr in look_frames if k not in in_flash] or [fr for _, fr in look_frames]
            shot = Shot(i=i, start=a0 / fps, end=b / fps, best=pick[1] / fps, file=name,
                        first=edges["first"], last=edges["last"],
                        flashes=[Flash(time=round((a0 + s) / fps, 3), frames=e - s) for s, e in runs],
                        motion=Motion(fps=fps, curve=curve,
                                      peaks=[round((a0 + k) / fps, 3) for k in motion_peaks(curve, fps)],
                                      mean=round(sum(speeds) / len(speeds), 4) if speeds else 0.0,
                                      max=max(speeds, default=0.0)),
                        look=look_stats(looks))
            if det is not None:
                shot["face"], shot["face_checks"] = found, checks
            shots.append(shot)
        (out / "scenes.json").write_text(json.dumps(shots, indent=2))
        return shots
    finally:
        cap.release()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("out_dir")
    add_detection_args(ap)
    a = ap.parse_args()
    shots = extract_frames(a.input, a.out_dir, faces=cast("FaceMode | None", a.faces), sample=a.sample,
                           threshold=a.threshold, min_len=a.min_len, proxy=a.proxy)
    print(f"{len(shots)} shots -> {a.out_dir}")


if __name__ == "__main__":
    main()
