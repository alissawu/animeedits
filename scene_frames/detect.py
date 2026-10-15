"""Split a video into shots and save the best frame of each.

usage: python detect.py INPUT OUT_DIR [--faces live|anime]
See docs/frame_selection.md.
"""
import argparse, json, os, subprocess, tempfile
from pathlib import Path
import cv2, numpy as np
from scenedetect import detect, ContentDetector

HERE = os.path.dirname(os.path.abspath(__file__))
YUNET = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")
ANIME = os.path.join(HERE, "models", "lbpcascade_animeface.xml")
N_CHUNKS = 3


def score_frame(fr):
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(g, cv2.CV_64F).var()
    contrast = g.std(); mean = g.mean()
    expo = 1 - abs(mean - 115) / 115  # prefer mid exposure
    return np.log1p(sharp) * 0.5 + np.log1p(contrast) * 0.35 + max(expo, 0) * 0.15 * 5


def make_detector(kind):
    if kind == "live":
        return cv2.FaceDetectorYN.create(YUNET, "", (320, 320), 0.7)
    if kind == "anime":
        return cv2.CascadeClassifier(ANIME)
    return None


def has_face(det, img):
    if isinstance(det, cv2.CascadeClassifier):
        g = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        m = min(g.shape) // 12
        return len(det.detectMultiScale(g, 1.1, 5, minSize=(m, m))) > 0
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    return faces is not None and len(faces) > 0


def usable_range(a0, b):
    """skip transition-blend frames at shot edges"""
    if b - a0 <= 4: return a0, b
    pad = max(1, (b - a0) // 6)
    return a0 + pad, b - pad


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def add_detection_args(ap):
    ap.add_argument("--faces", choices=["live", "anime"])
    ap.add_argument("--sample", type=positive_int, help="score N evenly spaced frames per shot (default: all)")
    ap.add_argument("--threshold", type=float, default=23)
    ap.add_argument("--min-len", type=positive_int, default=8)
    ap.add_argument("--proxy", type=positive_int, help="detect cuts on a temporary proxy of this width")


def extract_frames(input, out_dir, *, faces=None, sample=None, threshold=23, min_len=8, proxy=None):
    """Detect cuts, then stream original frames; keep only one shot in memory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det = make_detector(faces)
    n_chunks = N_CHUNKS if det is not None else 1
    cap = cv2.VideoCapture(str(input))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total <= 0:
        cap.release()
        raise ValueError("video has no readable frames or frame rate")
    try:
        with tempfile.TemporaryDirectory(prefix="animeedits-") as tmp:
            source = str(input)
            if proxy:
                source = str(Path(tmp) / "proxy.mp4")
                subprocess.run(["ffmpeg", "-v", "error", "-i", str(input), "-an",
                                "-vf", f"scale={proxy}:-2", "-c:v", "libx264",
                                "-preset", "ultrafast", "-crf", "27", "-threads", "4",
                                source], check=True)
            scenes = [(s.frame_num, e.frame_num) for s, e in
                      detect(source, ContentDetector(threshold=threshold, min_scene_len=min_len))]
        if not scenes:
            scenes = [(0, total)]
        out = []
        f = 0
        for i, (a0, b) in enumerate(scenes):
            lo, hi = usable_range(a0, b)
            selected = (set(np.linspace(lo, hi - 1, min(sample, hi - lo), dtype=int))
                        if sample else None)
            chunks = [None] * n_chunks
            while f < b:
                ok = cap.grab()
                if not ok:
                    raise RuntimeError(f"video ended unexpectedly at frame {f}")
                if lo <= f < hi and (selected is None or f in selected):
                    ok, fr = cap.retrieve()
                    if not ok:
                        raise RuntimeError(f"cannot decode frame {f}")
                    c = min(n_chunks - 1, (f - lo) * n_chunks // (hi - lo))
                    sc = score_frame(fr)
                    if chunks[c] is None or sc > chunks[c][0]:
                        chunks[c] = (sc, f, fr)
                f += 1
            cands = sorted([c for c in chunks if c is not None], key=lambda t: -t[0])
            if not cands:
                raise RuntimeError(f"no candidate for shot {i + 1}")
            pick, row = cands[0], {}
            if det is not None:
                row = dict(face=False, face_checks=0)
                for c in cands:
                    row["face_checks"] += 1
                    if has_face(det, c[2]):
                        pick, row["face"] = c, True
                        break
            name = f"scene_{i+1:04d}_{pick[1]/fps:06.2f}s.jpg"
            if not cv2.imwrite(str(out_dir / name), pick[2], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"cannot write {name}")
            out.append(dict(i=i+1, start=a0/fps, end=b/fps,
                            best=pick[1]/fps, file=name, **row))
            # Release the finished shot before decoding the next one.
            chunks = cands = pick = fr = None
        (out_dir / "scenes.json").write_text(json.dumps(out, indent=2))
        return out
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("out_dir")
    add_detection_args(ap)
    a = ap.parse_args()
    scenes = extract_frames(a.input, a.out_dir, faces=a.faces, sample=a.sample,
                            threshold=a.threshold, min_len=a.min_len, proxy=a.proxy)
    print(f"{len(scenes)} scenes -> {a.out_dir}")


if __name__ == "__main__":
    main()
