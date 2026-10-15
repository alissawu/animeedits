"""Split a video into shots and save the best frame of each.

usage: python detect.py INPUT OUT_DIR [--faces live|anime]
See docs/frame_selection.md.
"""
import argparse, json, os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    ap.add_argument("--faces", choices=["live", "anime"], help="prefer frames with a face, using a live-action or anime detector")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    det = make_detector(a.faces)
    n_chunks = N_CHUNKS if det else 1

    scenes = [(s.frame_num, e.frame_num) for s, e in detect(a.input, ContentDetector(threshold=20, min_scene_len=4))]
    cap = cv2.VideoCapture(a.input); fps = cap.get(5)
    if not scenes: scenes = [(0, int(cap.get(7)))]

    # one sequential pass: each chunk keeps a running best (score, frame_idx, image)
    best = [[None] * n_chunks for _ in scenes]
    si, f = 0, 0
    while si < len(scenes):
        ok, fr = cap.read()
        if not ok: break
        while si < len(scenes) and f >= scenes[si][1]: si += 1
        if si >= len(scenes): break
        lo, hi = usable_range(*scenes[si])
        if lo <= f < hi:
            c = min(n_chunks - 1, (f - lo) * n_chunks // (hi - lo))
            sc = score_frame(fr)
            if best[si][c] is None or sc > best[si][c][0]: best[si][c] = (sc, f, fr)
        f += 1

    out = []
    for i, ((a0, b), chunks) in enumerate(zip(scenes, best)):
        cands = sorted([c for c in chunks if c], key=lambda t: -t[0])
        if not cands: continue
        pick, row = cands[0], {}
        if det:
            row = dict(face=False, face_checks=0)
            for c in cands:
                row["face_checks"] += 1
                if has_face(det, c[2]): pick, row["face"] = c, True; break
        p = f"scene_{i+1:02d}_{pick[1]/fps:06.2f}s.jpg"  # e.g. scene_04_005.90s.jpg
        cv2.imwrite(os.path.join(a.out_dir, p), pick[2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append(dict(i=i+1, start=round(a0/fps, 2), end=round(b/fps, 2), best=round(pick[1]/fps, 2), file=p, **row))
    json.dump(out, open(os.path.join(a.out_dir, "scenes.json"), "w"), indent=1)
    print(f"{len(out)} scenes -> {a.out_dir}")


if __name__ == "__main__":
    main()
