"""Split a video into shots and save the best frame of each.

usage: python detect.py INPUT OUT_DIR
"""
import argparse, json, os
import cv2, numpy as np
from scenedetect import detect, ContentDetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    scenes = detect(a.input, ContentDetector(threshold=20, min_scene_len=4))
    cap = cv2.VideoCapture(a.input); fps = cap.get(5)
    out = []
    for i, (s, e) in enumerate(scenes):
        a0, b = s.frame_num, e.frame_num
        pad = max(1, (b - a0) // 6)  # skip transition-blend frames at edges
        best = None
        for f in range(a0 + pad if b - a0 > 4 else a0, b - pad if b - a0 > 4 else b):
            cap.set(1, f); ok, fr = cap.read()
            if not ok: continue
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(g, cv2.CV_64F).var()
            contrast = g.std(); mean = g.mean()
            expo = 1 - abs(mean - 115) / 115  # prefer mid exposure
            score = np.log1p(sharp) * 0.5 + np.log1p(contrast) * 0.35 + max(expo, 0) * 0.15 * 5
            if best is None or score > best[0]: best = (score, f, fr)
        if best is None: continue
        p = f"scene_{i+1:02d}_{best[1]/fps:06.2f}s.jpg"  # e.g. scene_04_005.90s.jpg
        cv2.imwrite(os.path.join(a.out_dir, p), best[2], [cv2.IMWRITE_JPEG_QUALITY, 95])
        out.append(dict(i=i+1, start=round(a0/fps, 2), end=round(b/fps, 2), best=round(best[1]/fps, 2), file=p))
    json.dump(out, open(os.path.join(a.out_dir, "scenes.json"), "w"), indent=1)
    print(f"{len(out)} scenes -> {a.out_dir}")


if __name__ == "__main__":
    main()
