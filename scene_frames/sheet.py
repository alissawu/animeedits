"""Draw one contact sheet of every shot in a detect.py output directory.

usage: python sheet.py OUT_DIR [--labels labels.json] [--cols 3] [--width 480] [--output sheet.jpg]

Each tile shows the shot's keyframe, ID and time range. --labels takes a JSON
object mapping shot IDs to text printed under the tiles, e.g. {"4": "fireworks kiss"}.
For a whole film, use index.py's numbered sheets instead: this loads one image at a time
but draws a single sheet.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

try:
    from detect import Shot, positive_int
    from index import make_sheet
except ImportError:  # imported as scene_frames.sheet
    from .detect import Shot, positive_int  # type: ignore[import-not-found,no-redef]
    from .index import make_sheet  # type: ignore[import-not-found,no-redef]


def read_labels(path: Path) -> dict[int, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise ValueError(f"{path} must map shot IDs to strings")
    try:
        return {int(k): v for k, v in data.items()}
    except ValueError:
        raise ValueError(f"{path} has a non-integer shot ID") from None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="directory holding scenes.json and its frames")
    ap.add_argument("--labels", type=Path, help="JSON object of shot ID -> text")
    ap.add_argument("--cols", type=positive_int, default=3)
    ap.add_argument("--width", type=positive_int, default=480, help="tile width in pixels")
    ap.add_argument("--output", type=Path, help="default: OUT_DIR/sheet.jpg")
    a = ap.parse_args()
    scenes = a.out_dir / "scenes.json"
    if not scenes.is_file():
        ap.error(f"no scenes.json in {a.out_dir}; run detect.py first")
    shots = cast("list[Shot]", json.loads(scenes.read_text()))
    try:
        labels = read_labels(a.labels) if a.labels else None
    except (OSError, ValueError) as e:
        ap.error(str(e))
    output = a.output or a.out_dir / "sheet.jpg"
    make_sheet(shots, a.out_dir, output, cols=a.cols, tile_width=a.width, labels=labels)
    print(f"{len(shots)} shots -> {output}")


if __name__ == "__main__":
    main()
