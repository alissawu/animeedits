# scene_frames

Exact scripts used to split an edit into shots and grab the best frame per shot.

detect.py INPUT OUT_DIR: PySceneDetect ContentDetector (threshold 20, min 4 frames) finds cuts. For each shot it skips the first/last ~1/6 of frames (dissolves, flashes) and scores each frame by sharpness (Laplacian variance), contrast (luma std) and closeness to mid exposure. Writes scene_XX.jpg + scenes.json.

sheet.py: builds a 3-column contact sheet from scenes.json. Run with any arg to use labels.json ({"1": "label", ...}) and write sheet.jpg; without, writes raw.jpg.

Deps: opencv-python-headless, scenedetect, pillow
