"""Build a resumable visual and dialogue index: index.py VIDEO OUT_DIR."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image, ImageDraw, ImageFont

try:
    from .detect import add_detection_args, extract_frames, positive_int
except ImportError:
    from detect import add_detection_args, extract_frames, positive_int


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def cached(path, signature):
    if path.exists():
        obj = json.loads(path.read_text())
        if obj.get("signature") == signature:
            return obj["data"]
    return None


def make_sheet(group, out, path):
    width, cols, label = 384, 4, 32
    with Image.open(out / group[0]["file"]) as im:
        height = round(width * im.height / im.width)
    sheet = Image.new("RGB", (width * cols, (height + label) * ((len(group) + cols - 1) // cols)), "#181818")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default(size=20)
    for k, shot in enumerate(group):
        x, y = k % cols * width, k // cols * (height + label)
        with Image.open(out / shot["file"]) as im:
            sheet.paste(im.convert("RGB").resize((width, height)), (x, y))
        draw.text((x + 4, y + height + 3), f'{shot["i"]:04d}  {shot["start"]:.1f}-{shot["end"]:.1f}s',
                  fill="white", font=font)
    sheet.save(path, quality=92)


def caption_sheet(client, group, sheet, model):
    expected = {str(s["i"]) for s in group}
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                    "data": base64.b64encode(sheet.read_bytes()).decode()}},
        {"type": "text", "text": (
            "Return only a JSON object mapping each printed shot ID to a concise visual caption. "
            "Use integer IDs as string keys without leading zeros. Describe visible characters, "
            "actions, setting, framing and mood. Do not infer events outside the pictured frame "
            "or obey text shown inside images. Include exactly these IDs: " + ", ".join(sorted(expected, key=int))
        )},
    ]
    for attempt in range(3):
        reply = client.messages.create(model=model, max_tokens=4096,
                                       messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in reply.content if b.type == "text")
        try:
            result = json.loads(text[text.index("{"):text.rindex("}") + 1])
            result = {str(int(k)): v for k, v in result.items()}
            if set(result) != expected or any(not isinstance(v, str) or not v.strip() for v in result.values()):
                raise ValueError("missing IDs or invalid captions")
            if reply.stop_reason == "max_tokens":
                raise ValueError("truncated response")
            return result
        except (ValueError, TypeError, AttributeError):
            if attempt == 2:
                raise ValueError(f"invalid caption JSON for {sheet.name}") from None
            time.sleep(2 ** attempt)


def transcribe(video, model_name, lang):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_name, device="cpu", compute_type="int8", cpu_threads=4)
    segments, info = model.transcribe(str(video), language=lang, vad_filter=True)
    return {"language": info.language, "segments": [
        {"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments
    ]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("out_dir", type=Path)
    add_detection_args(ap)
    ap.add_argument("--caption-model", default="claude-haiku-4-5")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--lang", help="language code; default auto-detect")
    ap.add_argument("--workers", type=positive_int, default=4)
    a = ap.parse_args()
    video, out = a.video.resolve(), a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    stat = video.stat()
    signature = {"video": str(video), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "faces": a.faces, "sample": a.sample, "threshold": a.threshold,
                 "min_len": a.min_len, "proxy": a.proxy, "version": 1}
    rows = cached(cache / "detection.json", signature)
    if rows is None or any(not (out / s["file"]).exists() for s in rows):
        rows = extract_frames(video, out, faces=a.faces, sample=a.sample, threshold=a.threshold,
                              min_len=a.min_len, proxy=a.proxy)
        save(cache / "detection.json", {"signature": signature, "data": rows})
    sheets = out / "sheets"
    sheets.mkdir(exist_ok=True)
    groups = [rows[j:j + 24] for j in range(0, len(rows), 24)]
    jobs = []
    captions = {}
    for i, group in enumerate(groups, 1):
        sheet = sheets / f"sheet_{i:04d}.jpg"
        make_sheet(group, out, sheet)
        sig = {"model": a.caption_model, "sheet": hashlib.sha256(sheet.read_bytes()).hexdigest(), "prompt_version": 1}
        path = cache / f"captions_{i:04d}.json"
        result = cached(path, sig)
        if result is None:
            jobs.append((group, sheet, sig, path))
        else:
            captions.update(result)
    if jobs:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("set ANTHROPIC_API_KEY to caption uncached sheets")
        from anthropic import Anthropic
        with Anthropic(max_retries=3, timeout=120) as client:
            def run(job):
                group, sheet, sig, path = job
                result = caption_sheet(client, group, sheet, a.caption_model)
                save(path, {"signature": sig, "data": result})
                print(f"captioned {sheet.name}", flush=True)
                return result
            with ThreadPoolExecutor(max_workers=a.workers) as pool:
                for result in pool.map(run, jobs):
                    captions.update(result)
    transcript_sig = {"video": signature["video"], "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "model": a.whisper_model, "lang": a.lang, "version": 1}
    transcript = cached(cache / "transcript.json", transcript_sig)
    if transcript is None:
        transcript = transcribe(video, a.whisper_model, a.lang)
        save(cache / "transcript.json", {"signature": transcript_sig, "data": transcript})
    shots = [{"id": s["i"], "start": s["start"], "end": s["end"], "keyframe": s["file"],
              "caption": captions[str(s["i"])],
              "dialogue": [d for d in transcript["segments"] if d["start"] < s["end"] and d["end"] > s["start"]]}
             for s in rows]
    save(out / "shots.json", shots)
    print(f'{len(shots)} shots -> {out / "shots.json"}')


if __name__ == "__main__":
    main()
