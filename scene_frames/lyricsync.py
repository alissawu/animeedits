"""Align an ordered English lyric reference to vocals, without deleting uncertain words.

Adapted from lyricsync v3. Acoustic inference is chunked; CTC alignment is global.
Imports of torch, torchaudio and separators are lazy, so help/reference retrieval
and caption rendering do not load models. See docs/lyricsync.md.
"""
import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

import lyrics

RATE = 16000
LOW_SCORE = 0.25
MODEL = "WAV2VEC2_ASR_BASE_960H"


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    ctc_score: float
    reference_index: int
    line_index: int
    low_score: bool


def tokens(text: str) -> list[str]:
    """Reject unsupported scripts rather than silently dropping foreign words."""
    text = text.replace("’", "'").replace("‘", "'")
    if any(c.isalnum() and c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in text):
        raise ValueError("alignment supports English A-Z and apostrophes; spell out numbers or use another aligner")
    return re.sub(r"[^A-Z' ]", " ", text.upper()).split()


def reference_tokens(lines: Sequence[str]) -> tuple[list[str], list[int]]:
    words, indices = [], []
    for i, line in enumerate(lines):
        ts = tokens(line)
        words.extend(ts)
        indices.extend([i] * len(ts))
    if not words:
        raise ValueError("the reference contains no supported words")
    return words, indices


def align(lines: Sequence[str], emission: Any, edges: Sequence[float], labels: Sequence[str]) -> list[Word]:
    """One CTC path for all reference words. Scores annotate; they never filter."""
    import torch
    import torchaudio
    expected, line_ids = reference_tokens(lines)
    dictionary = {c: i for i, c in enumerate(labels)}
    text = "|".join(expected)
    try:
        ids = [dictionary[c] for c in text]
    except KeyError as e:
        raise ValueError(f"unsupported model label {e}") from e
    if emission.ndim != 3 or emission.shape[0] != 1 or emission.shape[1] + 1 != len(edges):
        raise ValueError("expected emission [1, frames, labels] and frames+1 time edges")
    path, score = torchaudio.functional.forced_align(
        emission, torch.tensor([ids], dtype=torch.int64), blank=dictionary["-"])
    spans = torchaudio.functional.merge_tokens(path[0], score[0].exp(), blank=dictionary["-"])
    if [int(s.token) for s in spans] != ids:
        raise RuntimeError("CTC token sequence differs from reference")
    result: list[Word] = []
    k = 0
    for i, (word, line) in enumerate(zip(expected, line_ids)):
        ss = spans[k:k + len(word)]
        k += len(word) + 1
        start, end = float(edges[ss[0].start]), float(edges[ss[-1].end])
        conf = float(sum(float(s.score) for s in ss) / len(ss))
        result.append(Word(word.lower(), round(start, 6), round(end, 6),
                           round(conf, 6), i, line, conf < LOW_SCORE))
    validate(lines, result, float(edges[-1]))
    return result


def validate(lines: Sequence[str], words: Sequence[Word], duration: float) -> None:
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("invalid duration")
    expected, indices = reference_tokens(lines)
    if expected != [w.text.upper() for w in words] or indices != [w.line_index for w in words]:
        raise ValueError("reference words dropped, changed or reordered")
    if list(range(len(words))) != [w.reference_index for w in words]:
        raise ValueError("nonmonotonic reference indices")
    last = 0.0
    for w in words:
        if not all(math.isfinite(x) for x in (w.start, w.end, w.ctc_score)):
            raise ValueError("nonfinite word data")
        if not 0 <= w.start < w.end <= duration + 1e-5 or w.start < last - 1e-5:
            raise ValueError("invalid or overlapping serial word times")
        last = w.end


def infer(wave: NDArray[np.float32], model: Any) -> tuple[Any, list[float]]:
    """20-second acoustic inference chunks, then concatenate for ONE alignment."""
    import torch
    chunks = []
    edges: list[float] = []
    with torch.inference_mode():
        for start in range(0, wave.size, 20 * RATE):
            part = wave[start:start + 20 * RATE]
            # Conv feature encoder needs >=400 samples. Pad only the final tiny tail.
            data = np.pad(part, (0, max(0, 400 - part.size)))
            output, _ = model(torch.from_numpy(data.copy())[None])
            output = output.log_softmax(-1)[0]
            step = part.size / RATE / int(output.shape[0])
            chunks.append(output)
            edges.extend(start / RATE + i * step for i in range(int(output.shape[0])))
    if not chunks:
        raise ValueError("empty vocal audio")
    edges.append(wave.size / RATE)
    return torch.cat(chunks)[None], edges


def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, check=True, **kwargs)


def decode(file: Path) -> NDArray[np.float32]:
    proc = run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(file), "-map", "0:a:0",
                "-ar", str(RATE), "-ac", "1", "-f", "f32le", "-"], capture_output=True)
    return np.frombuffer(proc.stdout, dtype="<f4").copy()


def vocals(media: Path, supplied: Path | None, cache: Path) -> NDArray[np.float32]:
    if supplied is not None:
        return decode(supplied)
    import stems
    # Decode using the same clock as analyze_edit.py, then reuse its content-addressed cache.
    proc = run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-map", "0:a:0",
                "-ar", str(stems.SAMPLE_RATE), "-ac", "1", "-f", "f32le", "-"], capture_output=True)
    samples = len(proc.stdout) // 4
    separated = stems.load_or_separate(media, samples, cache)["vocals"]
    from scipy.signal import resample_poly
    return np.asarray(resample_poly(separated, 320, 441), dtype=np.float32)


def fetch_reference(title: str, artist: str, duration: float) -> dict[str, Any]:
    """Exact normalized identity only; no hardcoded songs or mashup guessing."""
    url = "https://lrclib.net/api/search?" + urlencode({"track_name": title, "artist_name": artist})
    with urlopen(Request(url, headers={"User-Agent": "animeedits-lyricsync/1.0"}), timeout=30) as response:
        candidates = json.load(response)
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    hits = [r for r in candidates if (r.get("plainLyrics") or r.get("syncedLyrics"))
            and norm(str(r.get("trackName", ""))) == norm(title)
            and norm(str(r.get("artistName", ""))) == norm(artist)]
    if not hits:
        raise ValueError("no exact LRCLIB match; supply --lyrics with the sung text")
    best: dict[str, Any] = min(hits, key=lambda r: abs(float(r.get("duration") or 1e9) - duration))
    if abs(float(best.get("duration") or 0) - duration) > 5:
        raise ValueError("reference duration differs by >5s; supply the ordered sung slice with --lyrics")
    return best


async def identify(media: Path, out: Path) -> list[dict[str, Any]]:
    """Opt-in external fingerprint lookup; results are candidates, not selected lyrics."""
    from shazamio import Shazam
    duration = len(decode(media)) / RATE
    results = []
    with tempfile.TemporaryDirectory(prefix="lyricsync-id-") as tmp:
        for offset in sorted({0.0, max(0.0, duration / 2 - 6), max(0.0, duration - 12)}):
            clip = Path(tmp) / "sample.wav"
            run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", str(offset), "-i", str(media),
                 "-t", "12", "-ar", str(RATE), "-ac", "1", str(clip)])
            try:
                data = await asyncio.wait_for(Shazam().recognize(str(clip)), 45)
                track = data.get("track", {})
                results.append({"offset": offset, "title": track.get("title"),
                                "artist": track.get("subtitle"), "url": track.get("url")})
            except Exception as e:
                results.append({"offset": offset, "error": str(e)})
            (out / "identification.json").write_text(json.dumps(results, indent=2))
    return results


def stamp(t: float, ass: bool = False) -> str:
    scale = 100 if ass else 1000
    h, n = divmod(round(max(0, t) * scale), 3600 * scale)
    m, n = divmod(n, 60 * scale)
    s, f = divmod(n, scale)
    return f"{h}:{m:02}:{s:02}.{f:02}" if ass else f"{h:02}:{m:02}:{s:02},{f:03}"


def captions(words: Sequence[Word], out: Path) -> None:
    """Show the reference line, highlight only the word actually under way."""
    groups: list[list[Word]] = []
    for word in words:
        if not groups or groups[-1][-1].line_index != word.line_index or len(groups[-1]) >= 7:
            groups.append([])
        groups[-1].append(word)
    srt: list[str] = []
    ass = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 1280", "PlayResY: 720", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
           "Style: Default,DejaVu Sans,60,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,5,60,60,40,1",
           "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for i, group in enumerate(groups, 1):
        start, end = group[0].start, group[-1].end
        srt.append(f"{i}\n{stamp(start)} --> {stamp(end)}\n{' '.join(w.text for w in group)}\n")
        boundaries = sorted({start, end, *(w.start for w in group), *(w.end for w in group)})
        for a, b in zip(boundaries, boundaries[1:]):
            if stamp(a, True) == stamp(b, True):
                continue
            mid = (a + b) / 2
            shown = []
            for w in group:
                text = w.text.replace("\\", "").replace("{", "").replace("}", "").replace("\n", " ")
                if w.start <= mid < w.end:
                    text = "{\\c&H66E8FF&}" + text + "{\\c&HFFFFFF&}"
                shown.append(text)
            ass.append(f"Dialogue: 0,{stamp(a, True)},{stamp(b, True)},Default,,0,0,0,,{' '.join(shown)}")
    (out / "lyrics.srt").write_text("\n".join(srt), encoding="utf-8")
    (out / "lyrics.ass").write_text("\n".join(ass) + "\n", encoding="utf-8")


def render_video(media: Path, out: Path, duration: float) -> None:
    run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
         "color=c=black:s=1280x720:r=24", "-i", str(media.resolve()), "-vf", "ass=lyrics.ass",
         "-map", "0:v:0", "-map", "1:a:0", "-t", str(duration), "-c:v", "libx264",
         "-preset", "ultrafast", "-crf", "22", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-movflags", "+faststart", "lyrics.mp4"], cwd=out)



def align_reference(media: Path, ref: lyrics.Reference, cache: Path, out: Path,
                    alignment_file: Path | None = None, video: bool = False) -> list[Word]:
    """Compute or validate alignment. Caption video is explicitly opt-in."""
    lines = [line.text for line in ref.lines]
    reference_tokens(lines)
    if alignment_file is not None:
        data = json.loads(alignment_file.read_text())
        if Path(data["media"]).resolve() != media.resolve():
            raise ValueError("alignment media differs from analyzed media")
        words = [Word(**w) for w in data["words"]]
        duration = len(decode(media)) / RATE
        validate(lines, words, duration)
    else:
        import torch
        import torchaudio
        torch.set_num_threads(3)
        wave = vocals(media, None, cache)
        duration = wave.size / RATE
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        emission, edges = infer(wave, bundle.get_model().eval())
        words = align(lines, emission, edges, bundle.get_labels())
        out.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "media": str(media.resolve()), "duration": duration,
                   "model": MODEL, "reference": asdict(ref), "words": [asdict(w) for w in words],
                   "warning": "Reference-conditioned timing; CTC scores are uncalibrated."}
        (out / "alignment.json").write_text(json.dumps(payload, indent=2))
    if video:
        out.mkdir(parents=True, exist_ok=True)
        captions(words, out)
        render_video(media, out, duration)
    return words


def main() -> None:
    import stems
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("media", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--lyrics", type=Path, help="ordered sung text/LRC; timestamps ignored for alignment")
    ap.add_argument("--lyrics-source")
    ap.add_argument("--title")
    ap.add_argument("--artist")
    ap.add_argument("--identify", action="store_true", help="opt-in Shazam lookup only; sends audio fingerprints")
    ap.add_argument("--vocals", type=Path, help="reuse an already separated vocal file on the same audio clock")
    ap.add_argument("--stem-cache", type=Path, default=stems.CACHE_DIR)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()
    if a.threads < 1:
        ap.error("--threads must be positive")
    if a.lyrics_source and not a.lyrics:
        ap.error("--lyrics-source needs --lyrics")
    media, out = a.media.resolve(), a.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        if a.identify:
            print(json.dumps(asyncio.run(identify(media, out)), indent=2))
            return
        duration = len(decode(media)) / RATE
        if a.lyrics:
            ref = lyrics.read(a.lyrics, a.lyrics_source)
        else:
            if not a.title or not a.artist:
                ap.error("supply --lyrics, or --title and --artist; --identify is a separate lookup")
            raw = fetch_reference(a.title, a.artist, duration)
            (out / "reference.json").write_text(json.dumps(raw, indent=2))
            file = out / "reference.lrc"
            file.write_text(raw.get("syncedLyrics") or raw["plainLyrics"], encoding="utf-8")
            ref = lyrics.read(file, f"https://lrclib.net/api/get/{raw['id']}")
        lines = [line.text for line in ref.lines]
        reference_tokens(lines)  # fail before separation or model downloads
        import torch
        import torchaudio
        torch.set_num_threads(a.threads)
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        wave = vocals(media, a.vocals, a.stem_cache)
        if abs(wave.size / RATE - duration) > .1:
            raise ValueError("--vocals must have the same duration and origin as the media")
        emission, edges = infer(wave, bundle.get_model().eval())
        words = align(lines, emission, edges, bundle.get_labels())
        payload = {"version": 1, "media": str(media), "duration": duration, "model": MODEL,
                   "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                   "reference": asdict(ref), "words": [asdict(w) for w in words],
                   "warning": "Reference-conditioned timing, not recognition or verified accuracy. CTC scores are not calibrated."}
        (out / "alignment.json").write_text(json.dumps(payload, indent=2))
        (out / "words.json").write_text(json.dumps(payload["words"], indent=2))
        (out / "low_score_words.json").write_text(json.dumps([asdict(w) for w in words if w.low_score], indent=2))
        captions(words, out)
        if a.video:
            render_video(media, out, duration)
        print(f"{len(words)} reference words retained; {sum(w.low_score for w in words)} low-score words. {out / 'alignment.json'}")
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError, ImportError) as e:
        ap.exit(2, f"lyricsync: {e}\n")


if __name__ == "__main__":
    main()
