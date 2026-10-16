"""Score cached reference-edit alignments and compare CSV word hits to all words.

No separation/inference/downloads. Baseline samples uniform words within each edit,
weighted by that edit's matched cut count (not random times or causal evidence).
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import stems
import word_stress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    rows = list(csv.DictReader((root / "all_cuts.csv").open()))
    results = []
    pools = []
    hits = []
    skipped = []
    details = []
    for path in sorted((root / "analysis").glob("*/timeline.json")):
        data = json.loads(path.read_text())
        aligned = data.get("aligned_lyrics")
        if not aligned:
            skipped.append(dict(edit=path.parent.name, reason="no checked-reference alignment"))
            continue
        media = Path(data["media"]["path"])
        stem = stems.stem_dir(media) / "vocals.flac"
        if not stem.exists():
            skipped.append(dict(edit=path.parent.name, reason="no cached vocal stem"))
            continue
        signal, rate = sf.read(stem, dtype="float32", always_2d=True)
        word_stress.annotate(aligned["lines"], signal.mean(axis=1), rate, data["media"]["audio_offset"])
        aligned["stress_method"] = word_stress.METHOD
        aligned["stress_threshold"] = word_stress.THRESHOLD
        words = [w for line in aligned["lines"] for w in line["words"]]
        selected = []
        for row in rows:
            if row["edit"] != path.parent.name or row["specific_lyric_word_match"] != "True":
                continue
            # Reproduce summarize.py ordering: start then end, in reference order.
            time = float(row["time_s"])
            word, edge = min(((w, e) for w in words for e in ("start", "end")),
                             key=lambda x: abs(x[0][x[1]] - time))
            if abs(word[edge] - time) > .08001:
                raise ValueError(f"CSV and aligned timeline disagree: {path.parent.name} {time}")
            target = f'{word["text"]} {edge}; uncertain={word.get("low_score", word.get("uncertain", False))}'
            if target != row["specific_lyric_word_target"]:
                raise ValueError(f"CSV target changed: {target!r} != {row['specific_lyric_word_target']!r}")
            selected.append(word)
            details.append(dict(edit=path.parent.name, cut=time, edge=edge, **word))
        baseline = float(np.mean([w["stressed"] for w in words]))
        observed = sum(w["stressed"] for w in selected)
        results.append(dict(edit=path.parent.name, words=len(words), word_cut_hits=len(selected),
                            high_stress_hits=observed,
                            hit_rate=observed / len(selected) if selected else None,
                            baseline_rate=baseline, expected_hits=len(selected) * baseline,
                            uncertain_hits=sum(w.get("uncertain", w.get("low_score", False)) for w in selected)))
        pools.append((words, selected))
        hits.extend(selected)
        path.write_text(json.dumps(data, indent=2) + "\n")
    n = len(hits)
    observed = sum(w["stressed"] for w in hits)
    expected = sum(r["expected_hits"] for r in results)
    rng = np.random.default_rng(20261016)
    null = np.zeros(20000)
    null_mean = np.zeros(20000)
    for words, selected in pools:
        draws = rng.integers(0, len(words), size=(20000, len(selected)))
        null += np.array([w["stressed"] for w in words])[draws].sum(axis=1)
        null_mean += np.array([w["stress"] for w in words])[draws].sum(axis=1)
    # An additional uncertainty-filtered comparison uses the same per-edit weighting.
    certain = []
    for words, selected in pools:
        clean = lambda w: not w.get("uncertain", w.get("low_score", False))
        ws, hs = list(filter(clean, words)), list(filter(clean, selected))
        if ws and hs:
            certain.append(dict(hits=len(hs), high=sum(w["stressed"] for w in hs),
                                expected=len(hs) * float(np.mean([w["stressed"] for w in ws]))))
    report = dict(method=word_stress.METHOD, threshold=word_stress.THRESHOLD,
                  baseline="uniform words within edit, weighted by matched cut counts; repeats retained",
                  word_cut_hits=n, high_stress_hits=observed, hit_rate=observed / n if n else None,
                  expected_hits=expected, baseline_rate=expected / n if n else None,
                  enrichment=observed / expected if expected else None,
                  monte_carlo_one_sided_p=float((1 + (null >= observed).sum()) / (len(null) + 1)),
                  observed_mean_stress=float(np.mean([w["stress"] for w in hits])) if hits else None,
                  baseline_mean_stress=float(null_mean.mean() / n) if n else None,
                  certain_only=certain, edits=results, skipped=skipped,
                  cautions=["Exploratory heuristic; threshold not calibrated to human emphasis.",
                            "CSV includes word starts AND ends, cut detector and CTC errors.",
                            "Uniform-word null does not control rhythm, line position, or boundary coverage.",
                            "Reference set has only five checked-reference alignments."])
    (root / "word_stress_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "word_stress_hits.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
