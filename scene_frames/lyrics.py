"""Published lyric references, kept separate from machine recognition.

LRC timestamps describe lines in the supplied reference. They are not inferred word
alignments. Plain text is retained with unknown times rather than assigned timings.
"""
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Line:
    start: float | None
    text: str


@dataclass(frozen=True)
class Reference:
    path: str
    source: str
    lines: list[Line]
    alignment: str = "unverified reference line times; no word alignment"


STAMP = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
WORD_STAMP = re.compile(r"<\d+:\d+(?:\.\d+)?>")
METADATA = re.compile(r"^\[[a-zA-Z]+:.*\]$")
OFFSET = re.compile(r"\[offset:([+-]?\d+)\]", re.IGNORECASE)


def read(file: Path, source: str | None = None) -> Reference:
    raw = file.read_text(encoding="utf-8-sig")
    offset_match = OFFSET.search(raw)
    offset = int(offset_match[1]) / 1000 if offset_match else 0.0
    lines: list[Line] = []
    for row in raw.splitlines():
        stamps = STAMP.findall(row)
        if not stamps and METADATA.fullmatch(row.strip()):
            continue
        text = WORD_STAMP.sub("", STAMP.sub("", row)).strip()
        if stamps:
            lines.extend(Line(int(m) * 60 + float(s) + offset, text) for m, s in stamps)
        elif text:
            lines.append(Line(None, text))
    return Reference(str(file.resolve()), source or str(file.resolve()), lines)
