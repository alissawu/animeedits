"""Cast notes: the markdown that says how to recognize a film's, season's or arc's characters.

index.py gives the notes' `## Cast` section to the captioner, and the `### ` headings in it are the names
captions should use. See docs/indexing.md#recipe-cast-notes for how to write them.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import re
from typing import TypedDict
import unicodedata

HEADING = re.compile(r"(#{1,2})\s+(.+?)[\s#]*")  # markdown # and ## headings
CAST_HEADING = re.compile(r"###\s+(.+?)[\s#]*")
PARENTHETICAL = re.compile(r"\s*\([^()]*\)\s*$")  # "Kenjaku (in Suguru Geto's body)" names Kenjaku


class Character(TypedDict):
    name: str
    certain: bool


def read_context(path: Path) -> tuple[str | None, str]:
    """The first `# ` heading and the `## Cast` section of a context file."""
    title: str | None = None
    cast: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = HEADING.fullmatch(line)
        if heading is None:
            if cast is not None:
                cast.append(line)
            continue
        if cast is not None:
            break
        if len(heading[1]) == 1 and title is None:
            title = heading[2]
        elif len(heading[1]) == 2 and heading[2].casefold() == "cast":
            cast = []
    section = "\n".join(cast or []).strip()
    if not section:
        raise ValueError(f"{path} has no '## Cast' section; see docs/pipeline.md for the format")
    return title, section


def cast_names(cast: str) -> list[str]:
    """The name of each `### ` entry in a cast section, without a trailing (parenthetical)."""
    names: list[str] = []
    for line in cast.splitlines():
        if heading := CAST_HEADING.fullmatch(line.strip()):
            name = PARENTHETICAL.sub("", heading[1]).strip() or heading[1]
            if name not in names:
                names.append(name)
    return names


def name_key(name: str) -> str:
    """A name for comparing: casefolded, without accents or apostrophes, other punctuation as spaces. "Zen'in",
    "Zenin" and "ZENIN" have the same key, and so do "Tamamo-no-Mae" and "Tamamo no Mae"."""
    text = "".join(ch for ch in unicodedata.normalize("NFKD", name.casefold()) if not unicodedata.combining(ch))
    return " ".join(re.sub(r"\W+", " ", re.sub(r"['’`]", "", text)).split())


def canonical_name(name: str, names: Sequence[str]) -> str:
    """The cast notes' spelling of a caption's name. A name matches a cast name with the same key, or else the one
    cast name whose words include all of its words, so "Gojo" and "Gojo Satoru" become "Satoru Gojo" while
    "Fushiguro" stays as it is when the notes have two. A trailing (parenthetical) is ignored. Other names are kept."""
    for candidate in (PARENTHETICAL.sub("", name), name):
        key = name_key(candidate)
        if not key:
            continue
        exact = [n for n in names if name_key(n) == key]
        if exact:
            return exact[0]
        words = set(key.split())
        partial = [n for n in names if words <= set(name_key(n).split())]
        if len(partial) == 1:
            return partial[0]
    return name


def merge_characters(characters: Sequence[Character]) -> list[Character]:
    """One entry per name, in order, certain if any entry was."""
    merged: dict[str, Character] = {}
    for c in characters:
        if c["name"] in merged:
            merged[c["name"]]["certain"] = merged[c["name"]]["certain"] or c["certain"]
        else:
            merged[c["name"]] = Character(name=c["name"], certain=c["certain"])
    return list(merged.values())


def canonical_characters(characters: Sequence[Character], names: Sequence[str]) -> list[Character]:
    return merge_characters([Character(name=canonical_name(c["name"], names), certain=c["certain"])
                             for c in characters])
