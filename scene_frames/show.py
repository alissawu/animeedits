"""Index every video of a show from its manifest: show.py prepare|try|finish|report SHOW.

SHOW is a folder under shows/, or a path to one, holding show.toml with the show's cast notes and synopses.
- prepare: shots, stills, contact sheets, transcript and subtitles for each video, with no Claude. Then it lists
  the research that finish needs and hasn't been written yet.
- try: captions a few of one video's contact sheets with its group's cast notes as they are now, and prints them.
  Nothing is cached, so it costs a few captions and tests a draft of the notes before finish captions everything.
- finish: captions each video with its group's cast notes, reports the names the captions give, and places the
  shots in the story.
- report: the name report again, from the finished indexes, with no Claude.

Each video goes to OUT/NAME (OUT defaults to out/index), NAME being its file name without the extension, with the
options every edit shares. A few videos run at a time, and each logs to OUT/NAME/index.log. Every stage is cached, so
a rerun picks up where the last one stopped. See docs/indexing.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import tomllib
from typing import TYPE_CHECKING, Literal, TypedDict

# Only the light modules load up front: index.py brings in scenedetect and OpenCV, which only `try` needs.
try:
    from cast_notes import Names, canonical_characters, canonical_name, cast_names, name_key, read_context
except ImportError:  # imported as scene_frames.show
    from .cast_notes import (Names, canonical_characters,  # type: ignore[import-not-found,no-redef]
                             canonical_name, cast_names, name_key, read_context)

if TYPE_CHECKING:
    from index import IndexedShot
    from story import StoryScene

MANIFEST = "show.toml"
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
INDEX_SCRIPT = HERE / "index.py"
STORY_SCRIPT = HERE / "story.py"
# Every index of every show is made with these, so edits share one index per video. Changing one redoes detection.
SHARED_OPTIONS = ("--sample", "5", "--proxy", "480", "--faces", "anime")
DEFAULT_JOBS = 2  # detection and Whisper are CPU-heavy; much past 3 at a time only makes each slower
WARNING = "warning: "
PREPARED = ("dialogue from ", "subtitles: ", "subtitle timing ")  # index.py lines prepare's summary shows
POSSESSIVE = re.compile(r"['’]s\b")
TAIL_LINES = 12  # of a failed run's output, in the summary
NOISE = ("objc[",)  # duplicate-library notices from cv2 and PyAV on macOS
WRAP = 110

Stage = Literal["prepare", "finish"]


class ManifestError(ValueError):
    """Everything wrong with a manifest, one problem per line."""


@dataclass(frozen=True)
class Group:
    name: str
    cast: Path  # cast notes for every video in the group


@dataclass(frozen=True)
class Video:
    name: str  # the index's folder name: the video's file name without its extension
    group: Group
    video: Path
    synopsis: Path
    subs: Path | None
    lang: str | None  # None: index.py takes it from the audio track's language tag


@dataclass(frozen=True)
class Show:
    title: str
    root: Path  # the folder holding show.toml; cast and synopsis paths are relative to it
    groups: tuple[Group, ...]
    videos: tuple[Video, ...]


@dataclass
class Outcome:
    video: Video
    ok: bool
    seconds: float = 0.0
    lines: list[str] = field(default_factory=list)  # printed under the video in the summary


class Run(TypedDict):
    returncode: int
    output: list[str]


class NameCount(TypedDict):
    name: str
    shots: int  # shots whose captions give the name
    unsure: int  # of those, the shots that say "possibly"
    in_cast: bool
    in_synopsis: bool


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text} is not a positive integer")
    return value


def relative_path(value: object, where: str, key: str, problems: list[str]) -> PurePosixPath | None:
    """A manifest path: relative, with forward slashes, and never climbing out with "..", so that the manifest means
    the same thing in every checkout."""
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{where}: {key} must be a path")
        return None
    path = PurePosixPath(value.strip())
    if path.is_absolute() or "\\" in value or ".." in path.parts:
        problems.append(f"{where}: {key} {value!r} must be relative, with / and no ..")
        return None
    return path


def check_keys(table: dict[str, object], allowed: set[str], required: set[str], where: str,
               problems: list[str]) -> None:
    for key in sorted(set(table) - allowed):
        problems.append(f"{where}: unknown key {key!r}; expected {', '.join(sorted(allowed))}")
    for key in sorted(required - set(table)):
        problems.append(f"{where}: missing {key}")


def load_show(root: Path, media: Path) -> Show:
    """Read and check root/show.toml. Paths in it are relative: videos and subtitles to media, cast notes and synopses
    to root. Files needn't exist yet; the commands check what each one needs. Raises ManifestError."""
    path = root / MANIFEST
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise ManifestError(f"can't read {path}: {e}") from None
    except tomllib.TOMLDecodeError as e:
        raise ManifestError(f"{path} is not valid TOML: {e}") from None
    problems: list[str] = []
    check_keys(data, {"title", "groups"}, {"title", "groups"}, MANIFEST, problems)
    title = data.get("title")
    if "title" in data and (not isinstance(title, str) or not title.strip()):
        problems.append(f"{MANIFEST}: title must be text")
    raw_groups = data.get("groups", [])
    if not isinstance(raw_groups, list) or not raw_groups:
        problems.append(f"{MANIFEST}: needs at least one [[groups]] table")
        raw_groups = []
    groups: list[Group] = []
    videos: list[Video] = []
    for g, raw in enumerate(raw_groups, 1):
        where = f"group {g}"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be a [[groups]] table")
            continue
        check_keys(raw, {"name", "cast", "videos"}, {"name", "cast", "videos"}, where, problems)
        name = raw.get("name")
        if isinstance(name, str) and name.strip():
            where = f"group {name.strip()!r}"
            if any(other.name == name.strip() for other in groups):
                problems.append(f"{where}: another group has this name")
        elif "name" in raw:
            problems.append(f"{where}: name must be text")
        cast = relative_path(raw.get("cast"), where, "cast", problems) if "cast" in raw else None
        group = Group(name=name.strip() if isinstance(name, str) else where, cast=root / (cast or ""))
        groups.append(group)
        raw_videos = raw.get("videos", [])
        if not isinstance(raw_videos, list) or not raw_videos:
            problems.append(f"{where}: needs at least one [[groups.videos]] table")
            continue
        for v, entry in enumerate(raw_videos, 1):
            at = f"{where}, video {v}"
            if not isinstance(entry, dict):
                problems.append(f"{at}: must be a [[groups.videos]] table")
                continue
            check_keys(entry, {"video", "synopsis", "subs", "lang"}, {"video", "synopsis"}, at, problems)
            file = relative_path(entry.get("video"), at, "video", problems) if "video" in entry else None
            synopsis = relative_path(entry.get("synopsis"), at, "synopsis", problems) if "synopsis" in entry else None
            subs = relative_path(entry.get("subs"), at, "subs", problems) if "subs" in entry else None
            lang = entry.get("lang")
            if lang is not None and (not isinstance(lang, str) or not lang.strip()):
                problems.append(f"{at}: lang must be a Whisper language code, such as ja")
                lang = None
            if file is None or synopsis is None:
                continue
            if any(other.name == file.stem for other in videos):
                problems.append(f"{at}: another video is also named {file.stem}, and both would index to "
                                f"out/index/{file.stem}")
                continue
            videos.append(Video(name=file.stem, group=group, video=media / file, synopsis=root / synopsis,
                                subs=media / subs if subs else None, lang=lang.strip() if lang else None))
    if problems:
        raise ManifestError("\n".join(f"{path}: {p}" for p in problems))
    assert isinstance(title, str)
    return Show(title=title.strip(), root=root, groups=tuple(groups), videos=tuple(videos))


def select(show: Show, only: Sequence[str]) -> list[Video]:
    """The videos named, by video name or group name, in manifest order; every video when none are."""
    if not only:
        return list(show.videos)
    unknown = [o for o in only if not any(o in (v.name, v.group.name) for v in show.videos)]
    if unknown:
        choices = "; ".join([*(g.name for g in show.groups), *(v.name for v in show.videos)])
        raise ManifestError(f"--only {', '.join(unknown)}: not a group or video in {show.root / MANIFEST}. "
                            f"Choose from: {choices}")
    return [v for v in show.videos if v.name in only or v.group.name in only]


def index_command(video: Video, out: Path, stage: Stage) -> list[str]:
    cmd = [sys.executable, str(INDEX_SCRIPT), str(video.video), str(out), *SHARED_OPTIONS]
    if video.lang:
        cmd += ["--lang", video.lang]
    if video.subs:
        cmd += ["--subs", str(video.subs)]
    return cmd + (["--context", str(video.group.cast)] if stage == "finish" else ["--no-captions"])


def story_command(video: Video, out: Path) -> list[str]:
    return [sys.executable, str(STORY_SCRIPT), str(out), "--context", str(video.group.cast),
            "--synopsis", str(video.synopsis)]


def run_logged(cmd: Sequence[str], log: Path) -> Run:
    """Run a command, appending its output to log, and return the output."""
    log.parent.mkdir(parents=True, exist_ok=True)
    output: list[str] = []
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n== {datetime.now():%Y-%m-%d %H:%M:%S} {shlex.join(cmd)}\n")
        f.flush()
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                              env={**os.environ, "PYTHONUNBUFFERED": "1"}) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                output.append(line.rstrip("\n"))
        return Run(returncode=proc.returncode, output=output)


def meaningful(output: Sequence[str]) -> list[str]:
    return [line for line in output if line.strip() and not line.startswith(NOISE)]


def result_lines(name: str, run: Run, log: Path, keep: tuple[str, ...] = ()) -> tuple[bool, list[str]]:
    """Whether a run worked, and what the summary shows of it: lines starting with one of keep, its warnings and its
    last line, or the end of a failure."""
    lines = meaningful(run["output"])
    if run["returncode"] != 0:
        return False, [f"{name} exited with {run['returncode']}; the end of its output (all of it is in {log}):",
                       *(f"  {line}" for line in lines[-TAIL_LINES:])]
    return True, [line for line in lines[:-1] if line.startswith((*keep, WARNING))] + lines[-1:]


def missing(paths: Sequence[tuple[str, Path | None]]) -> list[str]:
    return [f"no {what} at {path}" for what, path in paths if path is not None and not path.is_file()]


def research_problems(video: Video) -> list[str]:
    """What keeps a video's research from being ready for finish: cast notes without a `## Cast` section or `### `
    entries, or a synopsis that is missing or empty."""
    problems = missing([("cast notes", video.group.cast), ("synopsis", video.synopsis)])
    if video.group.cast.is_file():
        try:
            _, cast = read_context(video.group.cast)
        except (OSError, ValueError) as e:
            problems.append(str(e))
        else:
            if not cast_names(cast):
                problems.append(f"{video.group.cast} has no `### ` entries yet")
    if video.synopsis.is_file() and not video.synopsis.read_text(encoding="utf-8").strip():
        problems.append(f"{video.synopsis} is empty")
    return problems


def prepare(video: Video, out_root: Path) -> Outcome:
    problems = missing([("video", video.video), ("subtitle file", video.subs)])
    if problems:
        return Outcome(video, False, lines=problems)
    out = out_root / video.name
    ok, lines = result_lines("index.py", run_logged(index_command(video, out, "prepare"), out / "index.log"),
                             out / "index.log", PREPARED)
    return Outcome(video, ok, lines=lines)


def finish(video: Video, out_root: Path) -> Outcome:
    problems = missing([("video", video.video), ("subtitle file", video.subs)]) + research_problems(video)
    if problems:
        return Outcome(video, False, lines=problems)
    out, log = out_root / video.name, out_root / video.name / "index.log"
    ok, lines = result_lines("index.py", run_logged(index_command(video, out, "finish"), log), log)
    if not ok:
        return Outcome(video, False, lines=lines)
    lines += video_report(video, out)
    story_ok, story_lines = result_lines("story.py", run_logged(story_command(video, out), log), log)
    return Outcome(video, story_ok, lines=lines + story_lines + (story_corrections(out) if story_ok else []))


def mentioned(name: str, names: Names, synopsis: str) -> bool:
    """Whether the synopsis mentions a character, by one of their names or aliases in full, or by a word of one that
    no other character's has, such as "Gojo" for Satoru Gojo but not "Fushiguro" when the notes have two.
    synopsis is a name_key of the text."""
    name = canonical_name(name, names)
    own = [name_key(n) for n, who in names.items() if who == name] or [name_key(name)]
    padded = f" {synopsis} "
    if any(key and f" {key} " in padded for key in own):
        return True
    others = {word for n, who in names.items() if who != name for word in name_key(n).split()}
    words = set(synopsis.split())
    return any(word in words for key in own for word in key.split() if len(word) > 2 and word not in others)


def name_report(shots: Sequence[IndexedShot], names: Names, synopsis: str) -> list[NameCount]:
    """How many shots name each character, most named first, and whether the cast notes and synopsis know them."""
    count: Counter[str] = Counter()
    unsure: Counter[str] = Counter()
    for shot in shots:
        for c in shot["characters"]:
            count[c["name"]] += 1
            unsure[c["name"]] += not c["certain"]
    known = {name_key(n) for n in names}
    text = f"{name_key(synopsis)} {name_key(POSSESSIVE.sub('', synopsis))}"  # "Rika's soul" mentions Rika
    return [NameCount(name=name, shots=n, unsure=unsure[name], in_cast=name_key(name) in known,
                      in_synopsis=mentioned(name, names, text))
            for name, n in sorted(count.items(), key=lambda item: (-item[1], item[0]))]


def wrapped(heading: str, items: Sequence[str]) -> list[str]:
    return textwrap.wrap(f"{heading}: {', '.join(items)}", WRAP, subsequent_indent="    ")


def report_lines(report: Sequence[NameCount], shots: int) -> list[str]:
    def label(c: NameCount) -> str:
        return f'{c["name"]} {c["shots"]}' + (f' ({c["unsure"]} possibly)' if c["unsure"] else "")

    named = [label(c) for c in report]
    lines = wrapped(f"names in the captions of {shots} shots", named or ["none"])
    if strangers := [label(c) for c in report if not c["in_cast"]]:
        lines += wrapped("not in the cast notes", strangers)
    if unmentioned := [label(c) for c in report if not c["in_synopsis"]]:
        lines += wrapped("not in the synopsis", unmentioned)
    return lines


def video_report(video: Video, out: Path) -> list[str]:
    """The name report for one finished index, or why there isn't one."""
    try:
        shots: list[IndexedShot] = json.loads((out / "shots.json").read_text(encoding="utf-8"))
        _, cast = read_context(video.group.cast)
        synopsis = video.synopsis.read_text(encoding="utf-8")
    except (OSError, ValueError) as e:
        return [f"no name report: {e}"]
    return report_lines(name_report(shots, cast_names(cast), synopsis), len(shots))


def story_corrections(out: Path) -> list[str]:
    """What the story pass changed in a finished index: each swap of names, with how many shots it changed, most
    first. A swap to someone the cast notes don't have usually means they're missing from the notes."""
    try:
        story: list[StoryScene] = json.loads((out / "story.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"no story corrections: {e}"]
    swaps: Counter[str] = Counter()
    for scene in story:
        for shot in scene.get("shots", []):
            if "caption_characters" in shot:
                before = [c["name"] for c in shot["caption_characters"]]
                after = [c["name"] for c in shot["characters"]]
                removed = ", ".join(n for n in before if n not in after)
                added = ", ".join(n for n in after if n not in before) or "nobody"
                swaps[f"{removed} → {added}"] += 1
    return wrapped("story corrections", [f"{swap} {n}" for swap, n in swaps.most_common()]) if swaps else []


def try_notes(video: Video, out: Path, sheets: Sequence[int]) -> list[str]:
    """Caption some of a prepared video's sheets with its group's cast notes as they are now, the way finish would
    (one claude -p call per sheet), and return a line per shot. Raises ValueError for something missing."""
    try:
        from index import CAPTION_MODEL, SHEET_SIZE, caption_sheet_cli, detected_shots
    except ImportError:  # imported as scene_frames.show
        from .index import (CAPTION_MODEL, SHEET_SIZE,  # type: ignore[import-not-found,no-redef]
                            caption_sheet_cli, detected_shots)
    shots = detected_shots(out)
    count = -(-len(shots) // SHEET_SIZE)
    if bad := [n for n in sheets if not 1 <= n <= count]:
        raise ValueError(f"{video.name} has sheets 1-{count}, not {', '.join(map(str, bad))}")
    title, cast = read_context(video.group.cast)
    names = cast_names(cast)

    def caption(n: int) -> list[str]:
        group = shots[(n - 1) * SHEET_SIZE:n * SHEET_SIZE]
        sheet = out / "sheets" / f"sheet_{n:04d}.jpg"
        if not sheet.is_file():
            raise ValueError(f"no {sheet}; run prepare first")
        captions = caption_sheet_cli(group, sheet, CAPTION_MODEL, title, cast)
        lines = [f"sheet {n} ({sheet})"]
        for shot in group:
            c = captions[str(shot["i"])]
            named = "; ".join(ch["name"] if ch["certain"] else f'possibly {ch["name"]}'
                              for ch in canonical_characters(c["characters"], names))
            lines.append(f'  {shot["i"]:04d} {shot["start"]:.1f}-{shot["end"]:.1f}s [{named or "nobody named"}] '
                         f'{c["caption"]}')
        return lines

    with ThreadPoolExecutor(max_workers=len(sheets)) as pool:
        return [line for lines in pool.map(caption, sheets) for line in lines]


def research_missing(videos: Sequence[Video]) -> list[str]:
    """What finish needs from the research and doesn't have yet, once per problem."""
    return list(dict.fromkeys(problem for v in videos for problem in research_problems(v)))


def name_lines(video: Video, out: Path, name: str) -> list[str]:
    """Every shot whose caption names someone, with its keyframe and what the story pass made of the name."""
    try:
        shots: list[IndexedShot] = json.loads((out / "shots.json").read_text(encoding="utf-8"))
        _, cast = read_context(video.group.cast)
    except (OSError, ValueError) as e:
        return [f"no shots to search: {e}"]
    # The names the captions give count as well as the notes', so "Nobara" finds a cameo the notes leave out.
    target = canonical_name(name, {**{c["name"]: c["name"] for shot in shots for c in shot["characters"]},
                                   **cast_names(cast)})
    try:
        story: list[StoryScene] = json.loads((out / "story.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        story = []
    corrected = {shot["id"]: shot["characters"] for scene in story for shot in scene.get("shots", [])
                 if "caption_characters" in shot}
    lines: list[str] = []
    for shot in shots:
        named = next((c for c in shot["characters"] if c["name"] == target), None)
        if named is None:
            continue
        story_note = ""
        if shot["id"] in corrected and not any(c["name"] == target for c in corrected[shot["id"]]):
            story_note = f' [story: {", ".join(c["name"] for c in corrected[shot["id"]]) or "nobody"}]'
        lines.append(f'{shot["id"]:04d} {shot["start"]:.1f}-{shot["end"]:.1f}s '
                     f'{"" if named["certain"] else "possibly "}{target}{story_note}  {out / shot["keyframe"]["file"]}')
    return [f"{len(lines)} shots name {target}"] + [f"  {line}" for line in lines]


def clock(seconds: float) -> str:
    minutes, secs = divmod(round(seconds), 60)
    return f"{minutes}m{secs:02d}s"


def run_all(videos: Sequence[Video], work: Callable[[Video], Outcome], jobs: int) -> list[Outcome]:
    """Run work on every video, jobs at a time, printing each as it ends; outcomes in manifest order."""
    def timed(video: Video) -> Outcome:
        print(f"started {video.name}", flush=True)
        start = time.monotonic()
        try:
            outcome = work(video)
        except OSError as e:  # a missing interpreter, a full disk: report it with the rest
            outcome = Outcome(video, False, lines=[f"couldn't run: {e}"])
        outcome.seconds = time.monotonic() - start
        return outcome

    outcomes: dict[str, Outcome] = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(timed, v) for v in videos]
        for done, future in enumerate(as_completed(futures), 1):
            outcome = future.result()
            outcomes[outcome.video.name] = outcome
            print(f"[{done}/{len(videos)}] {outcome.video.name}: {'done' if outcome.ok else 'FAILED'} "
                  f"in {clock(outcome.seconds)}", flush=True)
    return [outcomes[v.name] for v in videos]


def print_summary(command: str, show: Show, outcomes: Sequence[Outcome], seconds: float) -> None:
    failed = sum(not o.ok for o in outcomes)
    print(f"\n{command} {show.root.name}: {len(outcomes) - failed} of {len(outcomes)} videos done"
          + (f", {failed} failed" if failed else "") + f", {clock(seconds)}")
    for o in outcomes:
        print(f"\n{'ok' if o.ok else 'FAILED'}  {o.video.name}  ({o.video.group.name}, {clock(o.seconds)})")
        for line in o.lines:
            print(f"  {line}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["prepare", "try", "finish", "report"])
    ap.add_argument("show", help="a folder in shows/, such as jujutsu_kaisen, or a path to a folder with show.toml")
    ap.add_argument("--only", nargs="+", default=[], metavar="NAME",
                    help="just these groups or videos (a video's name is its file name without the extension); "
                         "try takes one video")
    ap.add_argument("--sheets", nargs="+", type=positive_int, default=[], metavar="N",
                    help="for try: the contact sheets to caption, by number (sheets/sheet_NNNN.jpg)")
    ap.add_argument("--name", help="for report: list every shot whose caption gives this name, with its keyframe")
    ap.add_argument("--jobs", type=positive_int, default=DEFAULT_JOBS, help="videos indexed at once")
    ap.add_argument("--media", type=Path, default=REPO / "media", help="the folder the manifest's video paths start in")
    ap.add_argument("--out", type=Path, default=REPO / "out" / "index", help="where the indexes go")
    ap.add_argument("--dry-run", action="store_true", help="print the commands instead of running them")
    a = ap.parse_args()
    root = Path(a.show) if os.sep in a.show or "/" in a.show or a.show.startswith(".") else REPO / "shows" / a.show
    media, out = a.media.resolve(), a.out.resolve()
    if a.command in ("prepare", "finish") and not media.is_dir():
        ap.error(f"no media folder at {media}. Videos live in the main checkout's media/; run from there or pass "
                 "--media DIR")
    try:
        show = load_show(root.resolve(), media)
        videos = select(show, a.only)
    except ManifestError as e:
        raise SystemExit(str(e)) from None

    if a.dry_run:
        for v in videos:
            commands = ([index_command(v, out / v.name, "prepare")] if a.command == "prepare" else
                        [index_command(v, out / v.name, "finish"), story_command(v, out / v.name)])
            for cmd in commands if a.command in ("prepare", "finish") else []:
                print(shlex.join(cmd))
        return
    if a.command == "try":
        if len(videos) != 1 or not a.sheets:
            ap.error("try needs one video, --only NAME, and the sheets to caption, --sheets N [N ...]")
        if shutil.which("claude") is None:
            raise SystemExit("try captions through Claude Code's `claude` command, which isn't on PATH")
        try:
            lines = try_notes(videos[0], out / videos[0].name, a.sheets)
        except (OSError, ValueError, RuntimeError) as e:
            raise SystemExit(f"try {videos[0].name}: {e}") from None
        print("\n".join(lines))
        return
    if a.command == "report":
        for v in videos:
            print(f"\n{v.name}  ({v.group.name})")
            lines = (name_lines(v, out / v.name, a.name) if a.name
                     else video_report(v, out / v.name) + story_corrections(out / v.name))
            for line in lines:
                print(f"  {line}")
        return

    print(f"{a.command} {show.title}: {len(videos)} videos, {min(a.jobs, len(videos))} at a time. Each logs to "
          f"{out}/NAME/index.log", flush=True)
    start = time.monotonic()
    work = prepare if a.command == "prepare" else finish
    try:
        outcomes = run_all(videos, lambda v: work(v, out), a.jobs)
    except KeyboardInterrupt:
        raise SystemExit("\nstopped. Finished stages are cached; rerun the same command to resume.") from None
    print_summary(a.command, show, outcomes, time.monotonic() - start)
    if a.command == "prepare":
        todo = research_missing(videos)
        print("\nresearch to write before finish (docs/indexing.md#research):" if todo
              else "\nall research is written; next: finish")
        for line in todo:
            print(f"  {line}")
    if any(not o.ok for o in outcomes):
        sys.exit(1)


if __name__ == "__main__":
    main()
