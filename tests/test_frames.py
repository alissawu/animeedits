"""frames.py on tiny generated videos. Needs ffmpeg on PATH."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
import frames as fr  # noqa: E402

W, H = 160, 90
BITS = 8


def encode(frames: NDArray[np.uint8], rate: str, file: Path) -> None:
    n, h, w, _ = frames.shape
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", rate,
                    "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p", str(file)],
                   input=frames.tobytes(), check=True)


def counter_frames(count: int) -> NDArray[np.uint8]:
    """Frame n shows n in binary: BITS full-height stripes, most significant first, white for 1."""
    out = np.zeros((count, H, W, 3), np.uint8)
    stripe = W // BITS
    for n in range(count):
        for b in range(BITS):
            if n >> (BITS - 1 - b) & 1:
                out[n, :, b * stripe:(b + 1) * stripe] = 255
    return out


def read_counter(image: Image.Image) -> int:
    rgb = np.asarray(image.convert("RGB"), np.float32).mean(2)
    stripe = image.width / BITS
    value = 0
    for b in range(BITS):
        x = int((b + 0.5) * stripe)
        value = value * 2 + int(rgb[:, x - 1:x + 2].mean() > 128)
    return value


def scene_frames() -> NDArray[np.uint8]:
    """4 s at 24 fps: a gray shot where a box moves only in frames 20-32, a cut at 48 to a still blue shot
    with white flash frames 72-73."""
    yy, xx = np.mgrid[0:H, 0:W]
    room = np.stack([40 + xx * 0.6, 50 + yy * 0.8, 60 + 60 * ((xx // 16 + yy // 16) % 2)], -1).astype(np.uint8)
    sky = np.stack([200 - yy * 1.5, 90 + 70 * ((yy // 10) % 2), 220 - xx * 0.5], -1).astype(np.uint8)
    out = np.zeros((96, H, W, 3), np.uint8)
    for n in range(48):
        out[n] = room
        x = 16 + round(96 * min(1.0, max(0.0, (n - 20) / 12)))
        out[n, 33:57, x:x + 24] = 255
    for n in range(48, 96):
        out[n] = sky
    out[72:74] = 255
    return out


def write_wav(file: Path, seconds: float) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=440:d={seconds}:sample_rate=44100",
                    str(file)], check=True)


@unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg not on PATH")
class FramesTest(unittest.TestCase):
    tmp: tempfile.TemporaryDirectory[str]
    root: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        encode(counter_frames(96), "24000/1001", cls.root / "ntsc.mp4")
        encode(counter_frames(120), "24", cls.root / "counter.mp4")
        encode(scene_frames(), "24", cls.root / "scene.mp4")
        write_wav(cls.root / "song.wav", 8)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    # ---- timing ----

    def test_stamped_times_are_the_frames_shown(self) -> None:
        src = fr.open_source(self.root / "ntsc.mp4")
        fps = 24000 / 1001
        times, frames = fr.decode(src, 1.0, 2.5, (W, H))
        self.assertEqual(read_counter(Image.fromarray(frames[0])), round(1.0 * fps))
        for t, frame in zip(times, frames):
            n = read_counter(Image.fromarray(frame))
            self.assertAlmostEqual(t, n / fps, delta=1e-3)
        self.assertEqual(len(times), round(2.5 * fps) - round(1.0 * fps))

    def test_second_pass_returns_the_measured_frames(self) -> None:
        src = fr.open_source(self.root / "ntsc.mp4")
        m = fr.measure(src, 0.5, 3.0, None)
        picks = [0, 5, 6, 30, len(m.times) - 1]
        tiles = fr.grab(m, picks, 160)
        for i in picks:
            self.assertEqual(read_counter(tiles[i]), round(m.times[i] * 24000 / 1001))

    def test_parse_time(self) -> None:
        self.assertEqual(fr.parse_time("1255.5"), 1255.5)
        self.assertEqual(fr.parse_time("20:55.5"), 1255.5)
        self.assertEqual(fr.parse_time("0:20:55.5"), 1255.5)
        for bad in ("abc", "1:2:3:4", "-3", "nan"):
            with self.assertRaises(Exception):
                fr.parse_time(bad)

    # ---- errors ----

    def test_bad_spans_and_files(self) -> None:
        src = fr.open_source(self.root / "scene.mp4")
        with self.assertRaisesRegex(fr.FramesError, "before OUT"):
            fr.measure(src, 2.0, 2.0, None)
        with self.assertRaisesRegex(fr.FramesError, "past the end"):
            fr.measure(src, 3.0, 9.0, None)
        with self.assertRaisesRegex(fr.FramesError, "past the end"):
            fr.measure(src, 5.0, 6.0, None)
        junk = self.root / "junk.mp4"
        junk.write_text("not a video")
        with self.assertRaisesRegex(fr.FramesError, "cannot read"):
            fr.open_source(junk)
        with self.assertRaisesRegex(fr.FramesError, "not found"):
            fr.open_source(self.root / "missing.mp4")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            code = fr.main([str(self.root / "scene.mp4"), "3", "1", str(self.root / "x.jpg"), "--no-index"])
        self.assertEqual(code, 1)
        self.assertIn("before OUT", err.getvalue())

    # ---- scene board ----

    def test_detects_shots_and_flashes_without_an_index(self) -> None:
        m = fr.measure(fr.open_source(self.root / "scene.mp4"), 0.0, 4.0, None)
        shots = [s.start for s in m.segments if s.kind == "shot"]
        self.assertEqual(shots, [0, 48])
        self.assertEqual([i for i, label in enumerate(m.labels) if label == "white"], [72, 73])

    def test_middle_frames_follow_the_action(self) -> None:
        m = fr.measure(fr.open_source(self.root / "scene.mp4"), 0.0, 4.0, None)
        change = fr.masked_change(m, {0, 48})
        moving = fr.pick_middles(change, 0, 48)
        self.assertGreater(len(moving), 1)
        self.assertTrue(all(20 <= i <= 33 for i in moving), moving)
        still = fr.pick_middles(change, 48, 96)
        self.assertEqual(len(still), 1)

    def test_scene_board_with_an_index(self) -> None:
        index_dir = self.root / "index" / "scene"
        index_dir.mkdir(parents=True, exist_ok=True)
        line = {"start": 2.5, "end": 3.0, "text": " hi there",
                "words": [{"start": 2.5, "end": 2.7, "word": " hi"}, {"start": 2.7, "end": 3.0, "word": " there"}]}
        shots: list[dict[str, Any]] = [
            {"id": 7, "start": 0.0, "end": 2.0, "dialogue": [], "subtitles": []},
            {"id": 8, "start": 2.0, "end": 4.0, "dialogue": [line],
             "subtitles": [{"start": 2.4, "end": 3.1, "text": "Hi there."}]},
        ]
        (index_dir / "shots.json").write_text(json.dumps(shots))
        index = fr.load_index(index_dir)
        m = fr.measure(fr.open_source(self.root / "scene.mp4"), 1.0, 4.0, None)
        rows = fr.board_rows(m, index)
        self.assertEqual([r.title.split(" · ")[0] for r in rows], ["shot 7", "shot 8"])
        self.assertEqual((rows[0].first, rows[1].first), (0, 24))
        self.assertEqual([w.text for w in rows[1].words], ["hi", "there"])
        self.assertEqual([c.text for c in rows[1].cues], ["Hi there."])
        out = self.root / "board.jpg"
        fr.scene_board(m, index, out)
        with Image.open(out) as sheet:
            self.assertGreater(sheet.height, 200)

    def test_crop_preview_is_vertical(self) -> None:
        m = fr.measure(fr.open_source(self.root / "scene.mp4"), 0.0, 1.0, 0.0)
        self.assertLess(m.small.shape[2], m.small.shape[1])
        tile = fr.grab(m, [0], 160)[0]
        self.assertEqual(tile.size, fr.fit(9 / 16, 160))
        # x=0 keeps the left edge, where the box sits before it moves.
        self.assertGreater(np.asarray(tile, np.float32).mean(), 90)

    # ---- shot detail ----

    def test_dense_merges_holds_and_finds_the_move(self) -> None:
        m = fr.measure(fr.open_source(self.root / "scene.mp4"), 0.0, 2.0, None)
        tiles = fr.dense_tiles(m, 6.0, set())
        self.assertLess(len(tiles), 12)
        self.assertGreater(tiles[0].samples, 1)  # still until frame 20
        self.assertGreater(tiles[-1].samples, 1)  # still after frame 32
        moving = fr.stretches(fr.masked_change(m, set()), m.src.fps)
        self.assertEqual(len(moving), 1)
        self.assertEqual((moving[0].start, moving[0].last), (21, 32))
        out = self.root / "dense.jpg"
        fr.shot_detail(m, None, 6.0, out)
        self.assertTrue(out.is_file())

    def test_dense_refuses_too_many_tiles(self) -> None:
        m = fr.measure(fr.open_source(self.root / "counter.mp4"), 0.0, 5.0, None)
        with self.assertRaisesRegex(fr.FramesError, "lower --fps"):
            fr.shot_detail(m, None, 24.0, self.root / "too_many.jpg")

    # ---- cut pairs ----

    def write_edl(self) -> Path:
        edl = {
            "output": {"width": 64, "height": 36, "fps": 24},
            "song": {"source": "song.wav", "src_in": 2.0, "src_out": 4.0},
            "clips": [
                {"source": "counter.mp4", "src_in": 1.0, "timeline_in": 0.0, "timeline_out": 1.0},
                {"source": "counter.mp4", "src_in": 3.0, "timeline_in": 1.0, "timeline_out": 2.0, "speed": 2.0,
                 "transition": {"type": "flash", "frames": 2}},
            ],
        }
        file = self.root / "edl.json"
        file.write_text(json.dumps(edl))
        song_map = {"media": {"path": str(self.root / "song.wav")},
                    "audio": {"beats": [2.5, 2.98, 3.5], "drops": []},
                    "words": [{"start": 3.1, "end": 3.4, "probability": 0.9}]}
        (self.root / "map").mkdir(exist_ok=True)
        (self.root / "map" / "timeline.json").write_text(json.dumps(song_map))
        (self.root / "song.lrc").write_text("[00:01.00] first line\n[00:03.10] <00:03.10>second <00:03.30>line\n")
        return file

    def test_cut_pair_frames_match_the_render(self) -> None:
        file = self.write_edl()
        edl = fr.load_edl(file)
        probes, problems = fr.render.check_sources(edl, self.root)
        self.assertEqual(problems, [])
        with tempfile.TemporaryDirectory() as work:
            out_frames = fr.clip_frames_rendered(edl, self.root, probes, 0, 21, 3, None, Path(work))
            in_frames = fr.clip_frames_rendered(edl, self.root, probes, 1, 0, 3, None, Path(work))
        self.assertEqual([read_counter(f.image) for f in out_frames], [45, 46, 47])
        self.assertEqual([round(f.source * 24) for f in out_frames], [45, 46, 47])
        self.assertEqual([read_counter(f.image) for f in in_frames], [72, 74, 76])
        self.assertEqual([round(f.source * 24) for f in in_frames], [72, 74, 76])
        self.assertEqual([round(f.timeline * 24) for f in in_frames], [24, 25, 26])

    def test_cut_notes_use_the_song_map_on_the_timeline(self) -> None:
        file = self.write_edl()
        edl = fr.load_edl(file)
        song_map = fr.find_song_map(edl, self.root)
        assert song_map is not None
        self.assertEqual(song_map.beats, [0.5, 0.98, 1.5])
        lyrics = fr.load_lyrics(self.root / "song.lrc", edl)
        self.assertEqual([text for _, text in lyrics], ["first line", "second line"])
        self.assertAlmostEqual(lyrics[0][0], -1.0)
        self.assertAlmostEqual(lyrics[1][0], 1.1)
        notes = fr.cut_notes(edl, 2, song_map, lyrics)
        self.assertIn("beats -0.50 -0.02 +0.50 s", notes)
        self.assertIn("word starts +0.10 s", notes)
        self.assertIn("next reference lyric +0.10 s: second line", notes)
        self.assertIn("recognized [text unknown]", notes)  # legacy maps stay readable

    def test_recognition_and_published_reference_follow_song_speed(self) -> None:
        file = self.write_edl()
        data = json.loads(file.read_text())
        data['song'] = {'source': 'song.wav', 'src_in': 2, 'src_out': 6, 'speed': 2}
        file.write_text(json.dumps(data))
        map_file = self.root / 'map/timeline.json'
        sm = json.loads(map_file.read_text())
        sm['words'][0]['text'] = 'misheard'
        sm['words'][0]['probability'] = .35
        sm['lyrics'] = {'path': '/reference.lrc', 'source': 'published source',
                        'lines': [{'start': 3.2, 'text': 'different words'}, {'start': None, 'text': 'unplaced'}]}
        map_file.write_text(json.dumps(sm))
        edl = fr.load_edl(file)
        song_map = fr.load_song_map(map_file, edl, self.root)
        assert song_map is not None and song_map.lyrics is not None
        self.assertAlmostEqual(song_map.words[0].start, .55)
        self.assertAlmostEqual(song_map.words[0].end, .7)
        self.assertEqual(song_map.words[0].text, 'misheard')
        lines = fr.reference_lines(song_map.lyrics, edl, song_map.of_song)
        self.assertEqual(len(lines), 1)  # never attach untimed reference words to a cut
        self.assertAlmostEqual(lines[0][0], .6)
        notes = fr.cut_notes(edl, 2, song_map, lines)
        self.assertIn('recognized "misheard"', notes)
        self.assertIn('p=0.35', notes)
        self.assertIn('reference lyric at the cut', notes)
        self.assertIn('different words', notes)

    def test_review_hides_intent_until_requested(self) -> None:
        file = self.write_edl()
        data = json.loads(file.read_text())
        data['clips'][1].update(why='HIDDEN REASON', sync='word', sync_points=[
            {'t': .5, 'target_time': 1.48, 'label': 'NAMED INTERNAL ACTION', 'kind': 'word'}])
        data['audio'] = [{'source': 'song.wav', 'src_in': 0, 'src_out': 1, 'timeline_in': 1,
                          'why': 'HIDDEN AUDIO REASON'}]
        file.write_text(json.dumps(data))
        edl = fr.load_edl(file)
        plain = fr.cut_notes(edl, 2, None, [])
        self.assertNotIn('REASON', plain)
        self.assertNotIn('declared', plain)
        annotated = fr.cut_notes(edl, 2, None, [], show_intent=True)
        self.assertIn('HIDDEN REASON', annotated)
        self.assertIn('HIDDEN AUDIO REASON', annotated)
        self.assertIn('NAMED INTERNAL ACTION', annotated)
        self.assertIn('picture 1.500 s', annotated)
        self.assertIn('offset +0.020 s', annotated)
        # Exercise the actual sheet path as well, without OCR assertions.
        original = fr.cut_notes
        for show in (False, True):
            with mock.patch.object(fr, 'cut_notes', wraps=original) as notes:
                fr.cut_pairs(file, self.root / f'intent-{show}.jpg', '2', None, None, None, show_intent=show)
            self.assertEqual(notes.call_args_list[0].kwargs['show_intent'], show)

    def test_song_times_follow_spliced_pieces(self) -> None:
        file = self.write_edl()
        data = json.loads(file.read_text())
        data["song"] = {"source": "song.wav", "pieces": [{"src_in": 2.0, "src_out": 3.0},
                                                         {"src_in": 5.0, "src_out": 6.0}]}
        file.write_text(json.dumps(data))
        edl = fr.load_edl(file)
        self.assertAlmostEqual(fr._song_time(edl, 1.5), -0.5)   # before the first piece
        self.assertAlmostEqual(fr._song_time(edl, 2.5), 0.5)
        self.assertIsNone(fr._song_time(edl, 4.0))              # cut out between the pieces
        self.assertAlmostEqual(fr._song_time(edl, 5.5), 1.5)
        self.assertAlmostEqual(fr._song_time(edl, 6.5), 2.5)    # after the last piece

    def test_cut_pairs_sheet_and_cut_numbers(self) -> None:
        file = self.write_edl()
        out = self.root / "pairs.jpg"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fr.main(["--edl", str(file), str(out), "--cut", "2"]), 0)
        self.assertTrue(out.is_file())
        self.assertEqual(fr.parse_cuts(None, 3), [2, 3])
        self.assertEqual(fr.parse_cuts("1-2,3", 3), [1, 2, 3])
        with self.assertRaisesRegex(fr.FramesError, "numbered 1 to 3"):
            fr.parse_cuts("4", 3)
        with self.assertRaisesRegex(fr.FramesError, "already vertical"):
            vertical = json.loads(file.read_text())
            vertical["output"] = {"width": 36, "height": 64, "fps": 24}
            (self.root / "vertical.json").write_text(json.dumps(vertical))
            fr.cut_pairs(self.root / "vertical.json", out, "2", None, None, 0.5)


if __name__ == "__main__":
    unittest.main()
