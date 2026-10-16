import dataclasses
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scene_frames"))
import music  # noqa: E402

RATE = 22050
BEAT = 0.5


def tone(seconds: float, hz: float, level: float) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (level * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def pattern_bar(seed: int, level: float = 0.3) -> np.ndarray:
    """Two bars (eight beats) of noise bursts at seeded positions and pitches: a loop to repeat."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(8 * BEAT * RATE), np.float32)
    for k in range(16):
        if rng.random() < 0.6:
            start = int(k * BEAT / 2 * RATE)
            n = int(0.12 * RATE)
            hz = rng.choice([300.0, 600.0, 1200.0, 2400.0])
            t = np.arange(n) / RATE
            out[start:start + n] += (level * np.sin(2 * np.pi * hz * t) * np.exp(-t * 20)).astype(np.float32)
    return out


class LevelTests(unittest.TestCase):
    def test_block_db(self):
        y = np.concatenate([tone(1, 440, 0.5), np.zeros(RATE, np.float32)])
        db = music.block_db(y, RATE)
        self.assertEqual(db.size, 8)
        self.assertAlmostEqual(db[0], 20 * np.log10(0.5 / np.sqrt(2)), delta=0.1)
        self.assertEqual(db[-1], music.FLOOR_DB)

    def test_debounce_absorbs_short_runs(self):
        mask = np.array([1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
        self.assertEqual(music.debounce(mask, 2).tolist(), [True] * 8 + [False] * 4)

    def test_snap(self):
        self.assertEqual(music.snap(1.05, [0.5, 1.0, 1.5]), 1.0)
        self.assertEqual(music.snap(1.2, [0.5, 1.0, 1.5]), 1.2)
        self.assertEqual(music.snap(1.2, []), 1.2)

    def test_refine_edge_finds_the_crossing(self):
        fine = np.full(100, -60.0)
        fine[43:] = -10.0  # sound from 2.15 s
        self.assertAlmostEqual(music.refine_edge(fine, 2.0, -30.0, entering=True), 2.15)
        self.assertAlmostEqual(music.refine_edge(fine[::-1].copy(), 2.75, -30.0, entering=False), 2.85)


def stems_with(**parts: np.ndarray) -> dict[str, np.ndarray]:
    n = max(p.size for p in parts.values())
    out = {}
    for name in music.STEMS:
        p = parts.get(name, np.zeros(n, np.float32))
        out[name] = np.concatenate([p, np.zeros(n - p.size, np.float32)])
    return out


class ActivityTests(unittest.TestCase):
    def test_entry_exit_and_band(self):
        # Bass plays 6-14 s under a voice that sings throughout; a quiet guitar plays throughout.
        n = 20 * RATE
        bass = np.zeros(n, np.float32)
        bass[6 * RATE:14 * RATE] = tone(8, 55, 0.4)
        stems = stems_with(bass=bass, vocals=tone(20, 330, 0.1), other=tone(20, 880, 0.01))
        mix = sum(stems.values())
        beats = list(np.arange(0, 20, BEAT))
        levels = music.stem_levels(stems, mix, RATE)
        acts, events = music.activity(stems, levels, RATE, beats)
        by = {a.stem: a for a in acts}
        self.assertFalse(by["drums"].present)
        self.assertEqual(by["bass"].runs, [(6.0, 14.0)])
        self.assertEqual(by["vocals"].runs, [(0.0, 20.0)])
        kinds = {(e.part, e.change, e.time) for e in events}
        self.assertIn(("bass", "enters", 6.0), kinds)
        self.assertIn(("bass", "exits", 14.0), kinds)
        band = [e for e in events if e.part == "band"]
        self.assertEqual([(e.change, e.voice) for e in band], [("up", True), ("down", True)])
        found = music.turns(events, levels)
        self.assertEqual([t.time for t in found], [6.0, 14.0])
        self.assertGreater(found[0].low_db, 20)
        self.assertLess(found[1].low_db, -20)
        self.assertEqual([t.time for t in music.drops(found)], [6.0])
        exits = [e for e in events if e.change == "exits"]
        self.assertEqual([(e.part, e.final) for e in exits], [("bass", True)])  # it never comes back

    def test_faint_stem_has_runs_but_no_events(self):
        stems = stems_with(other=tone(12, 440, 0.5), drums=np.concatenate([np.zeros(4 * RATE, np.float32),
                                                                          tone(8, 2000, 0.02)]))
        mix = sum(stems.values())
        levels = music.stem_levels(stems, mix, RATE)
        acts, events = music.activity(stems, levels, RATE, [])
        drums = {a.stem: a for a in acts}["drums"]
        self.assertTrue(drums.faint)
        self.assertEqual(drums.runs, [(4.0, 12.0)])
        self.assertEqual([e for e in events if e.part == "drums"], [])

    def test_steps_inside_a_run(self):
        drums = np.concatenate([tone(8, 3000, 0.05), tone(8, 3000, 0.4)])
        stems = stems_with(drums=drums)
        levels = music.stem_levels(stems, drums, RATE)
        _, events = music.activity(stems, levels, RATE, [])
        steps = [e for e in events if e.part == "drums" and e.change == "up"]
        self.assertEqual(len(steps), 1)
        self.assertAlmostEqual(steps[0].time, 8.0, delta=0.06)
        self.assertAlmostEqual(steps[0].db or 0, 18, delta=1)


class TurnTests(unittest.TestCase):
    def test_drops_keep_entries_near_the_largest(self):
        def turn(t: float, low: float, part: str = "bass", change: music.Change = "enters") -> music.Turn:
            return music.Turn(t, [music.StemEvent(t, part, change)], 0.0, low)
        found = [turn(10, 30), turn(40, 36), turn(60, 20), turn(80, 40, "drums", "exits"),
                 turn(90, 35, "drums", "up")]
        self.assertEqual([t.time for t in music.drops(found)], [10, 40])
        self.assertEqual(music.drops([turn(10, 4.0)]), [])

    def test_grouping_and_time(self):
        events = [music.StemEvent(10.0, "vocals", "up", 7.0), music.StemEvent(10.6, "bass", "enters"),
                  music.StemEvent(11.2, "drums", "enters"), music.StemEvent(12.0, "other", "down", -8.0)]
        levels = music.Levels(np.zeros(100), {}, np.zeros(100), np.zeros(100))
        found = music.turns(events, levels)
        self.assertEqual([(t.time, len(t.events)) for t in found], [(10.6, 3), (12.0, 1)])

    def test_change_db(self):
        series = np.r_[np.full(40, -30.0), np.full(40, -10.0)]
        self.assertEqual(music.change_db(series, 10.0), 20.0)
        self.assertIsNone(music.change_db(series, 1.0))


class StopTests(unittest.TestCase):
    def test_stop_then_hit(self):
        y = np.concatenate([tone(12, 220, 0.3), np.zeros(int(1.2 * RATE), np.float32), tone(6, 220, 0.3)])
        found = music.stops(y, RATE, [13.2])
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].start, 12.0, delta=0.06)
        self.assertEqual(found[0].hit, 13.2)
        self.assertTrue(found[0].silent)

    def test_routine_dips_are_not_stops(self):
        # A 20 dB dip every 2 s is what this music does, so none is unusual; the same dip alone is a stop.
        # (Dips to near-silence still count through the fade rule: the trial saw that on sparse guitar.)
        bar = np.concatenate([tone(1.6, 220, 0.3), tone(0.4, 220, 0.03)])
        self.assertEqual(music.stops(np.tile(bar, 12), RATE, []), [])
        lone = np.concatenate([tone(12, 220, 0.3), tone(0.4, 220, 0.03), tone(6, 220, 0.3)])
        found = music.stops(lone, RATE, [])
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0].start, 12.0, delta=0.1)


class LoopTests(unittest.TestCase):
    def test_new_pattern_breaks_the_loop(self):
        beats = list(np.arange(0, 32, BEAT))
        other = np.concatenate([np.tile(pattern_bar(1), 4), np.tile(pattern_bar(2), 4)])  # new loop at 16 s
        drums = np.tile(pattern_bar(3), 8)                                              # one loop throughout
        stems = stems_with(other=other, drums=drums, bass=np.tile(tone(4, 55, 0.3), 8))
        levels = music.stem_levels(stems, sum(stems.values()), RATE)
        acts, _ = music.activity(stems, levels, RATE, beats)
        breaks = music.loop_breaks(stems, RATE, beats, acts)
        self.assertEqual([(e.part, e.time) for e in breaks], [("other", 16.0)])
        assert breaks[0].loop is not None
        self.assertGreater(breaks[0].loop[0], 0.9)


class PitchTests(unittest.TestCase):
    def test_bass_roots_per_bar(self):
        beats = list(np.arange(0, 16.01, BEAT))
        # A for two bars from 1 s, then C: the root change lands on beat 2 of the grid, so bars start there.
        bass = np.concatenate([tone(1, 55, 0.3), tone(4, 55, 0.3), tone(4, 65.41, 0.3), tone(4, 55, 0.3),
                               tone(3, 65.41, 0.3)])
        harmony = music.bass_roots(bass, RATE, beats, [(0.0, 16.0)])
        self.assertEqual(harmony.phase, 2)
        self.assertEqual([(b.start, b.root) for b in harmony.bars],
                         [(0.0, "A"), (1.0, "A"), (3.0, "A"), (5.0, "C"), (7.0, "C"), (9.0, "A"), (11.0, "A"),
                          (13.0, "C"), (15.0, "C")])
        self.assertEqual(music.bass_roots(bass, RATE, beats, []).bars, [])
        # Outside the bass stem's runs a bar has no root, however clear the pitch.
        partial = music.bass_roots(bass, RATE, beats, [(5.0, 9.0)])
        outside = [b.root for b in partial.bars if b.end <= 5 or b.start >= 9]
        self.assertTrue(outside and all(r is None for r in outside), partial.bars)
        self.assertIn("C", [b.root for b in partial.bars if 5 <= b.start < 9])

    def test_voice_sections_and_shift(self):
        t = np.arange(12 * RATE) / RATE
        hz = np.where(t < 6, np.where((t * 2).astype(int) % 2 == 0, 220.0, 261.6),
                      np.where((t * 2).astype(int) % 2 == 0, 329.6, 392.0))
        voice = (0.1 * np.sin(2 * np.pi * np.cumsum(hz) / RATE)).astype(np.float32)
        sections = music.voice_sections(voice, RATE, [0.0, 6.0, 12.0], [(0.0, 12.0)])
        self.assertTrue(all(s.reliable for s in sections))
        self.assertEqual([s.pitch_low for s in sections], ["A3", "E4"])
        shifts = music.voice_shifts(sections)
        self.assertEqual(len(shifts), 1)
        self.assertAlmostEqual(shifts[0].semitones, 7.0, delta=0.6)

    def test_voice_guard(self):
        # A constant low pitch is bass bleed, not singing; a silent stem has no pitch at all.
        bleed = tone(6, 92.5, 0.1)
        self.assertFalse(music.voice_sections(bleed, RATE, [0.0, 6.0], [(0.0, 6.0)])[0].reliable)
        quiet = tone(6, 220, 0.0005)  # -69 dBFS
        self.assertIsNone(music.voice_sections(quiet, RATE, [0.0, 6.0], [(0.0, 6.0)])[0].pitch)
        # Outside the vocal stem's runs nothing counts, however clear the pitch.
        self.assertIsNone(music.voice_sections(tone(6, 220, 0.1), RATE, [0.0, 6.0], [(0.0, 0.2)])[0].pitch)


class TempoTests(unittest.TestCase):
    def test_alternative(self):
        self.assertEqual(music.tempo_alternative(184.6), 92.3)
        self.assertEqual(music.tempo_alternative(58.7), 117.4)
        self.assertIsNone(music.tempo_alternative(123.0))


class JsonTests(unittest.TestCase):
    def test_round_trip_and_shift(self):
        turn = music.Turn(4.0, [music.StemEvent(4.0, "band", "up", 12.0, voice=True),
                                music.StemEvent(4.5, "other", "loop_break", loop=(0.9, 0.7))], 12.0, None)
        act = music.StemActivity("bass", -12.0, -32.0, True, False, [(4.0, 9.5)])
        harmony = music.Harmony(1, [music.Bar(0.5, 2.5, "A"), music.Bar(2.5, 4.5, None)])
        for obj in (turn, act, harmony):
            data = json.loads(json.dumps(dataclasses.asdict(obj)))
            self.assertEqual(music.from_json(type(obj), data), obj)
        later = music.shift_times(turn, 1.0)
        self.assertEqual([e.time for e in later.events], [5.0, 5.5])
        self.assertEqual(later.time, 5.0)
        self.assertEqual(music.shift_times(act, 1.0).runs, [(5.0, 10.5)])
        self.assertEqual(music.shift_times(harmony, 1.0).bars[1], music.Bar(3.5, 5.5, None))


if __name__ == "__main__":
    unittest.main()
