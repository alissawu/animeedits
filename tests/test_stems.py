from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scene_frames"))
import stems  # noqa: E402


def fake_stems(n: int) -> stems.Stems:
    rng = np.random.default_rng(0)
    return {name: (0.1 * rng.standard_normal(n)).astype(np.float32) for name in stems.STEMS}


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.song = self.root / "song.mp3"
        self.song.write_bytes(b"not really an mp3")

    def tearDown(self):
        self.tmp.cleanup()

    def test_key_follows_content(self):
        copy = self.root / "renamed.mp3"
        copy.write_bytes(self.song.read_bytes())
        self.assertEqual(stems.cache_key(self.song), stems.cache_key(copy))
        copy.write_bytes(b"different")
        self.assertNotEqual(stems.cache_key(self.song), stems.cache_key(copy))

    def test_round_trip_trims_and_pads(self):
        made = fake_stems(1000)
        folder = stems.stem_dir(self.song, self.root / "cache")
        stems.write_stems(folder, made, self.song)
        self.assertEqual([p.name for p in folder.parent.iterdir()], [folder.name])  # no temp folder left
        back = stems.read_stems(folder, 1200)
        assert back is not None
        for name in stems.STEMS:
            self.assertEqual(back[name].size, 1200)
            np.testing.assert_allclose(back[name][:1000], made[name], atol=1e-6)
            self.assertFalse(back[name][1000:].any())
        self.assertEqual(stems.read_stems(folder, 500)["bass"].size, 500)

    def test_incomplete_entry_is_a_miss(self):
        folder = stems.stem_dir(self.song, self.root / "cache")
        stems.write_stems(folder, fake_stems(100), self.song)
        (folder / "vocals.flac").unlink()
        self.assertIsNone(stems.read_stems(folder, 100))
        with self.assertRaises(ValueError):
            stems.write_stems(folder, {"drums": np.zeros(10, np.float32)}, self.song)

    def test_separates_once(self):
        calls = []

        def separate(media: Path) -> stems.Stems:
            calls.append(media)
            return fake_stems(800)
        cache = self.root / "cache"
        first = stems.load_or_separate(self.song, 800, cache, separate=separate)
        second = stems.load_or_separate(self.song, 800, cache, separate=separate)
        self.assertEqual(len(calls), 1)
        np.testing.assert_allclose(first["drums"], second["drums"], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
