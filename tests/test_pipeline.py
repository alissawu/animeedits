import importlib.util
import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scene_frames"))
from detect import usable_range, positive_int
spec = importlib.util.spec_from_file_location("shot_index", Path(__file__).resolve().parents[1] / "scene_frames/index.py")
index = importlib.util.module_from_spec(spec)
spec.loader.exec_module(index)


class PipelineTests(unittest.TestCase):
    def test_usable_range(self):
        self.assertEqual(usable_range(0, 4), (0, 4))
        self.assertEqual(usable_range(0, 60), (10, 50))

    def test_positive_int(self):
        with self.assertRaises(Exception):
            positive_int("0")

    def test_cache_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            index.save(path, {"signature": {"model": "a"}, "data": []})
            self.assertEqual(index.cached(path, {"model": "a"}), [])
            self.assertIsNone(index.cached(path, {"model": "b"}))

    def test_caption_json(self):
        reply = SimpleNamespace(content=[SimpleNamespace(type="text", text='{"0001":"A face"}')],
                                stop_reason="end_turn")
        client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: reply))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.jpg"
            path.write_bytes(b"image")
            self.assertEqual(index.caption_sheet(client, [{"i": 1}], path, "model"), {"1": "A face"})


if __name__ == "__main__":
    unittest.main()
