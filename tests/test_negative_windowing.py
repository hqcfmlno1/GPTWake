"""Window extraction must respect source ranges, labels, and bounded reads."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import soundfile as sf

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/02_positive_recording_augmentation.ipynb"
)


class NegativeWindowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="gptwake-windows-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="window-test"\n')
        self.cells = [
            c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code"
        ]
        self.ns = {}
        with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
            exec(self.cells[0], self.ns)
            exec(self.cells[1], self.ns)

    def extract(self, *args, **kwargs):
        self.assertTrue(
            callable(self.ns.get("extract_negative_windows")), "Missing interval extraction"
        )
        return self.ns["extract_negative_windows"](*args, **kwargs)

    def source(self, kind="negative", seconds=15, rate=16000, stereo=False):
        audio = np.full(round(seconds * rate), 0.1, dtype=np.float32)
        if stereo:
            audio = np.column_stack([audio, audio * 3])
        return self.ns["save_recording"](kind, audio, rate)

    def metadata(self, path):
        return json.loads(path.with_suffix(".json").read_text())

    def open_panel(self):
        with patch.object(self.ns["sd"], "query_devices", return_value=[]):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.cells[2], self.ns)
        self.assertTrue("negative_selection" in self.ns, "Missing negative range slider")

    def test_simple_extraction_uses_slider_without_copying_to_ranges(self):
        source = self.source(seconds=12)
        self.open_panel()
        self.ns["negative_source"].value = str(source)
        self.ns["negative_ranges"].value = "0,4"
        self.ns["negative_selection"].value = (8, 11)
        self.ns["negative_count"].value = 3
        self.ns["extract_button"].click()
        files = list((self.root / "data/negative/processed").rglob("*.wav"))
        self.assertEqual(len(files), 3)
        for path in files:
            a, b = self.metadata(path)["selection_seconds"]
            self.assertTrue(8 <= a < b <= 11)

    def test_hard_source_defaults_to_random_placement_and_preserves_the_selected_clip(self):
        first = self.source("hard_negative", seconds=4)
        second = self.source("hard_negative", seconds=1.672)
        self.open_panel()
        self.ns["negative_kind"].value = "hard_negative"
        self.ns["negative_source"].value = str(first)
        self.ns["negative_mode"].value = "clips"
        self.ns["negative_source"].value = str(second)
        self.assertEqual(self.ns["negative_mode"].value, "placement")
        self.assertNotEqual(self.ns["negative_count"].layout.display, "none")
        self.assertEqual(self.ns["negative_stride"].layout.display, "none")
        self.assertNotIn("random", dict(self.ns["negative_mode"].options).values())
        self.ns["negative_window"].value = 2.5
        self.ns["negative_count"].value = 3
        self.ns["negative_selection"].value = (0, 1.33)
        self.ns["extract_button"].click()
        files = list((self.root / "data/hard_negative/processed").rglob("*.wav"))
        self.assertEqual(len(files), 3)
        for path in files:
            self.assertEqual(self.metadata(path)["selection_seconds"], [0, 1.33])
            self.assertEqual(self.metadata(path)["windowing_mode"], "placement")
            self.assertEqual(sf.info(path).frames, 40000)

    def test_hard_random_placement_creates_requested_copies_without_cropping_the_clip(self):
        source = self.source("hard_negative", seconds=1)
        before = source.read_bytes()
        read = sf.SoundFile.read
        reads = []

        def selected_read(stream, frames=-1, *args, **kwargs):
            reads.append((stream.tell(), frames))
            return read(stream, frames, *args, **kwargs)

        with patch.object(sf.SoundFile, "read", selected_read):
            outputs = self.extract(
                "hard_negative", source, [(0.25, 0.75)], mode="placement", count=20, seed=17
            )
        self.assertEqual(reads, [(4000, 8000)])
        self.assertEqual(len(outputs), 20)
        offsets = []
        for path in outputs:
            meta = self.metadata(path)
            left = meta["placement_offset_samples"]
            offsets.append(left)
            self.assertTrue(0 <= left <= 24000)
            self.assertEqual(meta["placement_offset_seconds"], left / 16000)
            self.assertEqual(meta["padding_left_samples"] + meta["padding_right_samples"], 24000)
            self.assertEqual(meta["selection_seconds"], [0.25, 0.75])
            self.assertEqual(meta["label"], 0)
            audio, rate = sf.read(path, dtype="float32")
            self.assertEqual((rate, len(audio)), (16000, 32000))
            self.assertTrue(np.all(audio[:left] == 0))
            np.testing.assert_allclose(audio[left : left + 8000], 0.1, atol=1 / 32768)
            self.assertTrue(np.all(audio[left + 8000 :] == 0))
        self.assertGreater(len(set(offsets)), 1)
        self.assertEqual(source.read_bytes(), before)
        one = self.extract(
            "hard_negative", source, [(0.25, 0.75)], mode="placement", count=1, seed=17
        )
        self.assertEqual(len(one), 1)
        self.assertEqual(self.metadata(one[0])["placement_offset_samples"], offsets[0])

    def test_hard_random_placement_rejects_oversized_clip_and_handles_exact_fit(self):
        source = self.source("hard_negative", seconds=3)
        for intervals, count in [([(0, 3)], 1), ([(0, 1)], 0), ([(0, 1)], True)]:
            with self.subTest(intervals=intervals, count=count):
                with self.assertRaises(ValueError):
                    self.extract("hard_negative", source, intervals, mode="placement", count=count)
        self.assertFalse((self.root / "data/hard_negative/processed").exists())
        paths = self.extract("hard_negative", source, [(0, 2)], mode="placement", count=2)
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(self.metadata(p)["placement_offset_samples"] == 0 for p in paths))

    def test_multiple_hard_clips_get_requested_copies_per_clip(self):
        source = self.source("hard_negative", seconds=4)
        paths = self.extract("hard_negative", source, [(0, 1), (2, 3)], mode="placement", count=3)
        intervals = [self.metadata(p)["selection_seconds"] for p in paths]
        self.assertEqual(intervals.count([0, 1]), 3)
        self.assertEqual(intervals.count([2, 3]), 3)

    def test_short_random_selection_is_explained_before_extraction_and_window_controls_sync(self):
        source = self.source(seconds=4)
        self.open_panel()
        self.ns["negative_source"].value = str(source)
        self.ns["negative_selection"].value = (0, 1.5)
        self.assertTrue(self.ns["extract_button"].disabled)
        self.assertIn("1.50", self.ns["negative_summary"].value)
        self.assertIn("2.00", self.ns["negative_summary"].value)
        self.assertIn("longer", self.ns["negative_summary"].value)
        self.assertTrue("negative_window" in self.ns, "Missing local output window selector")
        self.ns["negative_window"].value = 1.5
        self.assertEqual(self.ns["window"].value, 1.5)
        self.assertFalse(self.ns["extract_button"].disabled)
        self.assertEqual(self.ns["negative_count"].value, 1)
        self.ns["window"].value = 2.5
        self.assertEqual(self.ns["negative_window"].value, 2.5)

    def test_mode_controls_follow_the_sampling_method(self):
        source = self.source(seconds=10)
        self.open_panel()
        self.ns["negative_source"].value = str(source)
        self.assertEqual(self.ns["negative_selection"].value, (0, 10))
        self.assertNotEqual(self.ns["negative_count"].layout.display, "none")
        self.assertEqual(self.ns["negative_stride"].layout.display, "none")
        self.ns["negative_mode"].value = "sliding"
        self.assertEqual(self.ns["negative_count"].layout.display, "none")
        self.assertNotEqual(self.ns["negative_stride"].layout.display, "none")

    def test_split_selector_locks_after_save_instead_of_allowing_a_failing_submission(self):
        source = self.source(seconds=10)
        self.open_panel()
        self.ns["negative_source"].value = str(source)
        self.assertFalse(self.ns["negative_split"].disabled)
        self.ns["negative_split"].value = "validation"
        self.ns["negative_count"].value = 1
        self.ns["extract_button"].click()
        self.assertTrue(self.ns["negative_split"].disabled)
        self.assertIn("validation", self.ns["negative_split_info"].value)
        self.ns["negative_split"].value = "train"
        self.assertEqual(self.ns["negative_split"].value, "validation")
        self.assertFalse(self.ns["extract_button"].disabled)

    def test_slider_preview_reads_the_selected_interval_without_five_second_limit(self):
        source = self.source(seconds=12, rate=44100, stereo=True)
        self.open_panel()
        self.ns["negative_source"].value = str(source)
        self.ns["negative_selection"].value = (2.5, 10.5)
        read = sf.SoundFile.read
        reads = []

        def selected_read(stream, frames=-1, *args, **kwargs):
            reads.append((stream.tell(), frames))
            return read(stream, frames, *args, **kwargs)

        with patch.object(sf.SoundFile, "read", selected_read):
            self.ns["negative_preview_button"].click()
        self.assertEqual(reads, [(110250, 352800)])
        players = [
            e
            for e in self.ns["output"].outputs
            if "<audio" in e.get("data", {}).get("text/html", "")
        ]
        self.assertEqual(len(players), 1)
        # Inspect the actual player payload, not just the logged duration.
        import base64
        import re

        encoded = re.search(r"base64,([^\"]+)", players[0]["data"]["text/html"]).group(1)
        audio, rate = sf.read(io.BytesIO(base64.b64decode(encoded)))
        self.assertEqual((rate, len(audio)), (16000, 128000))
        self.assertFalse((self.root / "data/negative/processed").exists())

    def test_slider_ranges_can_replace_and_append_before_extraction(self):
        source = self.source(seconds=12)
        self.open_panel()
        self.ns["negative_source"].value = str(source)
        self.ns["negative_ranges"].value = "0,12"
        self.ns["negative_selection"].value = (1, 4)
        self.ns["negative_use_range_button"].click()
        self.assertEqual(self.ns["parse_negative_ranges"](), [(1, 4)])
        self.ns["negative_selection"].value = (8, 11)
        self.ns["negative_add_range_button"].click()
        self.assertTrue(self.ns["negative_multiple"].value)
        self.assertEqual(self.ns["parse_negative_ranges"](), [(1, 4), (8, 11)])
        self.ns["negative_count"].value = 4
        self.ns["extract_button"].click()
        files = list((self.root / "data/negative/processed").rglob("*.wav"))
        self.assertEqual(len(files), 4)
        for path in files:
            a, b = self.metadata(path)["selection_seconds"]
            self.assertTrue(1 <= a < b <= 4 or 8 <= a < b <= 11)

    def test_slider_resets_for_short_hard_source_and_rejects_empty_selection(self):
        self.source(seconds=12)
        hard = self.source("hard_negative", seconds=0.25)
        self.open_panel()
        self.ns["negative_kind"].value = "hard_negative"
        self.ns["negative_source"].value = str(hard)
        self.assertEqual(self.ns["negative_selection"].value, (0, 0.25))
        self.assertEqual(self.ns["negative_selection"].max, 0.25)
        self.ns["negative_selection"].value = (0.1, 0.1)
        before = self.ns["negative_ranges"].value
        self.ns["negative_use_range_button"].click()
        self.assertEqual(self.ns["negative_ranges"].value, before)
        logs = "".join(e.get("text", "") for e in self.ns["output"].outputs)
        self.assertIn("ValueError", logs)
        self.ns["negative_source"].value = None
        self.assertTrue(self.ns["negative_selection"].disabled)
        self.assertTrue(self.ns["negative_preview_button"].disabled)

    def test_hard_negative_has_separate_storage_label_and_padded_event(self):
        source = self.source("hard_negative", seconds=1)
        before = source.read_bytes()
        outputs = self.extract(
            "hard_negative", source, [(0.25, 0.75)], mode="clips", window_seconds=2
        )
        audio, rate = sf.read(outputs[0], dtype="float32")
        self.assertEqual(rate, 16000)
        self.assertEqual(audio.shape, (32000,))
        self.assertTrue(np.all(audio[:12000] == 0))
        np.testing.assert_allclose(audio[12000:20000], 0.1, atol=1 / 32768)
        self.assertTrue(np.all(audio[20000:] == 0))
        meta = self.metadata(outputs[0])
        self.assertEqual(meta["label"], 0)
        self.assertEqual(meta["category"], "hard_negative")
        self.assertEqual(meta["selection_seconds"], [0.25, 0.75])
        self.assertEqual(meta["padding_left_samples"], 12000)
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(outputs[0].parent.parent, self.root / "data/hard_negative/processed")
        folder = self.ns["augment_recording"]("hard_negative", outputs[0])
        self.assertEqual(folder.parent, self.root / "data/hard_negative/augmented")
        augmented = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(augmented["label"], 0)
        self.assertEqual(augmented["source"], "data/hard_negative/raw/hard_negative_0001.wav")
        self.assertEqual(len(augmented["variants"]), 3)

    def test_random_windows_stay_inside_ranges_and_resample_only_selected_frames(self):
        source = self.source(rate=44100, stereo=True)
        read = sf.SoundFile.read
        reads = []

        def bounded_read(stream, frames=-1, *args, **kwargs):
            self.assertGreater(frames, 0)
            self.assertLessEqual(frames, 88200)
            reads.append(frames)
            return read(stream, frames, *args, **kwargs)

        with patch.object(sf.SoundFile, "read", bounded_read):
            with patch.dict(
                self.ns, {"normalize_source": lambda _: self.fail("Full-file decoding")}
            ):
                outputs = self.extract(
                    "negative", source, [(1, 5), (10, 15)], count=12, seed=7, split="train"
                )
        self.assertEqual(len(reads), 12)
        self.assertEqual(len(outputs), 12)
        starts = []
        for path in outputs:
            meta = self.metadata(path)
            a, b = meta["selection_seconds"]
            self.assertTrue(1 <= a < b <= 5 or 10 <= a < b <= 15)
            starts.append(a)
            self.assertEqual(meta["split"], "train")
            self.assertEqual(meta["label"], 0)
            self.assertEqual(meta["source_sample_rate"], 44100)
            self.assertEqual(sf.info(path).frames, 32000)
            audio, _ = sf.read(path)
            self.assertAlmostEqual(float(audio[100:-100].mean()), 0.2, places=3)
        self.assertEqual(len(set(starts)), 12)
        again = self.extract(
            "negative", source, [(1, 5), (10, 15)], count=12, seed=7, split="train"
        )
        self.assertNotEqual(outputs[0].parent, again[0].parent)
        self.assertEqual(
            [self.metadata(p)["selection_seconds"] for p in again],
            [self.metadata(p)["selection_seconds"] for p in outputs],
        )

    def test_sliding_windows_do_not_cross_ranges_or_include_partial_tails(self):
        source = self.source()
        outputs = self.extract(
            "negative", source, [(1, 5.5), (8, 12)], mode="sliding", stride_seconds=1
        )
        self.assertEqual(
            [self.metadata(p)["selection_seconds"] for p in outputs],
            [[1, 3], [2, 4], [3, 5], [8, 10], [9, 11], [10, 12]],
        )

    def test_hour_recording_is_seeked_without_reading_the_full_file(self):
        directory = self.root / "data/negative/raw"
        directory.mkdir(parents=True)
        source = directory / "negative_0001.wav"
        with sf.SoundFile(source, "w", samplerate=16000, channels=1, subtype="PCM_16") as stream:
            stream.seek(3600 * 16000 - 1)
            stream.write(np.array([0.1], dtype=np.float32))
        read = sf.SoundFile.read
        positions = []

        def bounded_read(stream, frames=-1, *args, **kwargs):
            self.assertEqual(frames, 32000)
            positions.append(stream.tell())
            return read(stream, frames, *args, **kwargs)

        with patch.object(sf.SoundFile, "read", bounded_read):
            outputs = self.extract("negative", source, [(3590, 3600)], count=3, seed=2)
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(p >= 3590 * 16000 for p in positions))

    def test_invalid_ranges_and_oversized_clips_do_not_create_outputs(self):
        source = self.source()
        invalid = [
            ([], {}),
            ([(0, 1)], {}),
            ([(-1, 3)], {}),
            ([(1, 16)], {}),
            ([(1, 4), (3, 6)], {}),
            ([(float("nan"), 4)], {}),
            ([(0, 2)], {"count": 2}),
        ]
        before = source.with_suffix(".json").read_bytes()
        for intervals, kwargs in invalid:
            with self.subTest(intervals=intervals):
                with self.assertRaises(ValueError):
                    self.extract("negative", source, intervals, **kwargs)
        self.assertFalse((self.root / "data/negative/processed").exists())
        self.assertEqual(source.with_suffix(".json").read_bytes(), before)
        hard = self.source("hard_negative", seconds=3)
        with self.assertRaisesRegex(ValueError, "window"):
            self.extract("hard_negative", hard, [(0, 3)], mode="clips")
        self.assertFalse((self.root / "data/hard_negative/processed").exists())

    def test_source_split_is_shared_by_all_windows_and_cannot_change(self):
        source = self.source()
        first = self.extract("negative", source, [(0, 5)], count=2, split="validation")
        second = self.extract("negative", source, [(6, 10)], count=2)
        self.assertTrue(all(self.metadata(p)["split"] == "validation" for p in first + second))
        self.assertEqual(len({self.metadata(p)["source_group"] for p in first + second}), 1)
        with self.assertRaisesRegex(ValueError, "split"):
            self.extract("negative", source, [(0, 5)], count=2, split="train")

    def test_extraction_panel_creates_windows_and_batch_augments_only_processed(self):
        source = self.source(seconds=8)
        with patch.object(self.ns["sd"], "query_devices", return_value=[]):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.cells[2], self.ns)
        self.assertTrue("negative_source" in self.ns, "Missing negative extraction controls")
        self.ns["negative_source"].value = str(source)
        self.ns["negative_multiple"].value = True
        self.ns["negative_ranges"].value = "1, 4\n5, 8"
        self.ns["negative_count"].value = 4
        self.ns["extract_button"].click()
        windows = list((self.root / "data/negative/processed").rglob("*.wav"))
        self.assertEqual(len(windows), 4)
        with patch.dict(
            self.ns, {"process_recording": lambda *a, **k: self.fail("Batch read raw")}
        ):
            self.ns["negative_aug_button"].click()
        runs = list((self.root / "data/negative/augmented").glob("*/metadata.json"))
        self.assertEqual(len(runs), 4)

    def test_augmentation_defaults_to_newly_saved_files_and_can_target_a_subset(self):
        source = self.source(seconds=8)
        old = self.extract("negative", source, [(0, 4)], count=2)
        self.open_panel()
        self.assertIn("negative_processed", self.ns, "Missing processed file selection")
        self.ns["negative_source"].value = str(source)
        self.ns["negative_selection"].value = (4, 8)
        self.ns["negative_count"].value = 3
        self.ns["extract_button"].click()
        selected = self.ns["negative_processed"].value
        self.assertEqual(len(selected), 3)
        self.assertTrue(set(selected).isdisjoint(map(str, old)))
        self.assertEqual(self.ns["negative_aug_scope"].value, "selected")
        self.ns["negative_processed"].value = (selected[1],)
        self.ns["negative_aug_button"].click()
        runs = list((self.root / "data/negative/augmented").glob("*/metadata.json"))
        self.assertEqual(len(runs), 1)
        meta = json.loads(runs[0].read_text())
        self.assertEqual(meta["processed_source"], str(Path(selected[1]).relative_to(self.root)))

    def test_all_scope_is_explicit_and_includes_every_saved_sample(self):
        first = self.source(seconds=8)
        second = self.source(seconds=8)
        self.extract("negative", first, [(0, 4)], count=2)
        self.extract("negative", second, [(0, 4)], count=3)
        self.open_panel()
        self.assertIn("negative_aug_scope", self.ns, "Missing explicit augmentation scope")
        self.assertEqual(self.ns["negative_aug_scope"].value, "selected")
        self.assertTrue(self.ns["negative_aug_button"].disabled)
        self.ns["negative_aug_scope"].value = "all"
        self.assertFalse(self.ns["negative_aug_button"].disabled)
        self.ns["negative_aug_button"].click()
        runs = list((self.root / "data/negative/augmented").glob("*/metadata.json"))
        self.assertEqual(len(runs), 5)

    def test_processed_selection_clears_when_category_changes(self):
        normal = self.source(seconds=8)
        hard = self.source("hard_negative", seconds=1)
        old = self.extract("negative", normal, [(0, 4)], count=1)
        self.extract("hard_negative", hard, [(0, 1)], mode="placement", count=1)
        self.open_panel()
        self.assertIn("negative_processed", self.ns)
        self.ns["negative_processed"].value = tuple(map(str, old))
        self.ns["negative_aug_scope"].value = "all"
        self.ns["negative_kind"].value = "hard_negative"
        self.assertEqual(self.ns["negative_aug_scope"].value, "selected")
        self.assertEqual(self.ns["negative_processed"].value, ())
        self.assertTrue(self.ns["negative_aug_button"].disabled)
        self.assertTrue(
            all("hard_negative" in value for label, value in self.ns["negative_processed"].options)
        )
