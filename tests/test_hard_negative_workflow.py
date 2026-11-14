"""Hard-negative phrases are manually prepared before fixed-window augmentation."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import sounddevice as sd
import soundfile as sf

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/dataset_recording_augmentation.ipynb"


class HardNegativeWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="gptwake-hard-phrase-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="hard-phrase-test"\n')
        self.ns = {}
        with patch.object(sd, "query_devices", return_value=[]):
            with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
                exec(nbformat.read(NOTEBOOK, as_version=4).cells[0].source, self.ns)
        self.addCleanup(lambda: self.ns["close_panel"](self.ns["_pilot_panel"]))
        self.ns["negative_kind"].value = "hard_negative"

    def raw(self, seconds=4, rate=48000):
        audio = np.full((round(seconds * rate), 2), 0.2, "float32")
        return self.ns["save_recording"]("hard_negative", audio, rate)

    def prepared(self, name, seconds=1, rate=16000, channels=1):
        path = self.root / "data/hard_negative/processed" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.full(round(seconds * rate), 0.2, "float32")
        if channels == 2:
            audio = np.column_stack([audio, audio])
        sf.write(path, audio, rate)
        return path

    def manifests(self):
        return [
            json.loads(p.read_text())
            for p in sorted((self.root / "data/hard_negative/augmented").glob("*/metadata.json"))
        ]

    def test_open_preview_and_manual_save_create_only_an_unpadded_phrase(self):
        raw = self.raw()
        before = raw.read_bytes()
        self.ns["refresh_negative_sources"](raw)
        self.assertFalse((self.root / "data/hard_negative/processed").exists())
        self.assertTrue(self.ns["extract_button"].disabled)
        self.ns["negative_selection"].value = (1, 1.75)
        self.ns["negative_preview_button"].click()
        self.assertFalse((self.root / "data/hard_negative/processed").exists())
        self.ns["extract_button"].click()
        files = list((self.root / "data/hard_negative/processed").rglob("*.wav"))
        self.assertEqual(len(files), 1)
        path = files[0]
        self.assertEqual(path.parent, self.root / "data/hard_negative/processed")
        info = sf.info(path)
        self.assertEqual((info.samplerate, info.channels, info.frames), (16000, 1, 12000))
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(metadata["label"], 0)
        self.assertEqual(metadata["selection_seconds"], [1, 1.75])
        self.assertEqual(metadata["selection_method"], "manual")
        self.assertEqual(raw.read_bytes(), before)
        self.assertFalse((self.root / "data/hard_negative/augmented").exists())
        self.assertEqual(self.ns["negative_processed"].value, (str(path),))
        self.ns["refresh_negative_sources"](raw)
        self.assertEqual(self.ns["negative_selection"].value, (1, 1.75))

    def test_oversize_crop_rejects_before_writing_or_overwriting(self):
        raw = self.raw()
        process = self.ns["process_recording"]
        with self.assertRaisesRegex(ValueError, "window"):
            process("hard_negative", raw, bounds=(0, 3), window_seconds=2.5)
        self.assertFalse((self.root / "data/hard_negative/processed").exists())
        path = process("hard_negative", raw, bounds=(1, 2), window_seconds=2.5)[0]
        previous = (path.read_bytes(), path.with_suffix(".json").read_bytes())
        with self.assertRaisesRegex(ValueError, "window"):
            process("hard_negative", raw, bounds=(0, 4), window_seconds=2.5)
        self.assertEqual((path.read_bytes(), path.with_suffix(".json").read_bytes()), previous)

    def test_synth_copy_in_needs_no_raw_or_json_and_is_placed_only_at_augmentation(self):
        source = self.prepared("tts_01.wav", seconds=0.75)
        before = source.read_bytes()
        self.ns["negative_processed_refresh"].click()
        self.assertIn(str(source), dict(self.ns["negative_processed"].options).values())
        self.ns["negative_processed"].value = (str(source),)
        self.ns["negative_aug_button"].click()
        manifests = self.manifests()
        self.assertEqual(len(manifests), 1)
        metadata = manifests[0]
        self.assertEqual(metadata["label"], 0)
        self.assertEqual(metadata["source"], "data/hard_negative/processed/tts_01.wav")
        self.assertEqual(metadata["source_type"], "unassigned")
        self.assertEqual(len(metadata["variants"]), 6)
        self.assertEqual(metadata["window_seconds"], 2.5)
        files = list((self.root / "data/hard_negative/augmented").rglob("*.wav"))
        self.assertEqual(len(files), 6)
        self.assertTrue(all(sf.info(p).frames == 40000 for p in files))
        placements = [v for v in metadata["variants"] if v["file"].startswith("placement_")]
        self.assertEqual(len(placements), 3)
        self.assertTrue(all(0 <= v["parameters"]["offset_seconds"] <= 1.75 for v in placements))
        self.assertEqual(source.read_bytes(), before)
        self.assertFalse(source.with_suffix(".json").exists())
        self.assertFalse((self.root / "data/hard_negative/raw").exists())

    def test_all_processed_skips_invalid_files_and_retains_valid_inputs(self):
        good = self.prepared("valid.wav")
        self.prepared("too_long.wav", seconds=3)
        self.prepared("wrong_rate.wav", rate=22050)
        self.prepared("stereo.wav", channels=2)
        self.prepared("empty.wav", seconds=0)
        bad = self.prepared("bad_metadata.wav")
        bad.with_suffix(".json").write_text("{")
        self.ns["negative_processed_refresh"].click()
        self.ns["negative_aug_scope"].value = "human"
        self.assertIn("all processed", self.ns["negative_aug_button"].description.lower())
        self.ns["negative_aug_button"].click()
        manifests = self.manifests()
        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0]["processed_source"], str(good.relative_to(self.root)))
        logs = "".join(e.get("text", "") for e in self.ns["output"].outputs)
        for name in (
            "too_long.wav",
            "wrong_rate.wav",
            "stereo.wav",
            "empty.wav",
            "bad_metadata.wav",
        ):
            self.assertIn(name, logs)

    def test_augmentation_uses_saved_phrase_and_current_window_without_reprocessing_raw(self):
        raw = self.raw()
        self.ns["refresh_negative_sources"](raw)
        self.ns["negative_selection"].value = (1, 2)
        self.ns["extract_button"].click()
        selected = self.ns["negative_processed"].value
        self.assertEqual(len(selected), 1)
        self.ns["negative_selection"].value = (0, 4)
        self.ns["window"].value = 3.0
        with patch.dict(self.ns, {"process_recording": lambda *a, **k: self.fail("Augmented raw")}):
            self.ns["negative_aug_button"].click()
        manifests = self.manifests()
        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0]["selection_seconds"], [1, 2])
        self.assertEqual(manifests[0]["window_seconds"], 3.0)
        self.assertEqual(manifests[0]["source_type"], "human")

    def test_hard_ui_hides_window_extraction_controls_and_preview_is_local(self):
        for name in ("negative_mode", "negative_count", "negative_stride", "negative_advanced"):
            self.assertEqual(self.ns[name].layout.display, "none", name)
        source = self.raw(seconds=1)
        self.ns["refresh_negative_sources"](source)
        children = self.ns["_pilot_panel"].children
        preview = self.ns["negative_preview_output"]
        self.assertLess(
            children.index(preview), children.index(self.ns["negative_augmentation_box"])
        )
        for _ in range(2):
            self.ns["negative_preview_button"].click()
            self.assertEqual(
                sum("<audio" in e.get("data", {}).get("text/html", "") for e in preview.outputs), 1
            )
        self.ns["negative_kind"].value = "negative"
        self.assertEqual(self.ns["negative_mode"].layout.display, "")
        self.assertEqual(self.ns["negative_advanced"].layout.display, "")
        self.assertEqual(self.ns["negative_augmentation_box"].layout.display, "none")


if __name__ == "__main__":
    unittest.main()
