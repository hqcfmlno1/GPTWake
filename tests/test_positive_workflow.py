"""Recorded and externally prepared positives share processed-only augmentation."""

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

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks/02_dataset_recording_augmentation.ipynb"


class PositiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="gptwake-positive-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="positive-test"\n')
        source = nbformat.read(NOTEBOOK, as_version=4).cells[0].source
        self.ns = {}
        with patch.object(sd, "query_devices", return_value=[]):
            with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
                exec(source, self.ns)
        self.addCleanup(lambda: self.ns["close_panel"](self.ns["_pilot_panel"]))

    def processed(
        self, name, seconds=1, rate=16000, channels=1, stage="processed", kind="positive"
    ):
        path = self.root / "data" / kind / stage / name
        path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.sin(np.arange(round(seconds * rate)) * 0.1).astype("float32") * 0.2
        if channels == 2:
            audio = np.column_stack([audio, audio])
        sf.write(path, audio, rate, subtype="PCM_16")
        return path

    def manifests(self):
        return [
            json.loads(p.read_text())
            for p in sorted((self.root / "data/positive/augmented").glob("*/metadata.json"))
        ]

    def test_positive_preview_stays_near_positive_controls_and_survives_negative_actions(self):
        source = self.ns["save_recording"]("positive", np.full(16000, 0.1, "float32"), 16000)
        self.ns["load_positive"](source)
        preview = self.ns["positive_preview_output"]
        children = self.ns["_pilot_panel"].children
        self.assertLess(children.index(preview), children.index(self.ns["negative_kind"]))
        self.ns["selection"].value = (0, 0.5)
        for _ in range(2):
            self.ns["preview_button"].click()
            entries = preview.outputs
            self.assertEqual(
                sum("<audio" in e.get("data", {}).get("text/html", "") for e in entries), 1
            )
            self.assertEqual(sum("image/svg+xml" in e.get("data", {}) for e in entries), 1)
        logs = "".join(e.get("text", "") for e in entries)
        self.assertIn("0.50s", logs)
        self.assertFalse(
            any(
                "<audio" in e.get("data", {}).get("text/html", "")
                for e in self.ns["output"].outputs
            )
        )
        raw = self.ns["save_recording"]("negative", np.full(32000, 0.1, "float32"), 16000)
        self.ns["refresh_negative_sources"](raw)
        self.ns["negative_preview_button"].click()
        self.assertEqual(preview.outputs, entries)
        self.assertEqual(
            sum(
                "<audio" in e.get("data", {}).get("text/html", "")
                for e in self.ns["output"].outputs
            ),
            1,
        )

    def test_save_positive_stores_only_reviewed_phrase_and_selects_it(self):
        source = self.ns["save_recording"]("positive", np.full(48000, 0.1, "float32"), 16000)
        original = source.read_bytes()
        self.ns["load_positive"](source)
        self.assertFalse((self.root / "data/positive/processed").exists())
        self.ns["selection"].value = (1, 1.75)
        self.ns["save_positive_button"].click()
        processed = self.root / "data/positive/processed" / source.name
        self.assertEqual(sf.info(processed).frames, 12000)
        meta = json.loads(processed.with_suffix(".json").read_text())
        self.assertEqual(meta["selection_seconds"], [1, 1.75])
        self.assertEqual(self.ns["positive_processed"].value, (str(processed),))
        self.assertEqual(self.ns["positive_aug_scope"].value, "selected")
        self.assertFalse((self.root / "data/positive/augmented").exists())
        self.assertEqual(source.read_bytes(), original)

    def test_synthetic_wav_without_raw_or_json_can_be_augmented(self):
        source = self.processed("tts_voice01_001.wav", stage="synthetic")
        before = source.read_bytes()
        self.ns["refresh_positive_processed"]()
        self.ns["positive_processed"].value = (str(source),)
        self.ns["positive_aug_button"].click()
        meta = self.manifests()[0]
        self.assertEqual(meta["label"], 1)
        self.assertEqual(meta["category"], "positive")
        self.assertEqual(meta["processed_source"], "data/positive/synthetic/tts_voice01_001.wav")
        self.assertEqual(meta["source_stage"], "processed")
        self.assertEqual(meta["source_type"], "synthetic")
        self.assertFalse((self.root / "data/positive/raw").exists())
        self.assertFalse(source.with_suffix(".json").exists())
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(len(meta["variants"]), 6)
        self.assertTrue(
            all(sf.info(p).frames == 40000 for p in source.parent.parent.glob("augmented/*/*.wav"))
        )

    def test_selected_subset_and_all_scope_use_processed_only(self):
        paths = [self.processed(f"tts_{i}.wav") for i in range(3)]
        source = self.ns["save_recording"]("positive", np.full(48000, 0.1, "float32"), 16000)
        self.ns["load_positive"](source)
        self.ns["selection"].value = (0, 0.5)
        self.ns["refresh_positive_processed"]()
        self.ns["positive_processed"].value = (str(paths[1]),)
        with patch.dict(
            self.ns,
            {
                "normalize_source": lambda *a: self.fail("Augmentation read raw"),
                "process_recording": lambda *a, **k: self.fail("Augmentation processed raw"),
            },
        ):
            self.ns["positive_aug_button"].click()
            self.assertEqual(len(self.manifests()), 1)
            self.assertEqual(
                self.manifests()[0]["processed_source"], "data/positive/processed/tts_1.wav"
            )
            self.ns["positive_aug_scope"].value = "human"
            self.ns["positive_aug_button"].click()
        self.assertEqual(len(self.manifests()), 4)
        self.assertFalse((paths[0].parent / source.name).exists())

    def test_raw_only_does_not_enable_augmentation(self):
        self.ns["save_recording"]("positive", np.full(16000, 0.1, "float32"), 16000)
        self.ns["refresh_positive_processed"]()
        self.ns["positive_aug_scope"].value = "human"
        self.assertTrue(self.ns["positive_aug_button"].disabled)
        with self.assertRaisesRegex(ValueError, "processed"):
            self.ns["augment_all"]("positive")
        self.assertFalse((self.root / "data/positive/processed").exists())
        self.assertFalse((self.root / "data/positive/augmented").exists())

    def test_refresh_preserves_selection_and_discovers_new_synthetic_files(self):
        first = self.processed("first.wav")
        self.ns["refresh_positive_processed"]()
        self.ns["positive_processed"].value = (str(first),)
        second = self.processed("second.wav")
        self.ns["positive_processed_refresh"].click()
        self.assertEqual(self.ns["positive_processed"].value, (str(first),))
        self.assertEqual(
            {value for _, value in self.ns["positive_processed"].options}, {str(first), str(second)}
        )
        self.ns["positive_aug_scope"].value = "human"
        self.assertTrue(self.ns["positive_processed"].disabled)
        self.assertFalse(self.ns["positive_aug_button"].disabled)

    def test_invalid_processed_files_are_reported_without_blocking_valid_files(self):
        valid = self.processed("valid.wav")
        self.processed("oversized.wav", seconds=3)
        self.processed("stereo.wav", channels=2)
        self.processed("wrong_rate.wav", rate=48000)
        self.processed("empty.wav", seconds=0)
        (valid.parent / "corrupt.wav").write_bytes(b"not audio")
        self.ns["refresh_positive_processed"]()
        self.ns["positive_aug_scope"].value = "human"
        self.ns["positive_aug_button"].click()
        self.assertEqual(len(self.manifests()), 1)
        self.assertEqual(
            self.manifests()[0]["processed_source"], "data/positive/processed/valid.wav"
        )
        logs = "".join(e.get("text", "") for e in self.ns["output"].outputs)
        for name in ["oversized.wav", "stereo.wav", "wrong_rate.wav", "empty.wav", "corrupt.wav"]:
            self.assertIn(name, logs)

    def test_missing_selection_is_rejected_before_any_augmentation(self):
        first = self.processed("first.wav")
        second = self.processed("second.wav")
        self.ns["refresh_positive_processed"]()
        self.ns["positive_processed"].value = (str(first), str(second))
        second.unlink()
        with self.assertRaisesRegex(ValueError, "Refresh"):
            self.ns["augment_chosen_positives"]()
        self.assertEqual(self.manifests(), [])

    def test_oversized_processed_is_rejected_before_decoding(self):
        source = self.processed("long.wav", seconds=3)
        with patch.object(
            sf, "read", side_effect=AssertionError("Decoded oversized processed audio")
        ):
            with self.assertRaisesRegex(ValueError, "window"):
                self.ns["augment_recording"]("positive", source, window_seconds=2)
        self.assertEqual(self.manifests(), [])

    def test_clearing_raw_picker_disables_saving_previous_source(self):
        source = self.ns["save_recording"]("positive", np.full(16000, 0.1, "float32"), 16000)
        self.ns["refresh_sources"]()
        self.ns["source_picker"].value = str(source)
        self.assertFalse(self.ns["save_positive_button"].disabled)
        self.ns["source_picker"].value = None
        self.assertTrue(self.ns["save_positive_button"].disabled)
        self.assertTrue(self.ns["preview_button"].disabled)
        self.assertIsNone(self.ns["state"]["source"])

    def test_failed_raw_load_cannot_save_the_previous_source(self):
        first = self.ns["save_recording"]("positive", np.full(16000, 0.1, "float32"), 16000)
        second = self.ns["save_recording"]("positive", np.full(16000, 0.1, "float32"), 16000)
        sidecar = self.root / "data/positive/processed" / second.with_suffix(".json").name
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text(json.dumps({"selection_method": "manual", "selection_seconds": [0, 3]}))
        self.ns["refresh_sources"]()
        self.ns["source_picker"].value = str(first)
        self.ns["source_picker"].value = str(second)
        self.assertTrue(self.ns["save_positive_button"].disabled)
        self.assertIsNone(self.ns["state"]["source"])

    def test_human_and_synthetic_all_scopes_are_separate_even_with_same_filename(self):
        human = self.processed("same.wav")
        synth = self.processed("same.wav", stage="synthetic")
        self.ns["refresh_positive_processed"]()
        self.assertEqual(
            {value for _, value in self.ns["positive_processed"].options}, {str(human), str(synth)}
        )
        self.ns["positive_aug_scope"].value = "human"
        self.ns["positive_aug_button"].click()
        self.assertEqual(
            [m["processed_source"] for m in self.manifests()], ["data/positive/processed/same.wav"]
        )
        self.ns["positive_aug_scope"].value = "synthetic"
        self.ns["positive_aug_button"].click()
        self.assertEqual(
            {m["processed_source"] for m in self.manifests()},
            {"data/positive/processed/same.wav", "data/positive/synthetic/same.wav"},
        )
        self.assertEqual({m["source_type"] for m in self.manifests()}, {"human", "synthetic"})

    def test_synthetic_negative_without_json_and_human_negative_use_separate_scopes(self):
        raw = self.ns["save_recording"]("negative", np.full(32000, 0.1, "float32"), 16000)
        human = self.ns["extract_negative_windows"]("negative", raw, [(0, 2)], mode="sliding")[0]
        synth = self.processed("tts_negative.wav", seconds=2, stage="synthetic", kind="negative")
        self.ns["refresh_processed_samples"]()
        self.ns["negative_aug_scope"].value = "human"
        self.ns["negative_aug_button"].click()
        folder = self.root / "data/negative/augmented"
        manifests = [json.loads(p.read_text()) for p in folder.glob("*/metadata.json")]
        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0]["processed_source"], str(human.relative_to(self.root)))
        self.ns["negative_aug_scope"].value = "synthetic"
        self.ns["negative_aug_button"].click()
        manifests = [json.loads(p.read_text()) for p in folder.glob("*/metadata.json")]
        synthetic = [m for m in manifests if m["source_type"] == "synthetic"]
        self.assertEqual(len(synthetic), 1)
        self.assertEqual(synthetic[0]["label"], 0)
        self.assertEqual(synthetic[0]["processed_source"], str(synth.relative_to(self.root)))
        self.assertEqual(len(synthetic[0]["variants"]), 3)
        self.assertFalse(synth.with_suffix(".json").exists())

    def test_bad_human_metadata_does_not_block_synthetic_negative_refresh_or_augmentation(self):
        human = self.processed("bad.wav", seconds=2, kind="negative")
        human.with_suffix(".json").write_text("{")
        synth = self.processed("good.wav", seconds=2, stage="synthetic", kind="negative")
        self.ns["negative_processed_refresh"].click()
        options = {value for _, value in self.ns["negative_processed"].options}
        self.assertIn(str(synth), options)
        logs = "".join(e.get("text", "") for e in self.ns["output"].outputs)
        self.assertIn("bad.wav", logs)
        self.ns["negative_aug_scope"].value = "synthetic"
        self.ns["negative_aug_button"].click()
        manifests = list((self.root / "data/negative/augmented").glob("*/metadata.json"))
        self.assertEqual(len(manifests), 1)
        self.assertEqual(json.loads(manifests[0].read_text())["source_type"], "synthetic")

    def test_supplied_speaker_split_and_lineage_are_preserved(self):
        source = self.processed("human_01.wav")
        metadata = {
            "source": "data/positive/raw/human_01.wav",
            "speaker_id": "human01",
            "split": "train",
            "label": 1,
            "category": "positive",
            "selection_seconds": [1, 2],
        }
        source.with_suffix(".json").write_text(json.dumps(metadata))
        self.ns["refresh_positive_processed"]()
        self.ns["positive_aug_scope"].value = "human"
        self.ns["positive_aug_button"].click()
        result = self.manifests()[0]
        for key, value in metadata.items():
            self.assertEqual(result[key], value)


if __name__ == "__main__":
    unittest.main()
