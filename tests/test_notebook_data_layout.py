"""Behavior checks for the notebook's dataset storage and audio lineage."""

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


class NotebookStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gptwake-layout-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="audio-fixture"\n')
        nb = nbformat.read(NOTEBOOK, as_version=4)
        cells = [cell.source for cell in nb.cells if cell.cell_type == "code"]
        self.panel = cells[2]
        self.ns = {}
        with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(cells[0], "<setup>", "exec"), self.ns)
            exec(compile(cells[1], "<helpers>", "exec"), self.ns)

    def function(self, name):
        self.assertTrue(callable(self.ns.get(name)), f"Missing notebook behavior: {name}")
        return self.ns[name]

    def fixture(self, seconds=4, rate=48000):
        audio = np.zeros(seconds * rate, dtype=np.float32)
        t = np.arange(rate) / rate
        audio[rate : rate * 2] = 0.2 * np.sin(2 * np.pi * 300 * t)
        return audio

    def test_setup_does_not_create_dataset_directories(self):
        self.assertFalse((self.root / "data").exists(), "Setup created empty dataset directories")

    def open_panel(self):
        with patch.object(self.ns["sd"], "query_devices", return_value=[]):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(self.panel, self.ns)

    def test_positive_defaults_to_full_recording_even_with_legacy_heuristic_bounds(self):
        source = self.function("save_recording")("positive", self.fixture(), 48000)
        process = self.function("process_recording")
        path = process("positive", source)[0]
        self.assertEqual(sf.info(path).frames, 64000)
        old = json.loads(path.with_suffix(".json").read_text())
        old.pop("selection_method", None)
        old["selection_seconds"] = [0.76, 2.86]
        path.with_suffix(".json").write_text(json.dumps(old))
        path = process("positive", source)[0]
        self.assertEqual(sf.info(path).frames, 64000)
        self.assertEqual(
            json.loads(path.with_suffix(".json").read_text())["selection_seconds"], [0, 4]
        )

    def test_each_variant_has_one_widget_preview_without_iopub_display(self):
        source = self.function("save_recording")("positive", self.fixture(), 48000)
        processed = self.function("process_recording")("positive", source, bounds=(1, 2))[0]
        self.open_panel()
        with patch.dict(self.ns, {"display": lambda _: self.fail("Unexpected IOPub display")}):
            self.ns["run_action"](
                lambda: self.ns["augment_recording"]("positive", processed, show_players=True)
            )
        entries = self.ns["output"].outputs
        audio = [e for e in entries if "<audio" in e.get("data", {}).get("text/html", "")]
        svg = [e for e in entries if "image/svg+xml" in e.get("data", {})]
        self.assertEqual(len(audio), 6)
        self.assertEqual(len(svg), 6)
        logs = "".join(e.get("text", "") for e in entries)
        self.assertEqual(logs.count("quieter:"), 1)
        runs = list((self.root / "data/positive/augmented").glob("*/metadata.json"))
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(json.loads(runs[0].read_text())["variants"]), 6)
        self.assertEqual(self.ns["output"].msg_id, "")

    def test_rerun_disposes_old_output_and_controls(self):
        self.open_panel()
        old_output, old_button = self.ns["output"], self.ns["augment_button"]
        self.open_panel()
        self.assertIsNone(old_output.comm)
        self.assertIsNone(old_button.comm)
        self.assertTrue(old_button.disabled)

    def test_action_error_is_reported_once_in_widget(self):
        self.open_panel()

        def fail():
            raise ValueError("Invalid phrase")

        self.ns["run_action"](fail)
        logs = "".join(e.get("text", "") for e in self.ns["output"].outputs)
        self.assertEqual(logs.count("ValueError: Invalid phrase"), 1)
        self.assertFalse(self.ns["record_button"].disabled)

    def test_preview_handles_overrange_without_changing_source_or_boosting_quiet_audio(self):
        cases = [
            ([0.0, 1.2, -0.6], [0.0, 0.999, -0.4995]),
            ([0.0, 0.2, -0.1], [0.0, 0.2, -0.1]),
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ]
        for samples, expected in cases:
            with self.subTest(samples=samples):
                audio = np.array(samples, dtype=np.float32)
                original = audio.copy()
                displayed = []
                with patch.dict(self.ns, {"display": displayed.append}):
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.function("show_audio")(audio, "Preview test")
                decoded, rate = sf.read(io.BytesIO(displayed[0].data), dtype="float32")
                self.assertEqual(rate, 16000)
                np.testing.assert_allclose(decoded, expected, atol=2 / 32768)
                np.testing.assert_array_equal(audio, original)

    def test_raw_preserves_rate_channels_and_samples(self):
        audio = np.column_stack([self.fixture(), self.fixture() * 0.5])
        path = self.function("save_recording")("positive", audio, 48000, {})
        self.assertEqual(path, self.root / "data/positive/raw/positive_0001.wav")
        decoded, rate = sf.read(path, dtype="float32")
        self.assertEqual(rate, 48000)
        np.testing.assert_array_equal(decoded, audio)
        self.assertEqual(sf.info(path).subtype, "FLOAT")
        self.assertFalse((self.root / "data/noise").exists())

    def test_positive_processing_crops_without_changing_raw(self):
        path = self.function("save_recording")("positive", self.fixture(), 48000, {})
        original = path.read_bytes()
        processed = self.function("process_recording")("positive", path, bounds=(0.9, 2.3))[0]
        self.assertEqual(processed.parent, self.root / "data/positive/processed")
        info = sf.info(processed)
        self.assertEqual(
            (info.samplerate, info.channels, info.frames, info.subtype), (16000, 1, 22400, "PCM_16")
        )
        metadata = json.loads(processed.with_suffix(".json").read_text())
        self.assertEqual(metadata["source"], "data/positive/raw/positive_0001.wav")
        self.assertEqual(metadata["selection_seconds"], [0.9, 2.3])
        self.assertEqual(metadata["label"], 1)
        self.assertEqual(path.read_bytes(), original)

    def test_noise_uses_raw_source_without_creating_processed_copy(self):
        path = self.function("save_recording")("noise", self.fixture(), 48000, {})
        with patch.dict(
            self.ns, {"normalize_source": lambda _: self.fail("Noise was fully decoded")}
        ):
            library = self.function("prepare_noise_library")()
        self.assertEqual(library, self.root / "data/noise/raw")
        self.assertTrue(path.exists())
        self.assertFalse((self.root / "data/noise/processed").exists())
        self.assertFalse((self.root / "data/noise/augmented").exists())

    def test_negative_processing_emits_complete_labeled_windows(self):
        path = self.function("save_recording")("negative", self.fixture(), 48000, {})
        windows = self.function("extract_negative_windows")(
            "negative", path, [(0, 4)], window_seconds=2.0, mode="sliding"
        )
        self.assertEqual(len(windows), 2)
        for i, output in enumerate(windows):
            self.assertEqual(output.parent.parent, self.root / "data/negative/processed")
            self.assertEqual(sf.info(output).frames, 32000)
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["label"], 0)
            self.assertEqual(metadata["selection_seconds"], [i * 2.0, (i + 1) * 2.0])

    def test_repeated_recording_uses_new_names_without_overwrite(self):
        save = self.function("save_recording")
        first = save("noise", np.ones(16000, dtype=np.float32) * 0.1, 16000, {})
        before = first.read_bytes()
        second = save("noise", np.ones(16000, dtype=np.float32) * 0.2, 16000, {})
        self.assertEqual((first.name, second.name), ("noise_0001.wav", "noise_0002.wav"))
        self.assertEqual(first.read_bytes(), before)

    def test_augmentation_uses_shared_noise_and_records_raw_lineage(self):
        save = self.function("save_recording")
        process = self.function("process_recording")
        noise = np.random.default_rng(1).normal(0, 0.01, 48000 * 3).astype("float32")
        save("noise", noise, 48000, {})
        source = save("positive", self.fixture(), 48000, {})
        phrase = process("positive", source, bounds=(0.9, 2.3))[0]
        original = source.read_bytes()
        folder = self.function("augment_recording")("positive", phrase, window_seconds=2.0)
        self.assertEqual(folder.parent, self.root / "data/positive/augmented")
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(metadata["source"], "data/positive/raw/positive_0001.wav")
        self.assertEqual(len(metadata["variants"]), 6)
        self.assertTrue(any(v["file"] == "room_noise.wav" for v in metadata["variants"]))
        for entry in metadata["variants"]:
            info = sf.info(folder / entry["file"])
            self.assertEqual((info.frames, info.samplerate, info.channels), (32000, 16000, 1))
        self.assertEqual(source.read_bytes(), original)

    def test_run_all_initializes_controls_without_capturing_or_creating_directories(self):
        def unexpected_capture(*args, **kwargs):
            raise AssertionError("Run All attempted microphone capture")

        with patch.object(self.ns["sd"], "rec", unexpected_capture):
            with patch.object(self.ns["sd"], "query_devices", return_value=[]):
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(self.panel, "<control-panel>", "exec"), self.ns)
        self.assertFalse((self.root / "data").exists())

    def test_batch_processing_reuses_saved_phrase_bounds(self):
        source = self.function("save_recording")("positive", self.fixture(), 48000, {})
        process = self.function("process_recording")
        path = process("positive", source, bounds=(0.8, 2.2))[0]
        path = process("positive", source)[0]
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(metadata["selection_seconds"], [0.8, 2.2])
        self.assertEqual(sf.info(path).frames, 22400)

    def test_negative_variants_preserve_class_length_and_source(self):
        source = self.function("save_recording")("negative", self.fixture(), 48000, {})
        processed = self.function("extract_negative_windows")(
            "negative", source, [(0, 4)], mode="sliding"
        )[0]
        folder = self.function("augment_recording")("negative", processed)
        self.assertEqual(folder.parent, self.root / "data/negative/augmented")
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(metadata["label"], 0)
        self.assertEqual(metadata["source"], "data/negative/raw/negative_0001.wav")
        self.assertEqual(len(metadata["variants"]), 3)
        for variant in metadata["variants"]:
            self.assertEqual(sf.info(folder / variant["file"]).frames, 32000)

    def test_legacy_import_keeps_original_and_is_idempotent(self):
        source = self.root / "data/negatives/room_noise_pilot/old_session/room_old.wav"
        source.parent.mkdir(parents=True)
        sf.write(source, self.fixture(rate=16000), 16000, subtype="PCM_16")
        original = source.read_bytes()
        migrate = self.function("import_legacy_recordings")
        imported = migrate()
        self.assertEqual(len(imported), 1)
        self.assertEqual(imported[0].parent, self.root / "data/noise/raw")
        self.assertEqual(imported[0].read_bytes(), original)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(migrate(), [])


if __name__ == "__main__":
    unittest.main()
