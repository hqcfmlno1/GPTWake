"""Exercise streaming noise capture and the independent one-cell entry point."""

import contextlib
import io
import json
import queue
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat
import numpy as np
import soundfile as sf

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/02_positive_recording_augmentation.ipynb"
)


class FakeInputStream:
    """Replace only microphone I/O; real recorder code writes and closes actual WAV files."""

    def __init__(self, blocks, failure=None):
        self.blocks = queue.Queue()
        for block in blocks:
            self.blocks.put(block)
        self.failure = failure
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def read(self, frames):
        if self.failure:
            raise self.failure
        try:
            return self.blocks.get(timeout=0.02), False
        except queue.Empty:
            return np.empty((0, 1), dtype=np.float32), False

    def abort(self):
        pass


class NoiseRecordingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="gptwake-noise-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "pyproject.toml").write_text('[project]\nname="noise-test"\n')
        nb = nbformat.read(NOTEBOOK, as_version=4)
        self.cells = [c.source for c in nb.cells if c.cell_type == "code"]
        self.ns = {}
        with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
            exec(self.cells[0], self.ns)
            exec(self.cells[1], self.ns)

    def recorder(self):
        self.assertTrue("AudioRecorder" in self.ns, "Missing continuous noise recorder")
        return self.ns["AudioRecorder"]()

    def start_fake(self, recorder, stream, kind="noise"):
        with patch.object(
            self.ns["sd"],
            "query_devices",
            return_value={"name": "Fake mic", "default_samplerate": 48000},
        ):
            with patch.object(self.ns["sd"], "check_input_settings"):
                with patch.object(self.ns["sd"], "InputStream", return_value=stream):
                    recorder.start(None, kind)
                    self.assertTrue(recorder.ready.wait(2), "Recorder did not initialize")

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(predicate(), "Recorder did not reach the expected state")

    def test_stream_writes_incrementally_until_stop_and_closes_valid_wav(self):
        blocks = [
            np.full((4096, 1), 0.1, dtype=np.float32),
            np.full((4096, 1), 0.2, dtype=np.float32),
        ]
        stream = FakeInputStream(blocks)
        recorder = self.recorder()
        self.addCleanup(recorder.stop)
        self.start_fake(recorder, stream)
        self.wait_for(lambda: recorder.frames == 8192)
        self.assertTrue(recorder.running)
        self.assertGreater(recorder.path.stat().st_size, 8192 * 4)
        self.assertFalse((self.root / "data/noise/processed").exists())
        recorder.stop()
        self.assertTrue(stream.closed)
        audio, rate = sf.read(recorder.path, dtype="float32", always_2d=True)
        self.assertEqual(rate, 48000)
        np.testing.assert_array_equal(audio, np.concatenate(blocks))
        metadata = json.loads(recorder.path.with_suffix(".json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["frames"], 8192)
        self.assertEqual(sf.info(recorder.path).format, "RF64")
        self.assertFalse(recorder.running)

    def test_device_failure_finalizes_file_and_excludes_it_from_noise_pool(self):
        recorder = self.recorder()
        self.addCleanup(recorder.stop)
        self.start_fake(recorder, FakeInputStream([], RuntimeError("device disconnected")))
        self.wait_for(lambda: not recorder.running)
        self.assertIsNotNone(recorder.error)
        metadata = json.loads(recorder.path.with_suffix(".json").read_text())
        self.assertEqual(metadata["status"], "failed")
        self.assertEqual(self.ns["available_audio_files"](self.ns["prepare_noise_library"]()), [])

    def test_active_recording_is_excluded_then_becomes_available_after_stop(self):
        recorder = self.recorder()
        self.addCleanup(recorder.stop)
        self.start_fake(recorder, FakeInputStream([np.full((4096, 1), 0.1, dtype=np.float32)]))
        self.wait_for(lambda: recorder.frames == 4096)
        self.assertEqual(self.ns["available_audio_files"](self.ns["prepare_noise_library"]()), [])
        recorder.stop()
        self.assertEqual(
            self.ns["available_audio_files"](self.ns["prepare_noise_library"]()), [recorder.path]
        )

    def test_failed_and_empty_speech_captures_do_not_enter_processing(self):
        for kind in ("positive", "negative", "hard_negative"):
            for failure in (None, RuntimeError("device disconnected")):
                with self.subTest(kind=kind, failure=failure):
                    recorder = self.recorder()
                    self.addCleanup(recorder.stop)
                    self.start_fake(recorder, FakeInputStream([], failure), kind)
                    recorder.stop()
                    self.assertEqual(self.ns["available_audio_files"](recorder.path.parent), [])
                    with self.assertRaisesRegex(ValueError, "completed"):
                        self.ns["process_recording"](kind, recorder.path)
                    self.assertFalse((self.root / "data" / kind / "processed").exists())

    def test_noise_mixing_samples_multiple_raw_files_and_offsets(self):
        self.assertTrue("make_noise_transform" in self.ns)
        save = self.ns["save_recording"]
        rng = np.random.default_rng(2)
        first = save("noise", rng.normal(0, 0.03, (48000 * 3, 2)).astype("float32"), 48000)
        second = save("noise", rng.normal(0, 0.02, 16000 * 5).astype("float32"), 16000)
        audio = np.sin(np.arange(32000) * 0.1).astype("float32") * 0.2
        chosen, offsets = set(), set()
        for seed in range(20):
            self.ns["random"].seed(seed)
            transform = self.ns["make_noise_transform"](self.ns["prepare_noise_library"]())
            mixed = transform(samples=audio.copy(), sample_rate=16000)
            self.assertEqual(mixed.shape, (32000,))
            self.assertTrue(np.isfinite(mixed).all())
            self.assertFalse(np.array_equal(mixed, audio))
            chosen.add(Path(transform.parameters["noise_file_path"]))
            offsets.add(transform.parameters["offset"])
        self.assertEqual(chosen, {first, second})
        self.assertGreater(len(offsets), 1)
        self.assertFalse((self.root / "data/noise/processed").exists())

    def test_standalone_cell_runs_in_fresh_namespace_and_keeps_stop_control(self):
        self.assertGreaterEqual(len(self.cells), 4, "Missing standalone code cell")
        namespace = {}
        with patch.object(self.ns["sd"], "query_devices", return_value=[]):
            with patch.object(
                self.ns["sd"], "InputStream", side_effect=AssertionError("Unexpected capture")
            ):
                with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
                    exec(self.cells[-1], namespace)
        self.assertIn("noise_stop_button", namespace)
        self.assertTrue(namespace["noise_stop_button"].disabled)
        self.assertFalse(namespace["noise_button"].disabled)
        self.assertFalse((self.root / "data").exists())

    def test_short_rf64_recording_repeats_to_fill_mixing_window(self):
        recorder = self.recorder()
        self.addCleanup(recorder.stop)
        block = np.sin(np.arange(4096) * 0.07).astype("float32")[:, None] * 0.05
        self.start_fake(recorder, FakeInputStream([block]))
        self.wait_for(lambda: recorder.frames == 4096)
        recorder.stop()
        transform = self.ns["make_noise_transform"](self.ns["prepare_noise_library"]())
        audio = np.full(32000, 0.1, dtype=np.float32)
        mixed = transform(samples=audio, sample_rate=16000)
        self.assertEqual(mixed.shape, audio.shape)
        self.assertTrue(np.isfinite(mixed).all())
        self.assertFalse(np.array_equal(mixed[-100:], audio[-100:]))
        self.assertEqual(Path(transform.parameters["noise_file_path"]), recorder.path)
        self.assertEqual(transform.parameters["offset"], 0)

    def test_stop_button_is_available_during_capture_and_restores_controls(self):
        self.assertGreaterEqual(len(self.cells), 4)
        namespace = {}
        with patch.object(self.ns["sd"], "query_devices", return_value=[]):
            with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
                exec(self.cells[-1], namespace)
        recorder = namespace["_recorder"]
        self.addCleanup(recorder.stop)
        stream = FakeInputStream([np.ones((4096, 1), dtype=np.float32) * 0.1])
        info = {"name": "Fake mic", "default_samplerate": 48000}
        with patch.object(self.ns["sd"], "query_devices", return_value=info):
            with patch.object(self.ns["sd"], "check_input_settings"):
                with patch.object(self.ns["sd"], "InputStream", return_value=stream):
                    with contextlib.redirect_stdout(io.StringIO()):
                        namespace["noise_button"].click()
                    self.assertTrue(recorder.ready.wait(2))
        self.wait_for(lambda: recorder.frames == 4096)
        self.assertTrue(namespace["record_button"].disabled)
        self.assertFalse(namespace["noise_stop_button"].disabled)
        with self.assertRaisesRegex(RuntimeError, "Stop"):
            exec(self.cells[2], namespace)
        self.assertIs(namespace["_recorder"], recorder)
        self.assertFalse(namespace["noise_stop_button"].disabled)
        with contextlib.redirect_stdout(io.StringIO()):
            namespace["noise_stop_button"].click()
        self.assertFalse(recorder.running)
        self.assertTrue(namespace["noise_stop_button"].disabled)
        self.assertFalse(namespace["record_button"].disabled)
        self.assertEqual(sf.info(recorder.path).frames, 4096)

    def test_rerunning_setup_during_capture_preserves_live_recorder(self):
        recorder = self.recorder()
        self.addCleanup(recorder.stop)
        self.start_fake(recorder, FakeInputStream([]))
        self.ns["_recorder"] = recorder
        with self.assertRaisesRegex(RuntimeError, "Stop"):
            exec(self.cells[0], self.ns)
        self.assertIs(self.ns["_recorder"], recorder)
        self.assertTrue(recorder.running)
        recorder.stop()

    def test_all_speech_buttons_record_until_stop_and_negatives_remain_raw(self):
        namespace = {}
        with patch.object(self.ns["sd"], "query_devices", return_value=[]):
            with contextlib.chdir(self.root), contextlib.redirect_stdout(io.StringIO()):
                exec(self.cells[-1], namespace)
        self.assertIn("record_stop_button", namespace)
        self.assertIn("negative_stop_button", namespace)
        for kind, start_name, stop_name, frames in [
            ("positive", "record_button", "record_stop_button", 48000),
            ("negative", "negative_button", "negative_stop_button", 144000),
            ("negative", "negative_button", "negative_stop_button", 4096),
            ("hard_negative", "hard_negative_button", "hard_negative_stop_button", 4097),
        ]:
            with self.subTest(kind=kind, frames=frames):
                block = np.full((frames, 1), 0.1, dtype=np.float32)
                stream = FakeInputStream([block])
                info = {"name": "Fake mic", "default_samplerate": 48000}
                with patch.object(self.ns["sd"], "query_devices", return_value=info):
                    with patch.object(self.ns["sd"], "check_input_settings"):
                        with patch.object(self.ns["sd"], "InputStream", return_value=stream):
                            namespace[start_name].click()
                            recorder = namespace["_recorder"]
                            self.addCleanup(recorder.stop)
                            self.assertTrue(recorder.ready.wait(2))
                self.wait_for(lambda: recorder.frames == frames)
                self.assertTrue(recorder.running)
                self.assertFalse(namespace[stop_name].disabled)
                self.assertTrue(namespace["noise_button"].disabled)
                if kind == "positive":
                    namespace[stop_name].click()
                else:
                    with patch.dict(
                        namespace,
                        {"normalize_source": lambda _: self.fail("Stop decoded full negative")},
                    ):
                        namespace[stop_name].click()
                self.assertFalse(recorder.running)
                self.assertTrue(namespace[stop_name].disabled)
                self.assertFalse(namespace["noise_button"].disabled)
                self.assertEqual(recorder.path.parent, self.root / "data" / kind / "raw")
                decoded, rate = sf.read(recorder.path, dtype="float32", always_2d=True)
                self.assertEqual(rate, 48000)
                np.testing.assert_array_equal(decoded, block)
                metadata = json.loads(recorder.path.with_suffix(".json").read_text())
                self.assertEqual(metadata["status"], "complete")
                self.assertEqual(metadata["label"], 1 if kind == "positive" else 0)
                if kind == "positive":
                    self.assertEqual(namespace["selection"].value, (0, 1))
                    self.assertEqual(
                        sf.info(self.root / "data/positive/processed" / recorder.path.name).frames,
                        16000,
                    )
                else:
                    self.assertFalse((self.root / "data" / kind / "processed").exists())
                    logs = "".join(e.get("text", "") for e in namespace["output"].outputs)
                    self.assertIn("Save samples", logs)
                    self.assertEqual(namespace["negative_source"].value, str(recorder.path))
                    if kind == "hard_negative":
                        namespace["extract_button"].click()
                        windows = list((self.root / "data/hard_negative/processed").rglob("*.wav"))
                        self.assertEqual(len(windows), 1)
                        self.assertEqual(sf.info(windows[0]).frames, 32000)


if __name__ == "__main__":
    unittest.main()
