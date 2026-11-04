"""Check the preview's single publication path in a real, isolated kernel."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat
from jupyter_client import KernelManager

NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "notebooks/02_positive_recording_augmentation.ipynb"
)


class KernelOutputTests(unittest.TestCase):
    def test_six_variants_publish_once_to_widget_state_without_captured_iopub(self):
        source = [
            c.source for c in nbformat.read(NOTEBOOK, as_version=4).cells if c.cell_type == "code"
        ][-1]
        with tempfile.TemporaryDirectory(prefix="gptwake-kernel-") as temp:
            Path(temp, "pyproject.toml").write_text('[project]\nname="preview-test"\n')
            manager = KernelManager()
            manager.kernel_spec.argv = [
                sys.executable,
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ]
            manager.start_kernel(cwd=temp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            client = manager.blocking_client()
            client.start_channels()
            try:
                client.wait_for_ready(timeout=30)

                def execute(code):
                    msg_id = client.execute(code)
                    messages = []
                    while True:
                        message = client.get_iopub_msg(timeout=30)
                        if message.get("parent_header", {}).get("msg_id") != msg_id:
                            continue
                        self.assertNotEqual(message["msg_type"], "error", message["content"])
                        messages.append(message)
                        if (
                            message["msg_type"] == "status"
                            and message["content"]["execution_state"] == "idle"
                        ):
                            return messages

                execute("import sounddevice as sd\nsd.query_devices = lambda *a, **k: []")
                execute(source)
                execute(
                    "p = save_recording('positive', np.sin(np.arange(16000)*0.1).astype('float32')*0.2, 16000)\n"
                    "refresh_sources(p)\nload_positive(p)"
                )
                messages = execute("augment_button.click()")
                self.assertFalse(
                    any(
                        m["msg_type"] in {"display_data", "stream", "clear_output"}
                        for m in messages
                    )
                )
                snapshots = [
                    m["content"]["data"]["state"]["outputs"]
                    for m in messages
                    if m["msg_type"] == "comm_msg"
                    and "outputs" in m["content"].get("data", {}).get("state", {})
                ]
                self.assertTrue(snapshots)
                entries = snapshots[-1]
                # One selected-source preview and six augmented previews, each with one waveform.
                self.assertEqual(
                    sum("<audio" in e.get("data", {}).get("text/html", "") for e in entries), 7
                )
                self.assertEqual(sum("image/svg+xml" in e.get("data", {}) for e in entries), 7)
                self.assertEqual("".join(e.get("text", "") for e in entries).count("quieter:"), 1)
                runs = list(Path(temp, "data/positive/augmented").glob("*/metadata.json"))
                self.assertEqual(len(runs), 1)
                self.assertEqual(len(json.loads(runs[0].read_text())["variants"]), 6)
            finally:
                client.stop_channels()
                manager.shutdown_kernel(now=True)
