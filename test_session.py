import importlib
import importlib.util
from pathlib import Path
from queue import Queue
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

import numpy as np


class SessionTests(TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("session"), "session implementation is missing")
        self.module = importlib.import_module("session")
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.events = Queue()
        self.session = self.module.MeetingSession([], "vi", self.events, Path(self.temp.name))
        self.device_mock = patch.object(self.module, "inference_device", return_value=("cpu", "int8"), create=True)
        self.device_mock.start()
        self.addCleanup(self.device_mock.stop)
        self.audio_mock = patch.object(self.module.pyaudio, "PyAudio", return_value=MagicMock())
        self.audio_mock.start()
        self.addCleanup(self.audio_mock.stop)

    def drain(self):
        result = []
        while not self.events.empty():
            result.append(self.events.get_nowait())
        return result

    def test_model_loader_uses_phowhisper_from_local_cache(self):
        with patch.object(self.module, "ROOT", Path(self.temp.name)), patch("faster_whisper.WhisperModel", return_value=SimpleNamespace(model=SimpleNamespace(device="cpu", compute_type="int8"))) as constructor:
            self.module.load_model.__wrapped__()
        self.assertEqual(constructor.call_count, 1)
        args, kwargs = constructor.call_args
        self.assertEqual(args[0], "qbsmlabs/PhoWhisper-small")
        self.assertTrue(kwargs["local_files_only"])

    def test_missing_cache_downloads_phowhisper_without_generic_fallback(self):
        with patch.object(self.module, "ROOT", Path(self.temp.name)), patch("faster_whisper.WhisperModel", side_effect=[FileNotFoundError("not cached"), SimpleNamespace()]) as constructor:
            self.module.load_model.__wrapped__()
        self.assertEqual(constructor.call_count, 2)
        self.assertTrue(all(call.args[0] == "qbsmlabs/PhoWhisper-small" for call in constructor.call_args_list))
        self.assertNotIn("local_files_only", constructor.call_args.kwargs)

    def test_gpu_model_uses_cuda_and_checks_real_inference(self):
        model = MagicMock()
        model.transcribe.return_value = (iter([]), None)
        with patch.object(self.module, "ROOT", Path(self.temp.name)), patch.object(self.module, "inference_device", return_value=("cuda", "float16")), patch("faster_whisper.WhisperModel", return_value=model) as constructor:
            self.assertIs(self.module.load_model.__wrapped__(), model)
        self.assertEqual(constructor.call_args.kwargs["device"], "cuda")
        self.assertEqual(constructor.call_args.kwargs["compute_type"], "float16")
        self.assertFalse(model.transcribe.call_args.kwargs["vad_filter"])

    def test_gpu_initialization_failure_does_not_silently_use_cpu(self):
        with patch.object(self.module, "ROOT", Path(self.temp.name)), patch.object(self.module, "inference_device", return_value=("cuda", "float16")), patch("faster_whisper.WhisperModel", side_effect=RuntimeError("CUDA out of memory")) as constructor:
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                self.module.load_model.__wrapped__()
        self.assertEqual(constructor.call_count, 1)
        self.assertEqual(constructor.call_args.kwargs["device"], "cuda")

    def test_gpu_inference_failure_is_reported_before_recording(self):
        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("missing CUDA library")
        with patch.object(self.module, "ROOT", Path(self.temp.name)), patch.object(self.module, "inference_device", return_value=("cuda", "float16")), patch("faster_whisper.WhisperModel", return_value=model):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                self.module.load_model.__wrapped__()

    def test_medium_loader_uses_selected_repo(self):
        with patch.object(self.module, "ROOT", Path(self.temp.name)), patch("faster_whisper.WhisperModel", return_value=SimpleNamespace(model=SimpleNamespace(device="cpu", compute_type="int8"))) as constructor:
            self.module.load_model.__wrapped__("PhoWhisper-medium")
        self.assertEqual(constructor.call_args.args[0], "qbsmlabs/PhoWhisper-medium")

    def test_unknown_model_rejected_before_loading(self):
        with patch("faster_whisper.WhisperModel") as constructor:
            with self.assertRaises(ValueError):
                self.module.load_model.__wrapped__("unknown")
        constructor.assert_not_called()

    def test_medium_session_uses_selected_model_and_labels_transcript(self):
        session = self.module.MeetingSession([], "vi", self.events, Path(self.temp.name), model_name="PhoWhisper-medium")
        with patch.object(self.module, "load_model", return_value=SimpleNamespace(model=SimpleNamespace(device="cpu", compute_type="int8"))) as load:
            session.start()
            session.thread.join(3)
        load.assert_called_once_with("PhoWhisper-medium")
        files = list(Path(self.temp.name).glob("*.txt"))
        self.assertEqual(len(files), 1)
        self.assertIn("PhoWhisper-medium", files[0].read_text(encoding="utf-8"))
        self.assertIn("CPU", files[0].read_text(encoding="utf-8"))
        self.assertTrue(any(e[0] == "backend" and "CPU" in e[1] for e in self.drain()))

    def test_session_uses_beam_search_on_gpu(self):
        for device, beam in (("cuda", 5), ("cpu", 1)):
            with self.subTest(device=device):
                session = self.module.MeetingSession([("Meeting", {})], "vi", Queue(), Path(self.temp.name))
                def capture(*args):
                    session._enqueue(("Meeting", 0, np.ones(1600, dtype=np.float32)))
                    session.stop()
                transcribe = MagicMock(return_value=(iter([SimpleNamespace(text="xin chao", start=0)]), None))
                model = SimpleNamespace(model=SimpleNamespace(device=device, compute_type="int8"), transcribe=transcribe)
                with patch.object(self.module, "load_model", return_value=model), patch.object(session, "_capture", side_effect=capture):
                    session.start()
                    session.thread.join(3)
                self.assertFalse(session.alive())
                self.assertEqual(transcribe.call_args.kwargs["beam_size"], beam)

    def test_stop_during_model_load_never_opens_capture(self):
        entered, release = Event(), Event()
        def load(*args):
            entered.set()
            release.wait(3)
            return SimpleNamespace()
        self.session.devices = [("Meeting", {})]
        with patch.object(self.module, "load_model", side_effect=load), patch.object(self.session, "_capture") as capture:
            self.session.start()
            self.assertTrue(entered.wait(2))
            self.session.stop()
            release.set()
            self.session.thread.join(3)
            self.assertFalse(self.session.alive())
            capture.assert_not_called()
            self.assertIn("done", [e[0] for e in self.drain()])

    def test_stop_drains_pending_audio_and_flushes_file(self):
        def capture(label, device):
            self.session._enqueue((label, 1.0, np.ones(1600, dtype=np.float32)))
            self.session.stop()
        fake_model = SimpleNamespace(model=SimpleNamespace(device="cpu", compute_type="int8"), transcribe=lambda *a, **k: (iter([SimpleNamespace(text=" hello", start=0.1)]), None))
        self.session.devices = [("Meeting", {})]
        with patch.object(self.module, "load_model", return_value=fake_model), patch.object(self.session, "_capture", side_effect=capture):
            self.session.start()
            self.session.thread.join(3)
        self.assertFalse(self.session.alive())
        files = list(Path(self.temp.name).glob("*.txt"))
        self.assertEqual(len(files), 1)
        self.assertIn("hello", files[0].read_text(encoding="utf-8"))
        events = self.drain()
        self.assertEqual(len([e for e in events if e[0] == "transcript"]), 1)
        self.assertEqual(events[-1][0], "done")

    def test_queue_overflow_is_visible_and_stops_capture(self):
        self.session.audio = Queue(maxsize=1)
        self.session.audio.put(("Meeting", 0, np.ones(1)))
        with self.assertRaisesRegex(RuntimeError, "queue"):
            self.session._enqueue(("Meeting", 1, np.ones(1)))

    def test_stop_without_available_frames_does_not_block_read(self):
        stream = MagicMock()
        stream.get_read_available.return_value = 0
        stream.read.side_effect = AssertionError("read would block without available audio")
        manager = MagicMock()
        manager.open.return_value = stream
        manager.__enter__.return_value = manager
        device = {"defaultSampleRate": 16000, "maxInputChannels": 1, "index": 1}
        self.session.manager = manager
        with patch.object(self.module.pyaudio, "PyAudio", return_value=manager):
            capture = Thread(target=self.session._capture, args=("Meeting", device))
            capture.start()
            import time
            time.sleep(0.03)
            self.session.stop()
            capture.join(1)
        self.assertFalse(capture.is_alive())
        stream.read.assert_not_called()
        self.assertFalse(any(e[0] == "error" for e in self.drain()))

    def test_capture_sources_share_one_audio_manager(self):
        stream = MagicMock()
        stream.get_read_available.return_value = 0
        manager = MagicMock()
        manager.open.return_value = stream
        manager.__enter__.return_value = manager
        device = {"defaultSampleRate": 16000, "maxInputChannels": 1, "index": 1}
        self.session.devices = [("Meeting", device), ("You", device)]
        with patch.object(self.module, "load_model", return_value=SimpleNamespace(model=SimpleNamespace(device="cpu", compute_type="int8"))), patch.object(self.module.pyaudio, "PyAudio", return_value=manager) as initialize:
            self.session.start()
            import time
            deadline = time.monotonic() + 2
            while manager.open.call_count < 2 and self.session.alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.session.stop()
            self.session.thread.join(2)
        self.assertFalse(self.session.alive())
        self.assertEqual(manager.open.call_count, 2)
        self.assertEqual(initialize.call_count, 1, "PortAudio must not initialize concurrently in capture threads")
        self.assertFalse(any(e[0] == "error" for e in self.drain()))

    def test_file_failure_is_reported_without_capture(self):
        self.session.directory = Path(self.temp.name)/"missing"/"nested"
        with patch.object(self.module, "load_model", return_value=SimpleNamespace(model=SimpleNamespace(device="cpu", compute_type="int8"))), patch("session.Path.mkdir", side_effect=PermissionError("denied")):
            self.session.start()
            self.session.thread.join(3)
        self.assertFalse(self.session.alive())
        events = self.drain()
        self.assertTrue(any(e[0] == "error" and "denied" in e[1] for e in events))
        self.assertEqual(events[-1][0], "done")

    def test_inference_failure_stops_capture_and_reports_error(self):
        def capture(label, device):
            self.session._enqueue((label, 0, np.ones(1600)))
            self.session.stopped.wait(2)
        fake_model = SimpleNamespace(model=SimpleNamespace(device="cpu", compute_type="int8"), transcribe=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("inference failed")))
        self.session.devices = [("Meeting", {})]
        with patch.object(self.module, "load_model", return_value=fake_model), patch.object(self.session, "_capture", side_effect=capture):
            self.session.start()
            self.session.thread.join(3)
        self.assertFalse(self.session.alive())
        self.assertTrue(any(e[0] == "error" and "inference failed" in e[1] for e in self.drain()))


if __name__ == "__main__":
    main()
