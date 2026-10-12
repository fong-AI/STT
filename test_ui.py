from threading import Event
import tkinter as tk
from unittest import TestCase, main
from unittest.mock import patch

import meeting_stt


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.stopped = Event()
        self.mic_enabled = Event()
        self.running = False
    def start(self): self.running = True
    def stop(self): self.stopped.set()
    def alive(self): return self.running


class UITests(TestCase):
    def setUp(self):
        self.window = tk.Tk()
        self.window.withdraw()
        self.addCleanup(self.window.destroy)
        self.device = {"index": 1, "name": "Test output"}
        self.mock = patch.object(meeting_stt, "devices", return_value=([self.device], []))
        self.mock.start()
        self.addCleanup(self.mock.stop)
        self.app = meeting_stt.App(self.window)

    def test_stop_before_file_does_not_claim_saved_transcript(self):
        self.app.saved.set("Saved to: old-session.txt")
        with patch.object(meeting_stt, "MeetingSession", FakeSession):
            self.app.start()
        self.app.stop()
        self.app.session.running = False
        self.app.poll()
        self.assertNotIn("Transcript saved", self.app.status.get())
        self.assertNotIn("old-session", self.app.saved.get())
        self.assertEqual(str(self.app.start_button["state"]), "normal")

    def test_transcript_events_update_ui_and_file_status(self):
        with patch.object(meeting_stt, "MeetingSession", FakeSession):
            self.app.start()
        self.app.events.put(("file", "test-session.txt"))
        self.app.events.put(("transcript", "[00:01] Meeting: hello"))
        self.app.poll()
        self.assertIn("hello", self.app.text.get("1.0", "end"))
        self.assertIn("test-session.txt", self.app.saved.get())
        self.app.stop()
        self.app.session.running = False
        self.app.poll()
        self.assertIn("Transcript saved", self.app.status.get())

    def test_model_selection_is_passed_and_locked_during_capture(self):
        self.assertEqual(self.app.model.get(), "PhoWhisper-small")
        self.app.model.set("PhoWhisper-medium")
        with patch.object(meeting_stt, "MeetingSession", side_effect=FakeSession) as constructor:
            self.app.start()
        self.assertEqual(constructor.call_args.kwargs["model_name"], "PhoWhisper-medium")
        self.assertEqual(str(self.app.model["state"]), "disabled")
        self.app.stop()
        self.app.session.running = False
        self.app.poll()
        self.assertEqual(str(self.app.model["state"]), "readonly")

    def test_backend_remains_visible_when_timing_status_updates(self):
        with patch.object(meeting_stt, "MeetingSession", FakeSession):
            self.app.start()
        self.app.events.put(("backend", "GPU CUDA / float16"))
        self.app.events.put(("timing", 0.3, 5.0))
        self.app.poll()
        self.assertIn("GPU CUDA", self.app.backend.get())
        self.assertIn("0.3s", self.app.status.get())

    def test_no_devices_disables_start(self):
        with patch.object(meeting_stt, "devices", return_value=([], [])):
            self.app.refresh_devices()
        self.assertEqual(str(self.app.start_button["state"]), "disabled")
        self.assertIn("No output", self.app.status.get())

    def test_close_waits_for_pending_work(self):
        with patch.object(meeting_stt, "MeetingSession", FakeSession):
            self.app.start()
        self.app.close()
        self.assertTrue(self.app.session.stopped.is_set())
        self.assertTrue(self.app.closing)
        self.assertTrue(self.window.winfo_exists())


if __name__ == "__main__": main()
