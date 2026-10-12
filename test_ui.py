from threading import Event
import tkinter as tk
from unittest import TestCase, main
from unittest.mock import patch

import meeting_stt


class FakeSession:
    def __init__(self, *args, **kwargs):
        self.stopped = Event()
        self.mic_enabled = Event()
        self.mic_enabled.set()
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

    def test_available_microphone_is_enabled_by_default_and_can_be_muted(self):
        window = tk.Tk()
        window.withdraw()
        self.addCleanup(window.destroy)
        microphone = {"index": 2, "name": "Test microphone"}
        with patch.object(meeting_stt, "devices", return_value=([self.device], [microphone])):
            app = meeting_stt.App(window)
        self.assertTrue(app.use_mic.get())
        with patch.object(meeting_stt, "MeetingSession", side_effect=FakeSession) as constructor:
            app.start()
        self.assertIn(("You", microphone), constructor.call_args.args[0])
        self.assertTrue(app.session.mic_enabled.is_set())
        app.use_mic.set(False)
        app.toggle_mic()
        self.assertFalse(app.session.mic_enabled.is_set())
        app.use_mic.set(True)
        app.toggle_mic()
        self.assertTrue(app.session.mic_enabled.is_set())

    def test_no_microphone_does_not_block_meeting_audio(self):
        self.assertFalse(self.app.use_mic.get())
        self.assertEqual(str(self.app.mic_toggle["state"]), "disabled")
        with patch.object(meeting_stt, "MeetingSession", side_effect=FakeSession) as constructor:
            self.app.start()
        self.assertEqual(constructor.call_args.args[0], [("Meeting", self.device)])

    def test_streaming_drafts_replace_previous_text_and_keep_sources_separate(self):
        self.app.events.put(("partial", "You", "alo mot"))
        self.app.events.put(("partial", "Meeting", "xin chao"))
        self.app.events.put(("partial", "You", "alo hai"))
        self.app.poll()
        self.assertIn("alo hai", self.app.partials["You"].get())
        self.assertNotIn("alo mot", self.app.partials["You"].get())
        self.assertIn("xin chao", self.app.partials["Meeting"].get())
        self.assertNotIn("alo hai", self.app.text.get("1.0", "end"))
        self.app.events.put(("partial", "You", ""))
        self.app.poll()
        self.assertEqual(self.app.partials["You"].get(), "")

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
