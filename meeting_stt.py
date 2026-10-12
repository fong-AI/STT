import argparse
from queue import Empty, Queue
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import pyaudiowpatch as pyaudio

from session import MeetingSession, ROOT, load_model


LANGUAGES = {"Vietnamese": "vi", "English": "en", "Auto detect": None}


def devices():
    with pyaudio.PyAudio() as manager:
        host = manager.get_host_api_info_by_type(pyaudio.paWASAPI)
        items = list(manager.get_device_info_generator_by_host_api(host_api_index=host["index"]))
        outputs = [d for d in items if d.get("isLoopbackDevice") and d["maxInputChannels"] > 0]
        microphones = [d for d in items if not d.get("isLoopbackDevice") and d["maxInputChannels"] > 0]
        try:
            default_output = manager.get_default_wasapi_loopback()["index"]
        except OSError:
            default_output = -1
        outputs.sort(key=lambda d: d["index"] != default_output)
        microphones.sort(key=lambda d: d["index"] != host["defaultInputDevice"])
        return outputs, microphones


class App:
    def __init__(self, window):
        self.window = window
        self.events = Queue()
        self.session = None
        self.closing = False
        self.has_error = False
        self.saved_path = None
        self.window.title("Meeting STT - PhoWhisper-small")
        self.window.geometry("940x700")
        self.window.minsize(720, 540)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        frame = ttk.Frame(window, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Meeting transcript", font=("Segoe UI", 19, "bold")).pack(anchor="w")
        ttk.Label(frame, text="Local speech recognition for Zoom and Google Meet. No STT API fees.").pack(anchor="w", pady=(4, 16))
        settings = ttk.Frame(frame)
        settings.pack(fill="x")
        settings.columnconfigure(1, weight=1)
        self.output = ttk.Combobox(settings, state="readonly")
        self.microphone = ttk.Combobox(settings, state="readonly")
        self.language = ttk.Combobox(settings, state="readonly", values=list(LANGUAGES))
        self.language.set("Vietnamese")
        for row, (label, control) in enumerate((("Meeting audio output", self.output), ("Microphone", self.microphone), ("Language", self.language))):
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=5)
            control.grid(row=row, column=1, sticky="ew", pady=5)
        self.refresh = ttk.Button(settings, text="Refresh devices", command=self.refresh_devices)
        self.refresh.grid(row=0, column=2, padx=(10, 0))
        self.use_mic = tk.BooleanVar(value=False)
        self.mic_toggle = ttk.Checkbutton(settings, text="Include my microphone", variable=self.use_mic, command=self.toggle_mic)
        self.mic_toggle.grid(row=1, column=2, padx=(10, 0))
        ttk.Label(frame, text="Choose the SAME output device as Zoom/Meet. Use headphones to avoid duplicate speech.").pack(anchor="w", pady=(12, 2))
        ttk.Label(frame, text="This captures all audio on that output. Zoom/Meet mute does not mute this tool's microphone.").pack(anchor="w", pady=(0, 12))
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="Start", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=8)
        self.status = tk.StringVar(value="Ready. Nothing is being recorded.")
        ttk.Label(frame, textvariable=self.status, wraplength=880).pack(anchor="w", pady=(12, 4))
        levels = ttk.Frame(frame)
        levels.pack(fill="x", pady=(0, 10))
        self.meters = {}
        for label in ("Meeting", "You"):
            ttk.Label(levels, text=label).pack(side="left", padx=(0, 6))
            meter = ttk.Progressbar(levels, maximum=100, length=150)
            meter.pack(side="left", padx=(0, 20))
            self.meters[label] = meter
        self.text = ScrolledText(frame, wrap="word", font=("Segoe UI", 11), state="disabled")
        self.text.pack(fill="both", expand=True)
        self.saved = tk.StringVar(value="Transcripts are saved to " + str(ROOT / "transcripts"))
        ttk.Label(frame, textvariable=self.saved, wraplength=880).pack(anchor="w", pady=(10, 0))
        self.refresh_devices()
        self.window.after(100, self.poll)

    def refresh_devices(self):
        try:
            self.outputs, self.microphones = devices()
            self.output["values"] = [f"{d['index']}: {d['name']}" for d in self.outputs]
            self.microphone["values"] = [f"{d['index']}: {d['name']}" for d in self.microphones]
            self.output.set("")
            self.microphone.set("")
            if self.outputs:
                self.output.current(0)
            if self.microphones:
                self.microphone.current(0)
            self.start_button["state"] = "normal" if self.outputs else "disabled"
            if not self.outputs:
                self.status.set("No output loopback device. Connect headphones/speakers and refresh devices.")
            else:
                self.status.set("Ready. Nothing is being recorded.")
        except Exception as error:
            self.outputs, self.microphones = [], []
            self.start_button["state"] = "disabled"
            self.status.set("Cannot list audio devices: " + str(error))

    def toggle_mic(self):
        if self.session and self.session.alive():
            if self.use_mic.get():
                self.session.mic_enabled.set()
            else:
                self.session.mic_enabled.clear()

    def start(self):
        if self.session and self.session.alive():
            return
        index = self.output.current()
        if index < 0:
            messagebox.showerror("Select output", "Select the audio output used by Zoom/Meet.")
            return
        selected = [("Meeting", self.outputs[index])]
        mic_index = self.microphone.current()
        if self.use_mic.get() and mic_index < 0:
            messagebox.showerror("Select microphone", "Select a microphone or turn off Include my microphone.")
            return
        # Keep the selected microphone available for a live enable/disable toggle.
        if mic_index >= 0:
            selected.append(("You", self.microphones[mic_index]))
        self.has_error = False
        self.saved_path = None
        self.saved.set("No transcript file yet; waiting for the local model.")
        self.session = MeetingSession(selected, LANGUAGES[self.language.get()], self.events, ROOT / "transcripts")
        if not self.use_mic.get():
            self.session.mic_enabled.clear()
        self.text["state"] = "normal"
        self.text.delete("1.0", "end")
        self.text["state"] = "disabled"
        self.start_button["state"] = "disabled"
        self.stop_button["state"] = "normal"
        for control in (self.output, self.microphone, self.language, self.refresh):
            control["state"] = "disabled"
        self.status.set("Loading PhoWhisper-small...")
        self.session.start()

    def stop(self):
        if self.session:
            self.session.stop()
        self.stop_button["state"] = "disabled"
        if not self.has_error:
            self.status.set("Stopping capture and finishing pending transcript...")

    def close(self):
        if self.session and self.session.alive():
            self.closing = True
            self.stop()
        else:
            self.window.destroy()

    def poll(self):
        for _ in range(200):
            try:
                event = self.events.get_nowait()
            except Empty:
                break
            kind = event[0]
            if kind == "status" and not self.has_error and not (self.session and self.session.stopped.is_set()):
                self.status.set(event[1])
            elif kind == "level":
                self.meters[event[1]]["value"] = min(100, event[2] * 500)
            elif kind == "file":
                self.saved_path = event[1]
                self.saved.set("Saved to: " + event[1])
            elif kind == "transcript":
                self.text["state"] = "normal"
                self.text.insert("end", event[1] + "\n")
                self.text.see("end")
                self.text["state"] = "disabled"
            elif kind == "error":
                self.has_error = True
                self.status.set("Stopped with error: " + event[1])
                self.stop()
            elif kind == "timing" and not self.has_error and self.session and not self.session.stopped.is_set():
                self.status.set(f"Listening | last {event[2]:.1f}s audio processed in {event[1]:.1f}s")
        if self.session and self.session.stopped.is_set() and not self.session.alive():
            self.session = None
            for control in (self.output, self.microphone, self.language):
                control["state"] = "readonly"
            self.refresh["state"] = "normal"
            self.start_button["state"] = "normal" if self.outputs else "disabled"
            self.stop_button["state"] = "disabled"
            for meter in self.meters.values():
                meter["value"] = 0
            if not self.has_error:
                self.status.set("Stopped. Transcript saved. Nothing is being recorded." if self.saved_path else "Stopped before recording started. No transcript created.")
            if self.closing:
                self.window.destroy()
                return
        self.window.after(100, self.poll)


def main():
    parser = argparse.ArgumentParser(description="Local Zoom/Meet transcription")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--self-check", action="store_true", help="Load model and transcribe silence; never records devices")
    args = parser.parse_args()
    if args.list_devices:
        outputs, microphones = devices()
        for category, items in (("OUTPUT LOOPBACK", outputs), ("MICROPHONE", microphones)):
            print(category)
            for item in items:
                print(f"  {item['index']}: {item['name']} ({int(item['defaultSampleRate'])} Hz)")
    elif args.self_check:
        import time
        import numpy as np
        model = load_model()
        start = time.monotonic()
        segments, _ = model.transcribe(np.zeros(16000 * 5, dtype=np.float32), language="vi", vad_filter=True)
        transcript = " ".join(segment.text for segment in segments).strip()
        assert not transcript, "Silence produced text"
        print(f"PASS: PhoWhisper-small loaded; silence produced no text ({time.monotonic() - start:.2f}s)")
    else:
        window = tk.Tk()
        ttk.Style().theme_use("vista")
        App(window)
        window.mainloop()


if __name__ == "__main__":
    main()
