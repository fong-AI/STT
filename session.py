from datetime import datetime
from functools import lru_cache
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time
import os
import sysconfig

import numpy as np
import pyaudiowpatch as pyaudio

from audio_utils import SpeechBuffer, to_16k

ROOT = Path(__file__).resolve().parent
MODELS = {name: "qbsmlabs/" + name for name in ("PhoWhisper-small", "PhoWhisper-medium")}


@lru_cache(maxsize=1)
def configure_cuda_runtime():
    if os.name == "nt":
        library = Path(sysconfig.get_path("purelib")) / "nvidia" / "cublas" / "bin"
        if library.is_dir():
            os.environ["PATH"] = str(library) + os.pathsep + os.environ.get("PATH", "")
            return os.add_dll_directory(str(library))


def inference_device():
    configure_cuda_runtime()
    import ctranslate2
    if ctranslate2.get_cuda_device_count() == 0:
        return "cpu", "int8"
    supported = ctranslate2.get_supported_compute_types("cuda")
    return "cuda", next((kind for kind in ("int8_float16", "float16", "float32") if kind in supported), "float32")


def backend_description(model):
    device = model.model.device
    label = "GPU CUDA" if device == "cuda" else "CPU (no CUDA-capable NVIDIA GPU detected)"
    return f"{label} / {model.model.compute_type}"


@lru_cache(maxsize=1)
def load_model(model_name="PhoWhisper-small"):
    if model_name not in MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    from faster_whisper import WhisperModel
    device, compute_type = inference_device()
    cache = ROOT / "models"
    cache.mkdir(exist_ok=True)
    try:
        # Prefer local files so subsequent meetings do not require Internet access.
        try:
            model = WhisperModel(MODELS[model_name], device=device, compute_type=compute_type, cpu_threads=6,
                                 download_root=str(cache), local_files_only=True)
        except (FileNotFoundError, OSError):
            model = WhisperModel(MODELS[model_name], device=device, compute_type=compute_type, cpu_threads=6,
                                 download_root=str(cache))
        if device == "cuda":
            # Force CUDA inference before recording; VAD would skip the silent probe.
            segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="vi", beam_size=1,
                                          vad_filter=False, condition_on_previous_text=False, max_new_tokens=1)
            list(segments)
        return model
    except (RuntimeError, OSError) as error:
        if device == "cuda":
            raise RuntimeError(f"GPU CUDA could not start: {error}. Check the NVIDIA driver and CUDA runtime; run setup.bat after updating them. CPU fallback is disabled when a CUDA GPU is detected.") from error
        raise


class MeetingSession:
    def __init__(self, devices, language, events, directory, model_name="PhoWhisper-small"):
        if model_name not in MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        self.model_name = model_name
        self.devices = devices
        self.language = language
        self.events = events
        self.directory = Path(directory)
        self.audio = Queue(maxsize=12)
        self.stopped = Event()
        self.mic_enabled = Event()
        self.mic_enabled.set()
        self.failed = Event()
        self.thread = None
        self.captures = []
        self.manager = None
        self.audio_lock = Lock()
        self.started = 0

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Create a new session before restarting")
        self.thread = Thread(target=self._run, name="transcription", daemon=True)
        self.thread.start()

    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def stop(self):
        self.stopped.set()

    def _error(self, message):
        self.events.put(("error", message))
        self.stop()

    def _enqueue(self, item):
        if self.failed.is_set():
            return
        try:
            self.audio.put(item, timeout=0.25)
        except Full:
            raise RuntimeError("Audio queue is full: recognition cannot keep up. Recording stopped; the latest audio could not be transcribed.")

    def _capture(self, label, device):
        rate = int(device["defaultSampleRate"])
        channels = int(device["maxInputChannels"])
        frames = max(1, rate // 10)
        try:
            manager = self.manager
            while not self.stopped.is_set():
                if label == "You" and not self.mic_enabled.is_set():
                    self.events.put(("level", label, 0.0))
                    self.stopped.wait(0.1)
                    continue
                buffer = SpeechBuffer(rate)
                position = 0
                offset = time.monotonic() - self.started
                with self.audio_lock:
                    stream = manager.open(format=pyaudio.paFloat32, channels=channels, rate=rate,
                                          input=True, input_device_index=int(device["index"]),
                                          frames_per_buffer=frames)
                try:
                    while not self.stopped.is_set():
                        if label == "You" and not self.mic_enabled.is_set():
                            break
                        available = stream.get_read_available()
                        if available <= 0:
                            self.stopped.wait(0.02)
                            continue
                        raw = stream.read(min(frames, available), exception_on_overflow=True)
                        samples = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).mean(axis=1)
                        self.events.put(("level", label, float(np.sqrt(np.mean(samples ** 2)))))
                        for chunk in buffer.add(samples):
                            self._enqueue((label, offset + position / rate, to_16k(chunk, rate)))
                            position += len(chunk)
                    remainder = buffer.flush()
                    if len(remainder):
                        self._enqueue((label, offset + position / rate, to_16k(remainder, rate)))
                finally:
                    with self.audio_lock:
                        stream.close()
        except Exception as error:
            self._error(f"{label}: {error}")

    def _run(self):
        output = None
        try:
            self.events.put(("status", f"Loading {self.model_name} (first run may download it)..."))
            model = load_model(self.model_name)
            if self.stopped.is_set():
                return
            backend = backend_description(model)
            self.events.put(("backend", backend))
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / (datetime.now().strftime("meeting-%Y%m%d-%H%M%S-%f") + ".txt")
            output = path.open("x", encoding="utf-8", buffering=1)
            output.write(f"Local meeting transcript - {self.model_name}\nBackend: {backend}\n" + datetime.now().isoformat(timespec="seconds") + "\n\n")
            self.events.put(("file", str(path)))
            self.started = time.monotonic()
            # PortAudio initialization/termination must not race between capture threads.
            self.manager = pyaudio.PyAudio()
            for label, device in self.devices:
                if self.stopped.is_set():
                    break
                thread = Thread(target=self._capture, args=(label, device), name=f"capture-{label}", daemon=True)
                self.captures.append(thread)
                thread.start()
            self.events.put(("status", "Listening. Text updates after each short audio segment."))
            while any(t.is_alive() for t in self.captures) or not self.audio.empty():
                try:
                    label, offset, samples = self.audio.get(timeout=0.1)
                except Empty:
                    continue
                began = time.monotonic()
                # shortcut: uninterrupted speech is still split at 10 seconds, add overlap with text reconciliation if boundary errors persist.
                segments, _ = model.transcribe(samples, language=self.language,
                                               beam_size=5 if model.model.device == "cuda" else 1,
                                               vad_filter=True, condition_on_previous_text=False,
                                               vad_parameters={"min_silence_duration_ms": 400})
                for segment in segments:
                    text = segment.text.strip()
                    if text:
                        seconds = offset + segment.start
                        minutes, seconds = divmod(int(seconds), 60)
                        line = f"[{minutes:02d}:{seconds:02d}] {label}: {text}"
                        output.write(line + "\n")
                        output.flush()
                        self.events.put(("transcript", line))
                self.events.put(("timing", time.monotonic() - began, len(samples) / 16000))
        except Exception as error:
            self.failed.set()
            self._error(str(error))
        finally:
            self.stop()
            for thread in self.captures:
                thread.join()
            if self.manager is not None:
                try:
                    self.manager.terminate()
                except Exception as error:
                    self.events.put(("error", f"Could not close audio devices: {error}"))
            if output is not None:
                try:
                    output.close()
                except OSError as error:
                    self.events.put(("error", f"Could not finish transcript file: {error}"))
            self.events.put(("done",))
