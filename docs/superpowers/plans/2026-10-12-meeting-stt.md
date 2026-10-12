# Meeting STT Implementation Plan

> For agentic workers: use superpowers:executing-plans for inline execution. User approved implementation on 2026-10-12.

**Goal:** Transcribe Zoom/Google Meet audio and optional microphone locally on Windows.
**Architecture:** WASAPI capture feeds bounded queues and one faster-whisper CPU worker; Tkinter receives events, without blocking capture or the UI. Save final text immediately to a unique local file.
**Tech Stack:** Python 3.12, Tkinter, PyAudioWPatch, faster-whisper, numpy, scipy, unittest.
**Spec:** ../specs/2026-10-12-meeting-stt-design.md

## Global constraints
Windows only; model small, CPU INT8; no STT cloud requests, API keys or raw audio files. Default Vietnamese, English/auto selectable. No capture until Start. Stop flushes remainder and drains pending work. Loopback covers the selected output device; separate optional microphone. No changes to MMVN.

## Review focus
1. Stop during model load: no streams opened after stop (session test).
2. Device failure / queue overflow: stop and show error, no silent drop (session tests).
3. Short final remainder: flush once, preserve order (audio test).
4. Transcript write failure: stop session and surface error (session test).
5. Empty device list / window close: UI stays usable and cleans up (runtime/manual checks).

## Task 1: Audio conversion and bounded segmentation
Files: audio_utils.py, test_audio_utils.py, requirements.txt, .gitignore.
Interfaces: to_16k(samples, rate) -> float32 mono ndarray; AudioBuffer(rate, seconds=5).add(samples) -> list[ndarray], .flush() -> ndarray.
- [x] Write failing unittest assertions for stereo averaging, 48k to 16k duration, preserved tone, ordered segmentation and remainder flushed once; reject invalid/NaN input.
- [x] Run unittest; confirm missing implementation fails assertions.
- [x] Implement using resample_poly with gcd ratio and sample-count buffer at native rate.
- [x] Install in project .venv, run tests to green; commit.

## Task 2: Capture and transcription session
Files: session.py, test_session.py. Ruling: extract session from UI so lifecycle can be tested without capturing user's audio.
Interfaces: MeetingSession(devices, language, events, directory).start(), .stop(), .alive(); events queue carries status, level, transcript, error, done. devices are (label, device-info) pairs.
- [x] Write failing tests for cancellation before capture, overflow, trailing audio, file errors and worker failure. Mock hardware/model only, actual queues and files.
- [x] Implement delayed model load, one consumer, per-source capture threads, bounded queue, terminal error events, immediate file flush, resource cleanup.
- [x] Run all tests; verify no worker deadlocks and no session overlap; commit.

## Task 3: Runnable UI and verification
Files: meeting_stt.py, README.md, run.bat; requirements.txt pinned after install.
- [x] Add accessible native controls, source meters, scrolling transcript, language selector, Start/Stop; poll events from Tk main thread only. Microphone toggle during session must silence input; explain Zoom mute separately.
- [x] Provide --list-devices and --self-check modes without automatic user recording.
- [x] Verify imports, syntax, actual device enumeration, model download and silent inference; generate known English speech for benchmark if platform voice is available.
- [x] Test UI start/stop with explicit synthetic devices in tests; open real UI without recording if authorized by GUI approval.
- [x] Review final code, run full suite, document measured limits, commit and hand off run.bat.
