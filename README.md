# Meeting STT local (Windows)

Local transcription for Google Meet (browser) and Zoom (desktop or browser).
No cloud STT, API key, account or per-minute STT charges. Python 3.12+.

## Run
1. Open run.bat in this folder. Dependencies are installed in .venv.
2. Set Meeting audio output to the same headphones/speakers used by Zoom/Meet.
3. Select Vietnamese (recommended). English and Auto detect remain available, but this model is tuned for Vietnamese.
4. Optionally select your microphone and tick Include my microphone.
5. Choose Model: PhoWhisper-small (default, about 1 GB) or PhoWhisper-medium (about 3 GB), then click Start. Both models are already cached on this machine.
6. Speak/listen; text arrives after audio chunks of up to 5 seconds plus processing time.
7. Click Stop and wait for pending audio to finish. Transcripts are saved in transcripts/.

Use setup.bat to install dependencies and download the model on a fresh machine.
After the model is cached in models/, inference runs offline. The microphone is
not captured until enabled. You can toggle it during a session. Microphone mute
in Zoom/Meet DOES NOT mute this app. Loopback includes all sound played through
the selected device, including other apps/notifications. Use headphones to avoid
capturing meeting sound again through your microphone.

## Limits
- CPU INT8 PhoWhisper-small or PhoWhisper-medium; RTX 3060 is not used in this first version.
- Chunk boundaries can split words. Text is segment-based, not word-by-word live captions.
- Labels are You and Meeting; no identification of individual meeting participants.
- Device changes require Stop, Refresh devices and Start.
- A full audio queue or device/file/model error stops capture and shows a visible error.
  Recent audio can be incomplete after an error; already written text stays saved.
- Raw audio is never saved. Model weights and installed packages require disk space.
- Stop during the initial model download waits for the current download/load to return;
  no capture starts after Stop is requested.

## Checks
.venv\Scripts\python.exe -m unittest discover -v
.venv\Scripts\python.exe meeting_stt.py --list-devices
.venv\Scripts\python.exe meeting_stt.py --self-check

Self-check uses generated silence and does not open a microphone or output capture.

## Verification on this machine
- 22 automated tests passed: conversion, segmentation, shutdown, errors and real Tk UI with simulated sessions.
- Actual Windows output devices enumerated, model downloaded into this folder, silent audio produced no transcript.
- Previous generic Whisper-small baseline, Windows-generated English speech: 8.96 seconds decoded in 2.14 seconds total on CPU (two chunks).
- Actual default output loopback -> STT -> transcript file -> Stop passed with the generated speech.
- PhoWhisper-small: silence check passed; a public Vietnamese speech sample of 6.61 seconds produced Vietnamese text in 2.38 seconds total on CPU (5-second chunks). Result: models/pho-benchmark.json.
- No live Zoom/Meet session or real microphone capture tested. Vietnamese accuracy on real meetings has not been measured.
- Independent review found no remaining important issue after cancellation/status fixes.

## Start failure correction
A native PortAudio assertion was observed after the initial handoff. Capture
threads were each initializing a separate PyAudio manager concurrently. The
session now owns one manager; stream open/close operations are serialized, and
termination happens after every capture thread exits. A regression test and
five actual loopback Start/Stop cycles passed with microphone disabled. Both
available microphone formats were checked without opening microphone capture.
The exact error text reported by the user has not yet been supplied; additional
errors may need separate diagnosis.

## Vietnamese model
The default is [VinAI PhoWhisper-small](https://huggingface.co/vinai/PhoWhisper-small), using the [qbsmlabs CTranslate2 conversion](https://huggingface.co/qbsmlabs/PhoWhisper-small) supported by faster-whisper. The converted weights are published by the community, not VinAI. No API key is required. Cached model files are reused offline. The previous generic Whisper-small cache and its unused weights were removed at the user's request (about 484 MB). PhoWhisper-small remains the default.

## Try Small versus Medium
Open run.bat, select Vietnamese and choose Model before Start. Stop and wait for pending text before choosing another model. Each saved transcript records the model name in its header. Small remains selected when the app opens.

Medium uses the [qbsmlabs conversion](https://huggingface.co/qbsmlabs/PhoWhisper-medium) of [VinAI PhoWhisper-medium](https://huggingface.co/vinai/PhoWhisper-medium), with the same installed faster-whisper engine. No new dependency or STT API is needed.

Both models passed silence checks and recognized the same public Vietnamese sample. CPU INT8 comparison, 6.61 seconds of audio split at 5 seconds: Small took 2.55 seconds; Medium took 7.58 seconds. This sample has no verified reference transcript, so it does not establish which model is more accurate. Results: models/pho-comparison.json. Medium may fall behind live speech, especially when both meeting output and microphone are enabled; a full queue stops capture with an error. Compare using the same spoken material and check the saved text against what was said.

To verify Medium without recording devices:
.venv\Scripts\python.exe meeting_stt.py --self-check --model PhoWhisper-medium
