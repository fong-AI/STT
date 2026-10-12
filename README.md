# Meeting STT local (Windows)

Local transcription for Google Meet (browser) and Zoom (desktop or browser).
No cloud STT, API key, account or per-minute STT charges. Python 3.12+.

## Run
1. Open run.bat in this folder. Dependencies are installed in .venv.
2. Set Meeting audio output to the same headphones/speakers used by Zoom/Meet.
3. Select Vietnamese (recommended). English and Auto detect remain available, but this model is tuned for Vietnamese.
4. Your microphone is enabled by default when a microphone is available. Select the same microphone as Zoom/Meet, or untick Include my microphone to capture meeting output only.
5. Choose Model: PhoWhisper-small (default, about 1 GB) or PhoWhisper-medium (about 3 GB), then click Start. Both models are already cached on this machine.
6. Speak/listen; draft captions above the transcript update about every second plus processing time. Stable words appear below and are saved; draft text may change.
7. Click Stop and wait for pending audio to finish. Transcripts are saved in transcripts/.

Use setup.bat to install dependencies and download the model on a fresh machine.
After the model is cached in models/, inference runs offline. The microphone is
only captured after Start when Include my microphone is enabled. You can toggle it during a session. Microphone mute
in Zoom/Meet DOES NOT mute this app. Loopback includes all sound played through
the selected device, including other apps/notifications. Use headphones to avoid
capturing meeting sound again through your microphone.

## Limits
- Automatic NVIDIA CUDA GPU detection on each machine; CPU INT8 is used only when no CUDA GPU is detected. AMD/Intel GPU acceleration is not supported by this build.
- Streaming is emulated by repeatedly decoding a growing audio window. This is not a native streaming Whisper model. Continuous speech is capped at 12 seconds with 0.5 seconds of overlap; forced boundaries can still cause errors.
- When both sources speak, processing is shared. Slow hardware can increase latency. Queued audio is coalesced for fewer previews without dropping samples.
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
- 49 automated tests passed: audio conversion, streaming reconciliation, microphone boundaries, bounded windows, shutdown, errors and real Tk UI with simulated sessions.
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

## Automatic GPU use
At Start, the app detects CUDA devices with CTranslate2 and uses the first NVIDIA GPU. It selects a compute type supported by that GPU: INT8/FP16 first, then FP16 or FP32. There is no hardcoded RTX model name, drive, CUDA Toolkit path or GPU configuration. Machines without a detected CUDA GPU run on CPU INT8. The processing device stays visible below the status line and is written to the transcript header.

On another Windows PC, install Python 3.12+ and a current NVIDIA driver compatible with CUDA 12.8, then run setup.bat. Setup installs the pinned CTranslate2 engine and, when CUDA is detected, cuBLAS into that PC's .venv. The app locates the DLL directory from its current Python environment; a separate CUDA Toolkit installation is not needed for this pinned Windows build. Model weights are cached/downloaded in the project folder. A missing driver, runtime error or insufficient GPU memory is reported; a detected GPU does not silently fall back to CPU. Close an old app window and reopen run.bat after updating the code.

The app checks actual GPU inference before opening capture streams, since a VAD silence check alone would not exercise CUDA. GPU recognition uses beam search with width 5; the CPU path retains width 1. Recognition now uses incremental 1-second capture and repeated decoding of the current speech window, with separate draft captions for each source.

RTX 3060 verification: the same 6.61-second Vietnamese sample took 0.58 seconds with Small and 1.05 seconds with Medium (GPU INT8/FP16, beam 5), versus the earlier CPU results of 2.55 and 7.58 seconds (CPU INT8, beam 1). The settings differ, so this is an app throughput comparison, not an isolated hardware speedup or accuracy score. Report: models/pho-gpu-comparison.json. CUDA Toolkit environment variables/paths were removed for the final check; loaded cuBLAS DLLs came from .venv. Other physical GPUs and a live meeting have not been tested.

## Your voice is missing
Meeting output contains remote participants; your own voice must come from the separate microphone source. Include my microphone is now enabled by default when the app starts with a microphone available. Recording still begins only at Start. If there is no microphone, the checkbox is disabled and meeting output can still be transcribed.

Select the same microphone used in Zoom/Meet. You can toggle Include my microphone during a session; the You meter should move when you speak. To change the microphone device, Stop, choose Microphone, and Start. If the You meter remains flat with capture enabled, check the selected input, its mute/volume and Windows microphone access. If You moves but there is no text, check the draft captions, pause briefly and check the session error/status.

## Streaming captions
Capture emits one second of audio per source. The worker uses the existing Silero VAD and PhoWhisper model, showing drafts separately for You and Meeting. A word prefix agreed on by two successive decodes is saved, with the newest word held back for context. A trailing pause of at least 0.5 seconds, Stop or microphone disable finalizes the remaining text. Drafts are never written to the transcript file. Consecutive confirmed chunks from the same source append to one paragraph, with one timestamp at its start. A source change starts a new paragraph: You, then Meeting, then You. This grouping applies to both the screen and the saved file; even a short pause does not split a paragraph while the source stays the same. All remote participants share the Meeting source, so changes between remote voices are not detected.

The recognition window is at most 12 seconds. Continuous speech uses 0.5 seconds of acoustic overlap and committed-word reconciliation to reduce duplicates. Timestamp drift and legitimate repeated words are covered by regression tests. If recognition falls behind, queued chunks are combined before the next preview; no raw samples are discarded. A full queue still produces a visible error rather than silently losing audio.

The approach uses local agreement, as described in [Whisper Streaming](https://github.com/ufal/whisper_streaming), implemented with the dependencies already installed here. No extra service, model or package was added. PhoWhisper's word alignment caused the decoder to revisit the tiny end of the window and hallucinate additional sentences. The bounded window is decoded once with word alignment; the short tail is not decoded separately.

Actual PhoWhisper-medium / GPU CUDA INT8-FP16 on RTX 3060, feeding the public Vietnamese sample at real-time speed: first draft at **1.312 seconds**, first saved words at **2.390 seconds**. The 6.612-second sample plus one second of silence finished at 8.297 seconds; decoding each growing window took about 0.27-0.61 seconds. The final saved text contained the sample sentence without the previous duplicate boundary word or added tail sentence. Report: models/pho-streaming-benchmark.json; reproducible helper: models/benchmark_streaming.py.

This is one public sample with one source, not a latency or accuracy guarantee for live meetings. Drafts can contain wrong words and are revised as context arrives; a wrong phrase that agrees twice can still be saved. Both sources speaking simultaneously, long uninterrupted speech, CPU performance and real Zoom/Meet audio need live evaluation. Close the old app window and reopen run.bat to load this version.
