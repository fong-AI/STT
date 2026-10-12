from math import gcd

import numpy as np
from scipy.signal import resample_poly


def to_16k(samples, rate):
    samples = np.asarray(samples, dtype=np.float32)
    if not isinstance(rate, int) or rate <= 0 or samples.ndim not in (1, 2):
        raise ValueError("Invalid sample rate or audio shape")
    if not np.isfinite(samples).all() or (samples.ndim == 2 and samples.shape[1] == 0):
        raise ValueError("Audio must have finite samples and at least one channel")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if len(samples) and rate != 16000:
        factor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // factor, rate // factor)
    return np.ascontiguousarray(samples, dtype=np.float32)


class AudioBuffer:
    def __init__(self, rate, seconds=5):
        self.size = int(rate * seconds)
        if self.size < 1:
            raise ValueError("Buffer length must be positive")
        self.pending = np.empty(0, dtype=np.float32)

    def add(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("Buffer requires mono audio")
        self.pending = np.concatenate((self.pending, samples))
        chunks = []
        while len(self.pending) >= self.size:
            chunks.append(self.pending[:self.size].copy())
            self.pending = self.pending[self.size:]
        return chunks

    def flush(self):
        remaining = self.pending
        self.pending = np.empty(0, dtype=np.float32)
        return remaining


class SpeechBuffer(AudioBuffer):
    def __init__(self, rate):
        super().__init__(rate, seconds=10)
        self.rate = rate
        self.check_at = rate * 5

    def add(self, samples):
        chunks = super().add(samples)
        if chunks:
            self.check_at = self.rate * 5
        if len(self.pending) >= self.check_at:
            from faster_whisper.vad import get_speech_timestamps
            self.check_at = len(self.pending) + self.rate // 2
            audio = to_16k(self.pending, self.rate)
            speech = get_speech_timestamps(audio, min_silence_duration_ms=400, speech_pad_ms=0)
            if not speech or speech[-1]["end"] <= len(audio) - 6400:
                chunks.append(self.flush())
                self.check_at = self.rate * 5
        return chunks
