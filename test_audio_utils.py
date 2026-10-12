import importlib
import importlib.util
import unittest

import numpy as np


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("audio_utils"), "audio conversion is missing")
        self.audio = importlib.import_module("audio_utils")

    def test_stereo_average_and_float32(self):
        samples = np.array([[0.2, 0.6], [-0.2, -0.6]], dtype=np.float32)
        result = self.audio.to_16k(samples, 16000)
        np.testing.assert_allclose(result, [0.4, -0.4])
        self.assertEqual(result.dtype, np.float32)

    def test_resample_preserves_duration_and_tone(self):
        samples = np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
        result = self.audio.to_16k(samples, 48000)
        self.assertEqual(len(result), 16000)
        peak = np.argmax(np.abs(np.fft.rfft(result)))
        self.assertEqual(peak, 440)

    def test_buffer_preserves_order_and_flushes_once(self):
        buffer = self.audio.AudioBuffer(10, seconds=1)
        self.assertEqual(buffer.add(np.arange(7)), [])
        chunks = buffer.add(np.arange(7, 25))
        self.assertEqual(len(chunks), 2)
        np.testing.assert_array_equal(chunks[0], np.arange(10))
        np.testing.assert_array_equal(chunks[1], np.arange(10, 20))
        np.testing.assert_array_equal(buffer.flush(), np.arange(20, 25))
        self.assertEqual(len(buffer.flush()), 0)

    def test_invalid_audio_and_rates_rejected(self):
        for samples, rate in [(np.array([np.nan]), 16000), (np.zeros(3), 0), (np.zeros((2, 2, 2)), 16000)]:
            with self.assertRaises(ValueError):
                self.audio.to_16k(samples, rate)

    def test_empty_audio_is_safe(self):
        self.assertEqual(len(self.audio.to_16k(np.array([]), 48000)), 0)


if __name__ == "__main__":
    unittest.main()
