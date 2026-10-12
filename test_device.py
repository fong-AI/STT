import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
import session


class DeviceTests(TestCase):
    def test_nvidia_gpu_selected_without_machine_specific_names(self):
        with patch("ctranslate2.get_cuda_device_count", return_value=1), patch("ctranslate2.get_supported_compute_types", return_value={"int8_float16", "float16", "float32"}):
            self.assertEqual(session.inference_device(), ("cuda", "int8_float16"))

    def test_older_cuda_gpu_uses_supported_float32(self):
        with patch("ctranslate2.get_cuda_device_count", return_value=1), patch("ctranslate2.get_supported_compute_types", return_value={"float32"}):
            self.assertEqual(session.inference_device(), ("cuda", "float32"))

    def test_machine_without_cuda_gpu_uses_cpu(self):
        with patch("ctranslate2.get_cuda_device_count", return_value=0), patch("ctranslate2.get_supported_compute_types") as types:
            self.assertEqual(session.inference_device(), ("cpu", "int8"))
        types.assert_not_called()

    def test_broken_cuda_detection_is_not_hidden(self):
        with patch("ctranslate2.get_cuda_device_count", side_effect=RuntimeError("driver failure")):
            with self.assertRaisesRegex(RuntimeError, "driver failure"):
                session.inference_device()

    def test_gpu_without_int8_float16_uses_supported_float16(self):
        with patch("ctranslate2.get_cuda_device_count", return_value=1), patch("ctranslate2.get_supported_compute_types", return_value={"float16", "float32"}):
            self.assertEqual(session.inference_device(), ("cuda", "float16"))

    def test_cuda_dlls_are_located_in_current_python_environment(self):
        with TemporaryDirectory() as folder:
            library = Path(folder) / "nvidia" / "cublas" / "bin"
            library.mkdir(parents=True)
            handle = object()
            with patch("session.sysconfig.get_path", return_value=folder), patch("session.os.add_dll_directory", return_value=handle) as register, patch.dict(os.environ):
                self.assertIs(session.configure_cuda_runtime.__wrapped__(), handle)
                self.assertIn(str(library), os.environ["PATH"].split(os.pathsep))
                register.assert_called_once_with(str(library))
