"""Regression tests for Studio desktop backend startup."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from triune import desktop


class TestDesktopLauncher(unittest.TestCase):
    def test_native_host_refuses_to_attach_to_an_existing_backend(self) -> None:
        with (
            patch.object(desktop, "_is_port_in_use", return_value=True),
            patch.object(desktop, "_is_triune_server_alive", return_value=True) as is_triune,
        ):
            with self.assertRaisesRegex(RuntimeError, "already in use"):
                desktop.launch_desktop_app(port=8123, native_host=True)

        is_triune.assert_not_called()

    def test_wsl_backend_stays_loopback_and_timeout_does_not_open_dead_window(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

        with (
            patch.object(desktop.sys, "platform", "win32"),
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(desktop, "_is_port_in_use", return_value=False),
            patch.object(desktop, "_find_wsl_python", return_value="/home/user/venv/bin/python"),
            patch.object(desktop, "_to_wsl_path", return_value="/mnt/c/TriuneTransformer"),
            patch.object(desktop, "_wait_for_server", return_value=False) as wait_for_server,
            patch.object(desktop.subprocess, "Popen", return_value=process) as popen,
            patch.object(desktop.subprocess, "CREATE_NO_WINDOW", 0, create=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not become ready"):
                desktop.launch_desktop_app(port=8123)

        command = popen.call_args.args[0][-1]
        self.assertEqual(wait_for_server.call_args.args[0], "http://127.0.0.1:8123/healthz")
        self.assertEqual(wait_for_server.call_args.kwargs["timeout"], 90.0)
        self.assertIn("TRIUNE_WINDOWS_PYTHON", command)
        self.assertIn("run_server(host='127.0.0.1', port=8123)", command)
        self.assertNotIn("host='0.0.0.0'", command)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=3.0)

    def test_native_host_mode_keeps_backend_alive_without_opening_pywebview(self) -> None:
        process = MagicMock()
        process.poll.return_value = None
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

        with (
            patch.object(desktop.sys, "platform", "win32"),
            patch.dict(sys.modules, {"torch": fake_torch}),
            patch.object(desktop, "_is_port_in_use", return_value=False),
            patch.object(desktop, "_find_wsl_python", return_value="/home/user/venv/bin/python"),
            patch.object(desktop, "_to_wsl_path", return_value="/mnt/c/TriuneTransformer"),
            patch.object(desktop, "_wait_for_server", return_value=True),
            patch.object(desktop, "_is_triune_server_alive", side_effect=[False, False, True]) as health_check,
            patch.object(desktop.subprocess, "Popen", return_value=process) as popen,
            patch.object(desktop.subprocess, "CREATE_NO_WINDOW", 0, create=True),
            patch.object(desktop.time, "sleep", side_effect=[None, None, KeyboardInterrupt]),
        ):
            desktop.launch_desktop_app(port=8123, native_host=True)

        command = popen.call_args.args[0][-1]
        self.assertIn("TRIUNE_WINDOWS_PYTHON", command)
        self.assertIn("run_server(host='127.0.0.1', port=8123)", command)
        self.assertEqual(health_check.call_count, 3)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=3.0)


if __name__ == "__main__":
    unittest.main()
