"""Regression tests for failed module and config operations."""

from __future__ import annotations

import tempfile
import unittest
import base64
import hashlib
import json
import os
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from triune.modules.manager import ModuleManager, _verified_plugin_entrypoints


class TestModuleManagerFailures(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="triune_module_test_")
        self.manager = ModuleManager(config_dir=Path(self.temp_dir.name) / "config")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_failed_package_install_is_not_recorded_as_installed(self) -> None:
        with patch("triune.modules.manager.subprocess.check_call", side_effect=RuntimeError("offline")):
            result = self.manager.install_module({"id": "test-broken-package", "package_name": "test-package"})

        self.assertFalse(result["success"])
        self.assertNotIn("test-broken-package", {m["id"] for m in self.manager.get_installed_modules()})
        self.assertFalse((self.manager.modules_dir / "test-broken-package").exists())

    def test_failed_checkpoint_download_does_not_create_fake_artifact(self) -> None:
        with patch("triune.modules.manager.urllib.request.urlretrieve", side_effect=OSError("offline")):
            result = self.manager.install_module({
                "id": "test-missing-weights",
                "name": "Missing Weights",
                "download_url": "https://example.invalid/model.safetensors",
                "type": "model",
            })

        self.assertFalse(result["success"])
        self.assertFalse((self.manager.modules_dir / "test-missing-weights" / "model.safetensors").exists())
        self.assertNotIn("test-missing-weights", {m["id"] for m in self.manager.get_installed_modules()})

    def test_uninstall_failure_preserves_module_registration(self) -> None:
        module_dir = self.manager.modules_dir / "test-protected-module"
        module_dir.mkdir(parents=True)
        self.manager._save_installed_modules([{
            "id": "test-protected-module",
            "name": "Protected Module",
            "installed_at": str(module_dir),
        }])

        with patch("triune.modules.manager.shutil.rmtree", side_effect=PermissionError("locked")):
            result = self.manager.uninstall_module("test-protected-module")

        self.assertFalse(result["success"])
        self.assertIn("test-protected-module", {m["id"] for m in self.manager.get_installed_modules()})

    def test_config_write_failure_is_reported_to_the_api(self) -> None:
        with patch("triune.modules.manager.os.replace", side_effect=OSError("read-only")):
            with self.assertRaisesRegex(RuntimeError, "Could not persist Studio configuration"):
                self.manager.save_config({"example": True})

    def test_remote_module_cannot_gain_trust_by_using_a_curated_id(self) -> None:
        remote_path = self.manager._module_path("triune-base-weights")
        self.assertIsNotNone(remote_path)
        remote_path.mkdir()
        with patch.object(self.manager, "get_installed_modules", return_value=[{
            "id": "triune-base-weights",  # Deliberately collides with a curated registry ID.
            "installed_at": str(remote_path),
            "repo_url": "https://example.invalid/untrusted.git",
        }]), patch.object(self.manager, "discover_and_register_nodes") as discover:
            self.manager._auto_register_installed_nodes()

        discover.assert_not_called()

    def test_invalid_module_id_cannot_escape_managed_modules_directory(self) -> None:
        result = self.manager.install_module({"id": "../../outside", "name": "Outside"})
        self.assertFalse(result["success"])
        self.assertFalse((Path(self.temp_dir.name) / "outside").exists())
        self.assertFalse(self.manager.uninstall_module("../../outside")["success"])
        self.assertFalse(self.manager.git_pull("../../outside")["success"])

    def test_cloning_plugin_does_not_execute_or_register_its_python(self) -> None:
        inspection = {
            "git_info": {"commit": "abc123"},
            "file_count": 1,
            "size_mb": 0.01,
            "artifacts": {"adapters": [], "weights": [], "datasets": [], "python_files": ["plugin.py"]},
        }
        with patch("triune.modules.manager.subprocess.run", return_value=SimpleNamespace(
            returncode=0, stdout="cloned", stderr=""
        )), patch.object(self.manager, "inspect_directory", return_value=inspection), patch.object(
            self.manager, "discover_and_register_nodes"
        ) as discover:
            result = self.manager.clone_repository(
                "https://github.com/example/plugin.git", module_type="plugin", custom_name="review-plugin"
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["module"]["registered_nodes"], [])
        self.assertTrue(result["module"]["requires_user_activation"])
        discover.assert_not_called()

    def test_local_plugin_template_rejects_outside_paths_and_preserves_existing_files(self) -> None:
        outside = Path(self.temp_dir.name) / "outside"
        result = self.manager.create_custom_plugin_template("Example", destination_dir=str(outside))
        self.assertFalse(result["success"])
        self.assertFalse(outside.exists())

        with tempfile.TemporaryDirectory(dir=self.manager._workspace_root(), prefix=".plugin-template-test-") as workspace_name:
            workspace = Path(workspace_name)
            target = workspace / "plugin_example.py"
            target.write_text("user-owned content\n", encoding="utf-8")
            result = self.manager.create_custom_plugin_template("Example", destination_dir=str(workspace))
            self.assertFalse(result["success"])
            self.assertEqual(target.read_text(encoding="utf-8"), "user-owned content\n")

    def test_plugin_template_escapes_names_and_categories_as_python_literals(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.manager._workspace_root(), prefix=".plugin-template-escape-test-") as workspace_name:
            workspace = Path(workspace_name)
            with patch.object(self.manager, "discover_and_register_nodes") as discover:
                result = self.manager.create_custom_plugin_template(
                    "2; import os", category='Custom\")\nimport os\n#', destination_dir=str(workspace)
                )
            discover.assert_not_called()
            self.assertTrue(result["success"], result)
            source = Path(result["file_path"]).read_text(encoding="utf-8")
            compile(source, result["file_path"], "exec")
            self.assertNotIn("\nimport os\n#", source)

    def test_plugin_template_handles_non_string_name_without_crashing(self) -> None:
        with tempfile.TemporaryDirectory(dir=self.manager._workspace_root(), prefix=".plugin-template-input-test-") as workspace_name:
            result = self.manager.create_custom_plugin_template(None, destination_dir=workspace_name)
            self.assertTrue(result["success"], result)
            self.assertEqual(result["node_name"], "CustomNode")

    def test_workspace_scan_rejects_external_directories(self) -> None:
        result = self.manager.scan_workspace_plugins(self.temp_dir.name)
        self.assertFalse(result["success"])
        self.assertEqual(result["discovered_nodes"], [])

    def test_user_api_keys_are_written_to_os_keyring_not_json(self) -> None:
        class FakeSecureBackend:
            __module__ = "keyring.backends.Windows"
            priority = 1

        class FakeKeyring:
            def __init__(self):
                self.backend = FakeSecureBackend()
                self.values = {}

            def get_keyring(self):
                return self.backend

            def set_password(self, service, username, password):
                self.values[(service, username)] = password

            def get_password(self, service, username):
                return self.values.get((service, username))

            def delete_password(self, service, username):
                self.values.pop((service, username), None)

        fake_keyring = FakeKeyring()
        with patch.dict("sys.modules", {"keyring": fake_keyring}), patch.object(
            ModuleManager, "_get_secure_keyring", return_value=fake_keyring
        ):
            self.manager.save_config({"byok_keys": {"openai": "sk-secret-test-value"}})
            stored = self.manager.config_file.read_text(encoding="utf-8")
            self.assertNotIn("sk-secret-test-value", stored)
            self.assertIn("[STORED_IN_OS_KEYRING]", stored)
            self.assertEqual(self.manager.get_config()["byok_keys"]["openai"], "sk-secret-test-value")

    def test_api_key_save_fails_closed_without_secure_credential_backend(self) -> None:
        with patch.object(ModuleManager, "_get_secure_keyring", side_effect=RuntimeError("no secure store")):
            with self.assertRaisesRegex(RuntimeError, "Could not store byok:openai"):
                self.manager.save_config({"byok_keys": {"openai": "sk-secret-test-value"}})
        if self.manager.config_file.exists():
            self.assertNotIn("sk-secret-test-value", self.manager.config_file.read_text(encoding="utf-8"))

    def test_explicit_byok_save_uses_secure_store_even_when_same_value_is_in_environment(self) -> None:
        secret = "hf_test_token_that_must_persist"
        with patch.dict(os.environ, {"HF_TOKEN": secret}), patch.object(
            ModuleManager, "_write_secure_secret"
        ) as write_secret:
            self.manager.save_config(
                {"byok_keys": {"huggingface": secret}},
                force_secure_secrets={"byok:huggingface"},
            )

        write_secret.assert_called_once_with("byok:huggingface", secret)
        stored = json.loads(self.manager.config_file.read_text(encoding="utf-8"))
        self.assertEqual(stored["byok_keys"]["huggingface"], "[STORED_IN_OS_KEYRING]")
        self.assertNotIn(secret, self.manager.config_file.read_text(encoding="utf-8"))

    def test_wsl_credential_bridge_uses_windows_python_and_stdin(self) -> None:
        fake_windows_python = Path(self.temp_dir.name) / "python.exe"
        fake_windows_python.write_bytes(b"test placeholder")
        result = SimpleNamespace(returncode=0, stdout='{"value":"stored-secret"}', stderr="")
        with patch.dict(os.environ, {"TRIUNE_WINDOWS_PYTHON": str(fake_windows_python)}), patch(
            "triune.modules.manager.subprocess.run", return_value=result
        ) as run:
            bridge = ModuleManager._get_secure_keyring()
            bridge.set_password("Triune Studio", "byok:openai", "new-secret")
            self.assertEqual(bridge.get_password("Triune Studio", "byok:openai"), "stored-secret")
            bridge.delete_password("Triune Studio", "byok:openai")

        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args_list[0].args[0][0], str(fake_windows_python))
        set_payload = json.loads(run.call_args_list[0].kwargs["input"])
        self.assertEqual(set_payload["password"], "new-secret")
        self.assertNotIn("new-secret", " ".join(run.call_args_list[0].args[0]))
        from triune.modules.manager import _WindowsCredentialStoreBridge
        self.assertIn("CredWriteW", _WindowsCredentialStoreBridge._HELPER)
        self.assertIn("CredReadW", _WindowsCredentialStoreBridge._HELPER)
        self.assertNotIn("import keyring", _WindowsCredentialStoreBridge._HELPER)

    def test_native_windows_credential_store_does_not_require_keyring_package(self) -> None:
        from triune.modules.manager import _WindowsCredentialStoreBridge

        with patch("triune.modules.manager.sys.platform", "win32"), patch.dict(
            os.environ, {"TRIUNE_WINDOWS_PYTHON": ""}
        ):
            bridge = ModuleManager._get_secure_keyring()

        self.assertIsInstance(bridge, _WindowsCredentialStoreBridge)
        self.assertEqual(bridge.python_executable, sys.executable)

    @unittest.skipUnless(__import__("importlib").util.find_spec("cryptography"), "cryptography is not installed")
    def test_external_plugin_requires_trusted_signature_and_source_hash(self) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        plugin_dir = Path(self.temp_dir.name) / "signed-plugin"
        plugin_dir.mkdir()
        entrypoint = plugin_dir / "plugin.py"
        entrypoint.write_text("from triune.plugins.registry import register_node\n@register_node('SignedDemo')\ndef run(x): return x\n", encoding="utf-8")
        private_key = Ed25519PrivateKey.generate()
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        manifest = {
            "manifest_version": 1,
            "name": "Signed Demo",
            "version": "1.0.0",
            "entrypoints": ["plugin.py"],
            "permissions": [],
            "files": {"plugin.py": hashlib.sha256(entrypoint.read_bytes()).hexdigest()},
        }
        canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        manifest["signature"] = {
            "algorithm": "ed25519",
            "key_id": "test-key",
            "value": base64.b64encode(private_key.sign(canonical)).decode("ascii"),
        }
        (plugin_dir / "triune_plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        with patch.dict(os.environ, {"TRIUNE_PLUGIN_TRUSTED_KEYS": json.dumps({
            "test-key": base64.b64encode(public_bytes).decode("ascii")
        })}):
            paths, error = _verified_plugin_entrypoints(plugin_dir)
            self.assertIsNone(error)
            self.assertEqual(paths, [entrypoint.resolve()])

            entrypoint.write_text(entrypoint.read_text(encoding="utf-8") + "# changed after signing\n", encoding="utf-8")
            paths, error = _verified_plugin_entrypoints(plugin_dir)
            self.assertEqual(paths, [])
            self.assertIn("hash mismatch", error)


if __name__ == "__main__":
    unittest.main()
