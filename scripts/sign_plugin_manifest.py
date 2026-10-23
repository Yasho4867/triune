"""Create or refresh a signed Triune plugin manifest using an Ed25519 private key.

Keep the signing key in a trusted release environment. This tool prompts for the
base64-encoded raw 32-byte private key without echoing or writing it to disk.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def sign_manifest(plugin_dir: Path, key_id: str, private_key_b64: str) -> str:
    manifest_path = plugin_dir / "triune_plugin.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Manifest root must be a JSON object.")
    if type(manifest.get("manifest_version")) is not int or manifest["manifest_version"] != 1:
        raise ValueError("Set manifest_version to 1 before signing.")
    if not isinstance(manifest.get("name"), str) or not manifest["name"].strip():
        raise ValueError("Manifest must contain a name.")
    if not isinstance(manifest.get("version"), str) or not re.fullmatch(
        r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?", manifest["version"]
    ):
        raise ValueError("Manifest version must use semantic versioning, for example 1.2.3.")
    entrypoints = manifest.get("entrypoints")
    if not isinstance(entrypoints, list) or not 1 <= len(entrypoints) <= 32:
        raise ValueError("Manifest must declare between 1 and 32 Python entrypoints.")

    root = plugin_dir.resolve()
    files: dict[str, str] = {}
    for entrypoint in entrypoints:
        if not isinstance(entrypoint, str) or "\\" in entrypoint or Path(entrypoint).is_absolute():
            raise ValueError("Entrypoints must be relative POSIX paths.")
        relpath = Path(entrypoint)
        path = (root / relpath).resolve()
        if relpath.suffix != ".py" or ".." in relpath.parts or root not in path.parents or not path.is_file():
            raise ValueError(f"Invalid plugin entrypoint: {entrypoint}")
        data = path.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            raise ValueError(f"Plugin entrypoint exceeds the 2 MiB safety limit: {entrypoint}")
        files[entrypoint] = hashlib.sha256(data).hexdigest()
    manifest["files"] = files
    manifest.pop("signature", None)

    private_bytes = base64.b64decode(private_key_b64.strip(), validate=True)
    private_key = Ed25519PrivateKey.from_private_bytes(private_bytes)
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["signature"] = {
        "algorithm": "ed25519",
        "key_id": key_id,
        "value": base64.b64encode(private_key.sign(canonical)).decode("ascii"),
    }

    fd, temp_name = tempfile.mkstemp(prefix="triune_plugin_manifest_", suffix=".tmp", dir=plugin_dir)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, manifest_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(public_key).decode("ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin_dir", type=Path, help="Directory containing triune_plugin.json and plugin entrypoints")
    parser.add_argument("--key-id", required=True, help="Stable identifier for the public key trusted by Triune users")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", args.key_id):
        parser.error("--key-id must contain only letters, digits, dot, underscore, or hyphen")
    private_key = getpass.getpass("Base64-encoded raw Ed25519 private key (32 bytes): ")
    try:
        public_key = sign_manifest(args.plugin_dir, args.key_id, private_key)
    except Exception as exc:
        parser.error(str(exc))
    print(f"Signed manifest: {args.plugin_dir / 'triune_plugin.json'}")
    print(f"Publish this public key out-of-band as trusted key '{args.key_id}': {public_key}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
