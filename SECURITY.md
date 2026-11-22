# Security Policy

## Supported Versions

The following versions of **Lucid Machines** (`lucid-machines` / the **Triune** architecture `TriuneTransformer` / **Polysom Studio**) are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 3.0.x   | :white_check_mark: |
| 2.1.x   | :white_check_mark: |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

---

## Reporting a Vulnerability

We take the security of Lucid Machines, its model architecture, and Polysom Studio very seriously. If you discover a security vulnerability, please do **NOT** open a public issue.

Instead, please send a private report to the maintainers:
- **Email**: `security@lucid-machines.org` (legacy alias: `security@triune-ai.org`, or directly contact the repository owner via GitHub Security Advisories)
- **Response Time**: We acknowledge receipt of vulnerability reports within 48 hours and strive to issue security advisories and patches within 7 business days.

Please include:
1. Description of the vulnerability and attack vector.
2. Steps to reproduce or proof-of-concept (PoC) code.
3. Affected versions, components, or configurations.
4. Suggested mitigation or patch if available.

---

## Security Architecture & Defenses (R4 Rock-Solid Hardening)

Lucid Machines (`lucid_machines`) implements defense-in-depth measures across all layers of the training, inference, DAG execution, plugin, and Polysom Studio serving stack:

### 1. Checkpoint & Adapter Deserialization Security (`weights_only=True`)
- Every PyTorch deserialization path across the runtime, `FineTuner.load_adapter`, API routes, `scripts/rescue_checkpoint.py`, and benchmark utilities enforces `torch.load(..., map_location="cpu", weights_only=True)` to prevent arbitrary pickle code execution and GPU memory exhaustion.
- **SafeTensors** (`.safetensors`) is the default format for LoRA adapters and model exports.

### 2. Strict Workspace Path Confinement
- All API and DAG file operations (`/v1/models/load`, `/v1/checkpoints/load`, `/v1/checkpoints/delete`, `/v1/checkpoints/export`, `/v1/datasets/local/register`, `/v1/datasets/select`, `/v1/models/export`, `/v1/finetune/start`, `/v1/finetune/adapter`, `/v1/finetune/export`, and DAG `LocalFileReader` / `CheckpointManager` / `SafeTensors` nodes) reject `..` traversal and UNC paths and enforce `Path.resolve(strict=False).is_relative_to(...)` against allowlisted workspace directories (`checkpoints`, `checkpoints_full`, `exports`, `data`, `datasets`, `models`, `outputs`).
- Export and adapter identifiers are validated against a strict alphanumeric regex (`^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$`).

### 3. Authentication, Reverse-Proxy Protection, CSRF & Security Headers
- Non-loopback requests and any request carrying reverse-proxy forwarding headers (`Forwarded`, `X-Forwarded-For`, `X-Real-IP`) or running with `LUCID_MACHINES_REQUIRE_AUTH=1` / `TRIUNE_REQUIRE_AUTH=1` require a configured `LUCID_MACHINES_API_TOKEN` (or legacy `TRIUNE_API_TOKEN`) of at least 32 characters, verified via constant-time `hmac.compare_digest` on both HTTP endpoints and `/ws/telemetry`.
- State-changing HTTP methods (`POST`, `PUT`, `PATCH`, `DELETE`) and `/ws/telemetry` WebSocket handshakes enforce `Origin` and `Host` allowlists to prevent cross-site request forgery (CSRF), cross-site WebSocket hijacking (CSWSH), and DNS rebinding.
- Every HTTP response includes hardened security headers: `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and a strict `Content-Security-Policy`.
- Request payload size caps (`LUCID_MACHINES_MAX_REQUEST_BYTES`) and configurable per-client rate limiting (`LUCID_MACHINES_RATE_LIMIT_PER_MIN`) guard against denial-of-service.

### 4. SSRF Protection & Curated Package Installation Gate
- Dataset URL streaming (`connect_url_stream`, `/v1/datasets/select`, `/v1/datasets/stream/connect`), agent HTTP fetching, and module repository/artifact downloads enforce `validate_http_url` across all redirect hops, blocking loopback, private (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local, and cloud metadata (`169.254.169.254`) endpoints.
- `/v1/modules/install` gates `pip` package installation behind an explicit curated allowlist unless `LUCID_MACHINES_ALLOW_PIP_INSTALL=1` is explicitly enabled by the host administrator.
- Cloned or imported module archives are validated against directory traversal (ZipSlip) before extraction.

### 5. Plugin Ed25519 Signature Verification, Sandbox Resource Limits & Secrets Hygiene
- External workspace and third-party plugin discovery (`discover_and_register_nodes`, `scan_workspace_plugins`, `plugins.loader`) defaults to `trusted=False`, requiring an Ed25519-signed version-1 plugin manifest with matching SHA-256 file hashes and strict AST validation before any module is loaded.
- `PythonSandbox` executes restricted code in an isolated subprocess (`python -I -S -u`) with sanitized environment variables, AST/import allowlists, timeouts, POSIX `RLIMIT_AS`, and Windows Job Object (`JOB_OBJECT_LIMIT_PROCESS_MEMORY` + `ActiveProcessLimit = 1`) memory and process caps.
- Autonomous agent workspace file writes via `/v1/agents/run` and `/v1/agents/orchestrate` are disabled by default unless `LUCID_MACHINES_ALLOW_AGENT_WRITES=1` is set.
- vLLM serving configurations (`vllm_bridge.py` and `/v1/models/export`) default to `"trust_remote_code": False`.
- Google Gemini API requests transmit credentials exclusively via the `x-goog-api-key` HTTP header and never in URL query strings (`?key=`).
- `/v1/system/config` masks stored credentials with partial-prefix obfuscation and persists BYOK secrets in the OS credential store (`Polysom Studio`, with legacy `"Triune Studio"` fallback), failing closed if no secure credential store is available.
