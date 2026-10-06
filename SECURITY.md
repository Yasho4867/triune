# Security Policy

## Supported Versions

The following versions of the Triune Transformer project are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 2.1.x   | :white_check_mark: |
| 2.0.x   | :white_check_mark: |
| < 2.0   | :x:                |

---

## Reporting a Vulnerability

We take the security of Triune Transformer and the safety of our users very seriously. If you discover a security vulnerability, please do **NOT** open a public issue.

Instead, please send a private report to the maintainers:
- **Email**: `security@triune-ai.org` (or directly contact the repository owner via GitHub Security Advisories)
- **Response Time**: We acknowledge receipt of vulnerability reports within 48 hours and strive to issue security advisories and patches within 7 business days.

Please include:
1. Description of the vulnerability and attack vector.
2. Steps to reproduce or proof-of-concept (PoC) code.
3. Affected versions, components, or configurations.
4. Suggested mitigation or patch if available.

---

## Security Architecture & Defenses

Triune Transformer implements defense-in-depth measures against common threat vectors in AI training and runtime systems:

### 1. Checkpoint Deserialization Security
- Standard checkpoint loading employs CPU-first hydration (`map_location="cpu"`) to prevent GPU memory exhaustion attacks.
- Checkpoint loading in the API enforces extension whitelist validation (`.pt`, `.bin`, `.safetensors`, `.ckpt`) and strictly checks file path confinement to prevent path traversal outside declared project root directories.
- We strongly recommend using **SafeTensors** (`.safetensors`) for distributing model weights across untrusted environments to avoid arbitrary Python code execution via `torch.load` pickle deserialization.

### 2. Community Module & Archive Ingestion (ZipSlip Prevention)
- Cloned or imported module archives are rigorously validated against directory traversal attacks (ZipSlip).
- Every archive member path is resolved to ensure it resides strictly within the target extraction root.
- Dynamic plugin modules cannot overwrite core built-in nodes and are namespaced under `community.<module_name>.*`.

### 3. API Credential & BYOK Masking
- The `/v1/system/config` API masks all external provider API keys (OpenAI, Anthropic, Google Gemini, Hugging Face) using partial-prefix obfuscation (`sk-...39a1`).
- Saved configuration writes never overwrite existing stored secrets with masked strings.

### 4. Sandbox Isolation
- Untrusted user code submitted to `PythonSandbox` runs in an isolated subprocess with strict memory caps, timeouts, and sanitized environment variables (`PATH`, `HOME`, `PYTHONPATH`), preventing direct parent process memory mutation.
