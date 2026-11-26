# Deployment & Release Publishing Guide — Lucid Machines `v1.0.0`

This guide provides step-by-step, reproducible instructions for building, verifying, signing, and deploying all **Lucid Machines v1.0.0** (the **Triune** architecture **`TriuneTransformer`**, inaugural flagship model configurations **`Junkie` / `Junkie-7B` / `Junkie-3B` / `Junkie-1B` / `Junkie-Mini`**, **`kcx`** CLI, & **Polysom Studio v1.0.0**) release artifacts across **GitHub Releases**, **GitHub Pages**, **PyPI**, **Vercel / Netlify / Cloudflare Pages**, **Hugging Face Hub**, **Google Colab Pro**, and the **Windows Native Polysom Studio Installer**.

> **R3 / R5 Safety Policy:** In accordance with project safety guardrails, external publishing commands (`git push`, `gh release create`, `twine upload`, `vercel`, `netlify`, `wrangler`, `hf upload`) must only be executed when authorized credentials/tokens are explicitly configured by the release operator.

---

## 1. Local Pre-Release Build & Verification Checklist

Run the full verification and packaging pipeline from the repository root (`PowerShell` on Windows or `bash` on Linux/macOS):

```powershell
# 1. Build native C/AVX2/CUDA shared library (lucid_machines_core.dll / liblucid_machines_core.so)
.\studio\studio_env\Scripts\python.exe csrc/build_native.py

# 2. Run mandatory lint, bytecode compile, unit, invariant, security, and E2E test suites
.\studio\studio_env\Scripts\ruff.exe check .
.\studio\studio_env\Scripts\ruff.exe format --check .
.\studio\studio_env\Scripts\python.exe -m compileall -q lucid_machines scripts tests benchmarks
.\studio\studio_env\Scripts\pytest.exe -q

# 3. Confirm WORKSPACE_RULES.md remains untracked (must return empty output) and session_state step == 40555
git ls-files WORKSPACE_RULES.md

# 4. Build Python Wheel (.whl) and Source Distribution (.tar.gz) into dist/
.\studio\studio_env\Scripts\python.exe -m build --sdist --wheel

# 5. Build & package Native Windows Polysom Studio v1.0.0 structured app/ bundle, standalone setup, ZIP & SHA-256 checksums
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/publish_studio_windows.ps1
```

### Windows Bundle Layout & Artifact Checksums (`dist/SHA256SUMS.txt`)

`scripts/publish_studio_windows.ps1` uses a **structured `app/` layout** inside `dist/PolysomStudio-1.0.0-win-x64.zip` (`app\PolysomStudio.exe`, `app\PolysomStudioSetup.exe`, shared `.NET 10` WPF runtime, and `lucid_machines_core.dll`), eliminating runtime `%TEMP%\.net\` self-extraction and ~123.7 MB of duplicate `.NET` runtime payload compared to dual single-file EXEs. It also emits standalone `dist/PolysomStudioSetup.exe` (via Inno Setup `iscc` when `studio/windows/PolysomStudioInstaller.iss` is compiled, or single-file `.NET` fallback), per-artifact `<filename>.sha256` files, and a unified UTF-8 (no BOM, LF) `dist/SHA256SUMS.txt`.

Verify any downloaded release artifact against `SHA256SUMS.txt` or its standalone `.sha256` file:

```powershell
# Windows PowerShell
Get-FileHash dist\PolysomStudio-1.0.0-win-x64.zip -Algorithm SHA256
Get-FileHash dist\PolysomStudioSetup.exe -Algorithm SHA256
Get-Content dist\SHA256SUMS.txt
```

```bash
# Linux / macOS / WSL2
sha256sum -c SHA256SUMS.txt
```

---

## 2. Authenticode Code Signing & SmartScreen Trust

### Automated Signing Hooks in `scripts/publish_studio_windows.ps1`

`scripts/publish_studio_windows.ps1` signs Lucid Machines binaries (`PolysomStudio.exe`, `PolysomStudioSetup.exe`, `PolysomStudio.dll`, `PolysomStudioSetup.dll`, `lucid_machines_core.dll`) **before** creating the ZIP and SHA-256 digests whenever either signing backend is configured:

1. **Standard Authenticode (`signtool.exe`)**:
   - Set `SIGNTOOL_CERT_THUMBPRINT` (and optionally `SIGNTOOL_CERT_BASE64` + `SIGNTOOL_CERT_PASSWORD` for ephemeral CI PFX import, plus `SIGNTOOL_TIMESTAMP_URL`, defaulting to `http://timestamp.digicert.com`).
2. **Azure Trusted Signing**:
   - Set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `TRUSTED_SIGNING_ENDPOINT`, `TRUSTED_SIGNING_ACCOUNT`, and `TRUSTED_SIGNING_PROFILE` (plus `AZURE_TRUSTED_SIGNING_DLIB` if using `signtool` with `Azure.CodeSigning.Dlib.dll`).
3. **Unsigned Local Dev Fallback**:
   - When neither credential set is configured, `publish_studio_windows.ps1` logs a clear warning and produces unsigned local artifacts without failing developer builds.

### Windows Defender SmartScreen Guidance

- **Unsigned local/pre-release builds**: Windows Defender SmartScreen may display *"Windows protected your PC"* until the binary accumulates reputation or is signed with an EV/OV Authenticode certificate or Azure Trusted Signing profile. Always verify the downloaded file's SHA-256 digest against `SHA256SUMS.txt` (from GitHub Releases) before clicking **More info -> Run anyway**.
- **Native DLL isolation**: `PolysomStudio.Windows` uses `NativeLibrary.SetDllImportResolver` scoped strictly to `lucid_machines_core.dll` (never mutating process-wide `SetDllDirectory`).
- **Support & Security Contact**: For installer troubleshooting, checksum verification issues, or security/signing inquiries, contact **`contact@lucidmachines.co`** (website: **`https://lucidmachines.co`**; research paper correspondence goes to `founder@lucidmachines.co`).

---

## 3. Deploying the Windows Native Polysom Studio Installer (`PolysomStudioSetup.exe`)

The Windows desktop package (`dist/PolysomStudio-1.0.0-win-x64.zip`) bundles the structured `app/` `.NET 10` + `WebView2` host, all 6 Polysom Studio themes (`Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`), Lucid Eye branding (`polysom.ico`), the native C/CUDA engine (`lucid_machines_core.dll`), and the automated setup wizard (`PolysomStudioSetup.exe`, which auto-bootstraps Python 3.12 via `winget` if Python 3.10+ is absent and registers `kcx`, `polysom`, and `lucid-machines` on User `PATH`).

### Interactive GUI Installation
```powershell
studio\publish\PolysomStudioSetup.exe
```

### Headless / Enterprise Unattended Deployment
```powershell
# Headless binary & workspace resource smoke check (no PyTorch required; exit 0 on pass)
studio\publish\app\PolysomStudio.exe --smoke-test
studio\publish\PolysomStudioSetup.exe --smoke-test

# Non-mutating verification of Python/PyTorch/FastAPI/Studio assets (does not create shortcuts or touch PATH)
studio\publish\PolysomStudioSetup.exe --verify

# Unattended install/repair (registers HKCU Uninstall entry + Desktop/Start Menu shortcuts + kcx/polysom/lucid-machines CLIs)
studio\publish\PolysomStudioSetup.exe --yes --no-launch

# Unattended uninstall of shortcuts, HKCU registry entry, and User PATH entry (preserves checkpoints/data)
studio\publish\PolysomStudioSetup.exe --uninstall --yes

# Full unattended uninstall (also removes studio_env, WebView2 cache, and extracted app/ binaries; guards .git/checkpoints/data/exports)
studio\publish\PolysomStudioSetup.exe --uninstall --yes --full
```

---

## 4. Automated CI & Release Pipeline (`.github/workflows/`)

- **Continuous Integration (`.github/workflows/ci.yml`)**:
  - `test-python`: Ubuntu matrix (`Python 3.10–3.13`), native C engine (`csrc/build_native.py`), `ruff` syntax lint, `compileall`, full `pytest -q --ignore=tests/e2e`, mathematical invariants, DAG catalog (`>= 30` nodes), and wheel/sdist build + CLI smoke (`kcx --help`, `lucid-machines --help`, `polysom --help`).
  - `build-windows-studio`: `windows-latest` with `.NET 10.0.x`, Python 3.12 + CPU `torch`, `csrc/build_native.py`, `dotnet build PolysomStudio.sln -c Release`, `scripts/publish_studio_windows.ps1`, `--smoke-test`, and non-mutating `--verify`.
  - `build-website`: Ubuntu + `pnpm` v9 + Node 22 (`pnpm install --frozen-lockfile`, `tsc --noEmit`, `vite build`).
  - `build-papers`: Ubuntu + Tectonic (`papers/scripts/build_papers.py`).
- **Tag-Driven Draft Releases (`.github/workflows/release.yml`)**:
  - Triggered by pushing a `v*.*.*` tag (or `workflow_dispatch` with `dry_run: true`).
  - Verifies version parity across `pyproject.toml`, `lucid_machines/__init__.py`, both `.csproj` files, `website/package.json`, and the git tag.
  - Builds Python wheel/sdist, signed Windows bundle + `PolysomStudioSetup.exe`, portable Linux `PolysomStudio.AppDir` & macOS `Polysom Studio.app` tarballs (`kcx build-native --platform <linux|macos>`), and Tectonic paper PDFs, generates/verifies `SHA256SUMS.txt` + `.sha256` files, and publishes a **draft** GitHub Release (`draft: true`, `generate_release_notes: true`).

---

## 5. Publishing to PyPI (`twine upload dist/*`)

### Prerequisites
- Install Twine: `.\studio\studio_env\Scripts\pip.exe install twine`
- Configure a PyPI API token (`pypi-...`) via environment variable (`TWINE_USERNAME=__token__`, `TWINE_PASSWORD=<token>`) or `~/.pypirc`.

### Validation & Upload Commands
```powershell
# 1. Verify wheel and sdist metadata formatting (lucid_machines-1.0.0*)
.\studio\studio_env\Scripts\python.exe -m twine check dist/lucid_machines-1.0.0*.whl dist/lucid_machines-1.0.0*.tar.gz

# 2. (Optional) Dry-run upload to TestPyPI first
.\studio\studio_env\Scripts\python.exe -m twine upload --repository testpypi dist/lucid_machines-1.0.0*.whl dist/lucid_machines-1.0.0*.tar.gz

# 3. Publish official v1.0.0 package to PyPI
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "<YOUR_PYPI_API_TOKEN>"
.\studio\studio_env\Scripts\python.exe -m twine upload dist/lucid_machines-1.0.0*.whl dist/lucid_machines-1.0.0*.tar.gz
```

---

## 6. Publishing a GitHub Release (`v1.0.0`) & Organization Transfer

The canonical repository lives under the **Lucid Machines** GitHub organization at **`https://github.com/lucid-machines/triune`** (`https://github.com/lucid-machines`).

### Updating Local Clones After `Yasho4867/triune` -> `lucid-machines/triune` Transfer
If your local clone still points to `Yasho4867/triune` (and to keep Vercel Git deployments connected to the transferred repository without disruption):
```powershell
# Point local origin to the canonical lucid-machines organization repository
git remote set-url origin https://github.com/lucid-machines/triune.git
git remote -v
```
*(In GitHub **Settings -> Danger Zone -> Transfer ownership**, transferring `Yasho4867/triune` to `lucid-machines` automatically preserves GitHub redirects and Vercel repository linkage; re-link the Vercel project Git connection only if Vercel prompts after org transfer.)*

### Prerequisites
- Install GitHub CLI (`gh`) and authenticate via `gh auth login` (or set `GH_TOKEN`).

### Tagging & Manual Release Creation
```powershell
# 1. Push verified release branch and annotated v1.0.0 tag (triggers .github/workflows/release.yml draft release)
git tag -a v1.0.0 -m "Lucid Machines v1.0.0 Inaugural Release (Triune Architecture, Junkie & Polysom Studio)"
git push origin release/lucid-machines-v1.0.0
git push origin v1.0.0

# 2. Or manually create GitHub Release with binary artifacts and RELEASE_NOTES.md
gh release create v1.0.0 `
    dist/lucid_machines-1.0.0*.whl `
    dist/lucid_machines-1.0.0*.tar.gz `
    dist/PolysomStudio-1.0.0-win-x64.zip `
    dist/PolysomStudioSetup.exe `
    dist/*.sha256 `
    dist/SHA256SUMS.txt `
    --title "Lucid Machines v1.0.0 — Triune Architecture, CentroidSteer & Polysom Studio" `
    --notes-file RELEASE_NOTES.md
```

---

## 7. Deploying the Project Website (`website/`)

The live project website is hosted at **`https://lucidmachines.co`** (deployed from `website/` via Vercel). The source is a React/Vite application (`index.html`, `src/*`, `public/*`) built to `website/dist/`.

### Local Smoke Test
```powershell
Set-Location website
pnpm install --frozen-lockfile
pnpm run build
pnpm run preview -- --host 127.0.0.1 --port 4173
# Open http://127.0.0.1:4173/
```

### Option A: GitHub Pages
1. In your GitHub repository, navigate to **Settings -> Pages**.
2. Set **Source** to **GitHub Actions** (or deploy from branch `/website/dist` folder).
3. Use a workflow in `.github/workflows/pages.yml` configured with `actions/configure-pages@v5`, `actions/upload-pages-artifact@v3` (`path: "website/dist"`), and `actions/deploy-pages@v4`.

### Option B: Vercel
```bash
VERCEL_TOKEN="<YOUR_VERCEL_TOKEN>" npx vercel deploy website/dist --prod --yes
```

### Option C: Netlify
```bash
NETLIFY_AUTH_TOKEN="<YOUR_NETLIFY_TOKEN>" npx netlify-cli deploy --dir=website/dist --prod
```

### Option D: Cloudflare Pages
```bash
CLOUDFLARE_API_TOKEN="<YOUR_CF_TOKEN>" npx wrangler pages deploy website/dist --project-name=lucid-machines
```

---

## 8. Publishing Checkpoints & SafeTensors to Hugging Face Hub (`hf upload`)

### Prerequisites
- Authenticate with Hugging Face Hub: `.\studio\studio_env\Scripts\hf.exe auth login` or set `$env:HF_TOKEN = "hf_..."`.

### Export & Upload Commands
```powershell
# 1. Export checkpoint to Hugging Face SafeTensors format (exports/)
.\studio\studio_env\Scripts\python.exe -c "import lucid_machines; m = lucid_machines.load_model('triune-nano'); lucid_machines.export.hf_compat.export_to_huggingface(m, 'exports/triune-nano-hf')"

# 2. Upload exported model directory to Hugging Face Hub
$env:HF_TOKEN = "<YOUR_HF_WRITE_TOKEN>"
.\studio\studio_env\Scripts\hf.exe upload <your-hf-org>/triune-nano-v1.0 exports/triune-nano-hf --repo-type model
```
