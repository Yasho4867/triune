# Deployment & Release Publishing Guide — Lucid Machines `v1.0.0`

This guide provides step-by-step, reproducible instructions for building, verifying, and deploying all **Lucid Machines v1.0.0** (the **Triune** architecture **`TriuneTransformer`**, inaugural flagship model configurations **`Junkie` / `Junkie-7B` / `Junkie-3B` / `Junkie-1B` / `Junkie-Mini`**, & **Polysom Studio v1.0.0**) release artifacts across **GitHub Releases**, **GitHub Pages**, **PyPI**, **Vercel / Netlify / Cloudflare Pages**, **Hugging Face Hub**, **Google Colab Pro**, and the **Windows Native Polysom Studio Installer**.

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

# 5. Build & package Native Windows Polysom Studio v1.0.0 App, Installer, and portable ZIP + SHA256
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/publish_studio_windows.ps1
```

### Generating `dist/SHA256SUMS.txt`

```powershell
Get-ChildItem dist -File -Exclude SHA256SUMS.txt,*.sha256 | ForEach-Object {
    $h = (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower()
    "$h  $($_.Name)"
} | Set-Content -Path dist/SHA256SUMS.txt -Encoding UTF8
Get-Content dist/SHA256SUMS.txt
```

---

## 2. Deploying the Windows Native Polysom Studio Installer (`PolysomStudioSetup.exe`)

The Windows desktop package (`dist/PolysomStudio-1.0.0-win-x64.zip`) is a self-contained distribution bundling the `.NET 10` runtime, `WebView2` bridge, all 6 Polysom Studio themes (`Hypnagogic`, `Phosphene`, `Pareidolia`, `Afterimage`, `Mirage`, `Euphoria`), Lucid Eye branding (`polysom.ico`), and the native C/CUDA engine (`lucid_machines_core.dll`).

### Interactive GUI Installation
```powershell
studio\publish\PolysomStudioSetup.exe
```

### Headless / Enterprise Unattended Deployment
```powershell
# Verify existing environment health (returns exit code 0 on pass)
studio\publish\PolysomStudioSetup.exe --verify

# Unattended install/repair (registers HKCU Uninstall entry + Desktop/Start Menu shortcuts)
studio\publish\PolysomStudioSetup.exe --yes --no-launch

# Unattended clean uninstall of shortcuts and HKCU registry entry (preserves checkpoints/data)
studio\publish\PolysomStudioSetup.exe --uninstall --yes
```

---

## 3. Publishing to PyPI (`twine upload dist/*`)

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

## 4. Publishing a GitHub Release (`v1.0.0`) & Organization Transfer

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

### Tagging & Release Creation
```powershell
# 1. Push verified release branch and annotated v1.0.0 tag
git tag -a v1.0.0 -m "Lucid Machines v1.0.0 Inaugural Release (Triune Architecture, Junkie & Polysom Studio)"
git push origin release/lucid-machines-v1.0.0
git push origin v1.0.0

# 2. Create GitHub Release with binary artifacts and RELEASE_NOTES.md
gh release create v1.0.0 `
    dist/lucid_machines-1.0.0*.whl `
    dist/lucid_machines-1.0.0*.tar.gz `
    dist/PolysomStudio-1.0.0-win-x64.zip `
    dist/*.sha256 `
    dist/SHA256SUMS.txt `
    --title "Lucid Machines v1.0.0 — Triune Architecture, CentroidSteer & Polysom Studio" `
    --notes-file RELEASE_NOTES.md
```

---

## 5. Deploying the Project Website (`website/`)

The live project website is hosted at **`https://lucidmachines.co`** (deployed from `website/` via Vercel). The source is a React/Vite application (`index.html`, `src/*`, `public/*`) built to `website/dist/`.

### Local Smoke Test
```powershell
Set-Location website
npm.cmd run build
npm.cmd run preview -- --host 127.0.0.1 --port 4173
# Open http://127.0.0.1:4173/
```

### Option A: GitHub Pages
1. In your GitHub repository, navigate to **Settings -> Pages**.
2. Set **Source** to **GitHub Actions** (or deploy from branch `/website/dist` folder).
3. Use the following workflow in `.github/workflows/pages.yml`:

```yaml
name: Deploy Lucid Machines Website to GitHub Pages
on:
  push:
    branches: ["main", "release/lucid-machines-v1.0.0"]
    paths: ["website/**"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v5
      - uses: actions/upload-pages-artifact@v3
        with:
          path: "website/dist"
      - id: deployment
        uses: actions/deploy-pages@v4
```

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

## 6. Publishing Checkpoints & SafeTensors to Hugging Face Hub (`hf upload`)

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
