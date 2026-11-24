# Release Name Selection & Conflict Screening Report (`R1`)

## Locked 4-Layer Naming Boundary

**Company / Lab / Python Package / Repository Brand:** **Lucid Machines** (`lucid-machines`, `lucid_machines`).  
**Core Neural Architecture:** **Triune** (`TriuneTransformer`, `TriuneModel`, `TriuneConfig`, Hugging Face `model_type: "triune"`).  
**First Flagship Model:** **Junkie** (`models/configs/junkie.json`). Junkie is a model name, not a package namespace.  
**Primary CLI (`Polysom KCX`) & Desktop Workbench:** **`kcx` CLI** (`polysom` & `lucid-machines` compatibility aliases) and **Polysom Studio** (`PolysomStudio.exe`, `PolysomStudioSetup.exe`).

The inaugural release version is **1.0.0** and the release branch is `release/lucid-machines-v1.0.0`. Legacy `triune` Python imports and `TRIUNE_*` environment variables remain compatibility shims only; they do not replace the Lucid Machines release identity.

## Conflict Screening

The prior draft codename has been rejected and fully removed as a naming authority. It must not appear in release documents, package metadata, launch copy, website copy, or desktop branding.

Automated public checks were performed for **PyPI**, **npm**, **GitHub**, DNS, and general web search. The screening notes the dormant `lucidmachines.github.io` home-lighting page (2023) and nearby Lucid-branded products. The final names preserve a separate academic architecture name (`Triune`) and a separate flagship model name (`Junkie`) to avoid presenting the model or architecture as the company brand.

### Locked Release Surfaces

| Surface | Locked value |
| --- | --- |
| Python distribution | `lucid-machines` |
| Python import | `lucid_machines` |
| Primary CLI (`Polysom KCX`) | `kcx` (`polysom` / `lucid-machines` aliases) |
| Live Website | `https://lucidmachines.co` |
| Canonical GitHub Repository | `https://github.com/lucid-machines/triune` (`https://github.com/lucid-machines`) |
| Desktop binary | `PolysomStudio.exe` |
| Desktop installer | `PolysomStudioSetup.exe` |
| Architecture | `TriuneTransformer`, `TriuneModel`, `TriuneConfig` |
| Model | `Junkie` |

## Compatibility & Asset Preservation

The naming boundary preserves architecture assets and user data without restoring legacy package layouts:

- `lucid_machines` directly exports all Triune architecture classes, architecture presets, tokenizer discovery, and `model_type: "triune"` Hugging Face exports.
- `LUCID_MACHINES_*` configuration takes precedence over legacy `TRIUNE_*` fallback variables.
- Native loader lookup prefers `lucid_machines_core.*`.
- Existing `triune_tokenizer.json`, checkpoints, datasets, exports, and papers remain unchanged.
- OS credential keyring lookup stores under `"Polysom Studio"` (retaining legacy keyring fallback read support).

## Rejected Draft Purge

No release artifact may use the rejected draft codename or its Studio variant. The repository tests explicitly scan release documentation, model configuration names, website HTML, Studio UI sources, release notes, and deployment guides for the purge condition.

## Non-Legal / Trademark Disclaimer

Automated registry, DNS, GitHub, and web checks are preliminary engineering collision screens only. They do not constitute legal advice, a formal trademark opinion, or comprehensive clearance. Qualified intellectual-property counsel should review national and international trademark registers and common-law usage before filing, registering, or commercially launching this brand.
