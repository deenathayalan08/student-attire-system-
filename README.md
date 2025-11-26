
# Student Attire — Quick Start

This repository contains a small image classification/verification pipeline and a Streamlit demo for detecting student attire categories. The README below is a concise, beginner-friendly guide to get you up and running.

## What this repo contains

- A Streamlit demo: `app/streamlit_app.py`
- Core library code under `src/` (dataset handling, model, verification, utilities)
- Datasets are in `datasets/` (some archived into `archive/` if removed)
- Helpful scripts (evaluators, data mapping, cleanup) in the repo root or `scripts/`

## Quick setup (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

	python -m venv .venv; .\.venv\Scripts\Activate.ps1

2. Install dependencies:

	pip install -r requirements.txt

3. Verify the environment (optional):

	python check_setup.py

4. Run the Streamlit demo locally:

	streamlit run app\streamlit_app.py

## Running the evaluator (non-invasive)

If you want to evaluate a dataset without changing the main code, use the standalone evaluator created during the project exploration.

1. Generate mappings for a dataset (if not already present):

	python scripts\make_mappings.py --dataset "datasets\uniform 1" --out tmp_mappings\uniform1.csv

2. Run the evaluator:

	python evaluate_dataset.py --mapping tmp_mappings\uniform1.csv --out results\uniform1_accuracy.json

Output will include cross-validated accuracy and a small report in `results/`.

## Where to look next

- `src/model.py` — model training and prediction utilities
- `src/dataset.py` — dataset parsing, mapping helpers
- `scripts/` — mapping and cleanup utilities
- `results/` — evaluation, downsample, and cleanup reports
- `archive/` — ZIP archives of removed datasets (kept for safety)

## Notes and caveats

- Large dataset folders were archived and removed when you requested smaller project size; archives are in `archive/`.
- Some datasets are single-class or heavily imbalanced; cross-validation accuracy may be unreliable for them.
- Downsampling produced minimal space savings because images were already compressed.

If you want any further cleanup (remove archives, results, or helper scripts) or want the evaluator integrated into the main repo instead of being standalone, tell me which files to modify and I’ll do it.

---
Short, friendly guide created per your request — only this README remains in the repo root as the single guide file.

## Backups and removed guides

I archived the previous top-level guide files to `archive/removed_guides/` so the repository root stays clean but nothing was permanently lost.

To restore a specific guide back to the project root (PowerShell):

```powershell
Copy-Item -Path .\archive\removed_guides\BEGINNER_GUIDE.md -Destination .\ -Force
```

Or to restore them all:

```powershell
Get-ChildItem -Path .\archive\removed_guides -Filter '*.md' | ForEach-Object { Copy-Item -Path $_.FullName -Destination .\ -Force }
```

If you prefer I permanently remove those backups to free space, tell me and I will delete `archive/removed_guides/` after your confirmation.

