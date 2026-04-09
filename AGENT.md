# AGENT.md

## Purpose
This repository is an early-stage sign language data preparation and landmark extraction project built around a small demo subset of the MS-ASL dataset. The current repo appears to focus on:

1. filtering metadata from `MSASL_train.json`
2. downloading trimmed raw videos for a small target vocabulary
3. extracting MediaPipe Holistic landmarks from those videos
4. saving fixed-length `.npy` landmark arrays for later training

Where the project intent is not yet explicit, placeholders are included so the repo owner can fill them in later.

---

## Repository structure

```text
DATN_SignLanguageDetection/
├── .idea/                               # IDE/project-local settings (do not rely on this in code)
├── data/
│   ├── raw_videos/                      # downloaded sign-language clips, grouped by action/word
│   ├── MSASL_train_valid_only.csv       # filtered metadata for successfully usable videos
│   └── extracted_landmarks/             # expected output folder from notebooks/extract_data.ipynb
├── notebooks/
│   ├── data_prep.ipynb                  # filter MS-ASL metadata and download video segments
│   └── extract_data.ipynb               # extract pose/hand landmarks and save .npy arrays
├── .gitignore
├── .python-version
├── MSASL_train.json                     # source metadata used by the data prep notebook
├── README.md
├── pyproject.toml                       # Python project metadata and dependencies
├── test.py                              # quick local inspection script for downloaded videos
└── uv.lock                              # locked dependency resolution for uv
```

---

## What this agent should understand about the repo

### Current workflow
The agent should treat this repository as a **data pipeline repo first**, not yet as a full training/inference application.

Observed pipeline:

1. `notebooks/data_prep.ipynb`
   - reads `MSASL_train.json`
   - filters to a small vocabulary subset
   - downloads trimmed video segments with `yt-dlp`
   - writes valid rows to `data/MSASL_train_valid_only.csv`
   - stores videos under `data/raw_videos/<word>/`

2. `notebooks/extract_data.ipynb`
   - reads videos from `data/raw_videos/`
   - runs MediaPipe Holistic
   - extracts pose + left hand + right hand keypoints
   - normalizes each sample to `TARGET_FRAMES = 30`
   - saves outputs as `.npy` arrays under `data/extracted_landmarks/<word>/`

3. `test.py`
   - is a lightweight local script for printing the contents of `data/raw_videos/`
   - should be treated as an ad hoc utility, not a stable production entrypoint

---

## Agent goals
When working in this repo, the agent should prioritize:

1. **Preserving the data pipeline**
   - keep the metadata → download → landmark extraction flow understandable
   - avoid breaking relative paths between root files, `data/`, and `notebooks/`

2. **Improving reproducibility**
   - prefer configurable paths over hardcoded machine-specific paths
   - prefer scripts/modules over notebook-only logic when moving toward maintainability
   - keep dependency updates compatible with Python `>=3.11`

3. **Keeping outputs deterministic where practical**
   - do not silently change frame count, feature dimensionality, output naming, or directory layout
   - if such a change is necessary, document it clearly in README / comments / migration notes

4. **Protecting large or generated artifacts**
   - avoid committing generated videos, extracted arrays, caches, or notebook outputs unless explicitly requested
   - keep generated assets under `data/` or another declared artifacts directory

5. **Supporting incremental cleanup**
   - prefer small refactors
   - preserve notebook behavior unless explicitly asked to convert notebooks into scripts/modules

---

## Agent rules: do

### Do preserve these conventions
- Keep `MSASL_train.json` as the source metadata file unless explicitly changed.
- Keep downloaded raw videos grouped by label/word.
- Keep extracted landmark outputs grouped by label/word.
- Keep the fixed-length sample contract unless intentionally versioning it.
- Prefer relative paths rooted from the repo over absolute local paths.

### Do make safe improvements
- Replace hardcoded paths with config constants, environment variables, or CLI arguments.
- Add docstrings, comments, and README notes when behavior is non-obvious.
- Add validation checks before filesystem operations.
- Add idempotency where possible, especially for re-runs of download/extraction pipelines.
- Add `.gitignore` coverage for generated artifacts if missing.
- Split notebook logic into reusable utility modules **only if requested** or if a change clearly improves maintainability without changing outputs.

### Do be careful with data semantics
- Treat label names, folder names, frame count, and keypoint dimensionality as part of the dataset contract.
- Assume downstream training code may depend on the current `.npy` shapes and folder structure.
- Flag any change that alters:
  - target words/actions
  - number of frames per sample
  - which landmarks are extracted
  - file naming conventions
  - output directory layout

### Do leave placeholders instead of guessing
If the repo lacks context for an important policy, keep a placeholder like:
- `[OWNER TODO: define training entrypoint]`
- `[OWNER TODO: define accepted model architectures]`
- `[OWNER TODO: define evaluation metric and target baseline]`
- `[OWNER TODO: define dataset versioning policy]`

---

## Agent rules: do not

### Do not rewrite the repo around assumptions
- Do not assume there is already a training pipeline, inference API, or deployment target.
- Do not invent new architecture folders (`src/`, `app/`, `models/`, etc.) unless explicitly requested.
- Do not rename files or folders just for style.

### Do not break the dataset pipeline
- Do not change path layout in a way that breaks notebook execution.
- Do not change output tensor shape or frame padding/truncation rules silently.
- Do not switch landmark extraction libraries without explicit approval.
- Do not replace MediaPipe Holistic with a different backend unless explicitly requested.

### Do not commit or encourage unsafe artifact handling
- Do not commit large downloaded videos or generated landmark arrays unless the owner explicitly asks for tracked sample data.
- Do not assume YouTube links remain valid; build workflows that tolerate deleted/private/unavailable videos.
- Do not embed secrets, browser cookies, tokens, or local machine paths in committed code.

### Do not over-promote `test.py`
- Do not treat `test.py` as production code.
- Do not add major new pipeline logic into `test.py`.
- If the repo needs a real CLI/script entrypoint, create a dedicated script instead of expanding `test.py` into a catch-all.

---

## File-by-file guidance

### `pyproject.toml`
Use this as the source of truth for Python version and dependencies.

Agent should:
- keep dependency changes minimal and justified
- preserve compatibility with the current notebook workflow
- prefer adding only dependencies that are directly needed

Agent should not:
- add heavy libraries speculatively
- perform broad dependency upgrades without reason

### `notebooks/data_prep.ipynb`
This notebook currently acts as a dataset filtering + video download stage.

Agent should:
- preserve its ability to filter target words
- preserve successful/failed download handling
- preserve export of cleaned metadata CSV
- improve resilience to re-runs and invalid links where possible

Agent should not:
- remove the cleaned CSV output unless a replacement contract is introduced
- silently change naming conventions for downloaded clips

### `notebooks/extract_data.ipynb`
This notebook currently acts as the landmark extraction stage.

Agent should:
- preserve extraction of pose + left hand + right hand keypoints
- preserve fixed-length output behavior
- preserve per-action output grouping
- be careful about GUI display requirements from OpenCV

Agent should not:
- silently alter keypoint ordering or dimensionality
- silently change `TARGET_FRAMES`
- assume headless environments unless explicitly adapting for them

### `test.py`
This is a local inspection helper.

Agent should:
- keep it simple if touched at all
- prefer converting hardcoded paths into relative paths if asked

Agent should not:
- turn this into the main pipeline entrypoint

### `data/`
This directory contains source, filtered, and generated data artifacts.

Agent should:
- treat files here as data assets, not source code
- avoid destructive edits
- document when outputs are regenerated

Agent should not:
- delete or rewrite data files without explicit instruction
- move large data files around unnecessarily

---

## Expected coding style for future changes

- Prefer small, explicit functions.
- Prefer clear filesystem contracts over implicit conventions.
- Prefer reproducible scripts over one-off manual notebook edits when logic stabilizes.
- Prefer defensive checks around missing files, empty folders, invalid videos, and failed extraction.
- Keep comments practical and project-specific.

[OWNER TODO: define formatting/linting tools, e.g. ruff/black/isort]
[OWNER TODO: define test strategy]
[OWNER TODO: define whether notebooks should remain first-class or be migrated into `scripts/`]

---

## Suggested future structure (only when explicitly requested)

If the owner later asks for cleanup, a safe direction would be:

```text
scripts/
├── download_dataset.py
├── extract_landmarks.py
└── inspect_raw_videos.py

src/
└── datn_sign_language_detection/
    ├── config.py
    ├── io_utils.py
    ├── download.py
    └── landmarks.py
```

This is only a suggestion. Do **not** introduce this structure unless requested.

---

## Known assumptions and placeholders

- Primary dataset source appears to be **MS-ASL** metadata.  
  `[OWNER TODO: confirm exact dataset scope and citation requirements]`

- Current target vocabulary appears to be a small demo subset: `hello`, `thank you`, `please`, `help`, `sorry`.  
  `[OWNER TODO: confirm whether this is permanent, demo-only, or phase 1]`

- Landmark output currently appears to be pose + left hand + right hand for 30 frames.  
  `[OWNER TODO: confirm final sample tensor shape contract for downstream training]`

- No training/evaluation/inference entrypoint is visible yet.  
  `[OWNER TODO: define model training workflow]`
  `[OWNER TODO: define evaluation workflow]`
  `[OWNER TODO: define inference/demo workflow]`

- No contribution workflow is visible yet.  
  `[OWNER TODO: define branch strategy, PR expectations, and review checklist]`

---

## Preferred agent behavior when context is missing

When requirements are unclear, the agent should:

1. preserve existing behavior
2. avoid destructive refactors
3. add clear placeholders instead of inventing policy
4. make the smallest change that improves clarity or maintainability
5. document assumptions in comments or markdown

---

## Quick decision checklist for the agent

Before making a change, ask:

- Does this preserve the current data pipeline?
- Does this avoid changing output structure or tensor shape silently?
- Does this avoid hardcoding machine-specific paths?
- Does this avoid committing generated artifacts?
- Am I adding a placeholder where project policy is still unknown?

If any answer is “no”, stop and reduce the scope of the change.