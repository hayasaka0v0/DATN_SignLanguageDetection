# DATN_SignLanguageDetection

This repository currently focuses on a small sign-language data pipeline and a first baseline classifier built from extracted landmarks.

## Baseline Training Workflow

The extracted landmark arrays under `data/extracted_landmarks/<word>/` remain the source dataset.
Instead of moving `.npy` files into train/test folders, this repo uses a split manifest so the pipeline stays reproducible.

### 1. Create a reproducible split manifest

```bash
uv run python prepare_splits.py
```

This writes `data/splits/landmark_split.csv` with:

- `path`
- `label`
- `label_id`
- `split`

Default split ratios are `train=0.65`, `val=0.15`, `test=0.20`.

### 2. Train the first PyTorch baseline

```bash
uv run python train_baseline.py --manifest data/splits/landmark_split.csv
```

This baseline:

- loads each `(30, 258)` landmark sample
- flattens it into `7740` input features
- trains a small MLP classifier for the 5 demo words

Outputs are written to `artifacts/baseline_mlp/`:

- `best_model.pt`
- `metrics.json`

## Notes

- `data/extracted_landmarks/` is not modified by the split or training steps.
- generated split files and model artifacts are ignored by git
- this baseline is intentionally simple so you can validate the landmark pipeline before trying GRU/LSTM models
