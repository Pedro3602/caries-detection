# caries-detection

Deep learning course project: fine-tune **YOLO26** for **binary caries detection** on panoramic dental X-rays, and compare augmentation recipes across three families.

## Goal

- Detect caries lesions (merge *Caries* + *Deep Caries* into one class).
- Compare augmentation families on a small labeled set (~705 disease-annotated images).
- Ablate with a light model/budget, then retrain the winner with a stronger setup.

## Augmentation families

| Family | Recipes | Idea |
|--------|---------|------|
| `none` | `baseline` | No meaningful augmentation |
| `classic` | `photometric`, `geometric`, `combined` | Light/HSV and/or rotation, translate, scale, flip |
| `crops` | `mosaic`, `mosaic_mixup` | Composition / crop-style augs |

Recipes are **not** checked in. Create them with `scripts/create_experiment.py` when needed. Each recipe folder will contain:

- `README.md` — what the recipe does + analysis after training
- `recipe.yaml` — augmentation hyperparameters
- `dataset/` — pointer to the shared split + (later) previews
- `results/` — training metrics and weights (later)

**Mode (Option C):** recipes store config + previews + results; augmentation is applied online at train time (no full offline expanded dataset in v1).

## Train budgets

| Stage | Model | Epochs | Patience | Image size |
|-------|-------|--------|----------|------------|
| Ablation | `yolo26n` | 50 | 15 | 640 |
| Final | `yolo26s` | 80 | 20 | 640 |

## Layout

```
configs/           # shared train knobs (ablation vs final)
datasets/          # shared YOLO split (generated later)
experiments/       # family folders; recipes created by create_experiment.py
scripts/           # prepare / create / run helpers (stubs for now)
data/              # raw DENTEX archive (local, gitignored)
```

## Scripts

| Script | Role | Status |
|--------|------|--------|
| `prepare_dataset.py` | COCO disease JSON → shared `datasets/caries` | **ready** |
| `create_experiment.py` | Scaffold `experiments/<family>/<recipe>/` | **ready** |
| `run_experiment.py` | Train one job: YOLO26 size + recipe | **ready** |
| `generate_previews.py` | Sample augmented preview tiles | stub |
| `run_ablation.py` | Loop all recipes with ablation config | stub |
| `plot_ablation.py` | Plot `experiments/summary.csv` | stub |
| `eda.py` | Dataset statistics | stub |

## Prepare dataset

Converts DENTEX disease COCO annotations to a binary caries YOLO split (`Caries` + `Deep Caries` → class `0`), stratified **70/15/15**, seed `42`:

```bash
python scripts/prepare_dataset.py
# or: python scripts/prepare_dataset.py --force   # overwrite datasets/caries
```

Writes `datasets/caries/{images,labels}/{train,val,test}` and `datasets/caries/data.yaml`.

## Create experiment

Scaffolds `experiments/<family>/<recipe>/` with `README.md`, `recipe.yaml`, and `dataset/data.yaml` pointing at the shared split:

```bash
python scripts/create_experiment.py --family none --recipe baseline
python scripts/create_experiment.py --all
python scripts/create_experiment.py --family classic --recipe combined --force
```

Known recipes: `none/baseline`, `classic/{photometric,geometric,combined}`, `crops/{mosaic,mosaic_mixup}`.

## Run one experiment

Prerequisites:

1. Shared YOLO split: `python scripts/prepare_dataset.py`
2. Recipe folder: `python scripts/create_experiment.py --family none --recipe baseline`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/run_experiment.py \
  --recipe experiments/classic/combined \
  --model yolo26n \
  --train-config configs/train_ablation.yaml
```

Useful flags: `--dry-run` (print merged kwargs only), `--device 0|cpu|mps`.

Outputs land in `experiments/<family>/<recipe>/results/<model>/` (weights, curves, confusion matrices). Metrics are upserted into `experiments/summary.csv`.

### Metrics

| Metric | Role |
|--------|------|
| mAP50-95 | Primary ranking key |
| mAP50 | Tie-break + report |
| Precision / Recall | Report |
| Confusion matrix | `confusion_matrix.png` + normalized variant (caries vs background) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
