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

## Scripts (planned)

| Script | Role |
|--------|------|
| `prepare_dataset.py` | COCO disease JSON → shared `datasets/caries` |
| `create_experiment.py` | Scaffold `experiments/<family>/<recipe>/` |
| `run_experiment.py` | Train one job: YOLO26 size + recipe |
| `generate_previews.py` | Sample augmented preview tiles |
| `run_ablation.py` | Loop all recipes with ablation config |
| `plot_ablation.py` | Plot `experiments/summary.csv` |
| `eda.py` | Dataset statistics |

## Status

**Phase 0 (current):** repository organization only — no dataset conversion, no augmentation generation, no training runs yet.

## Setup (later)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
