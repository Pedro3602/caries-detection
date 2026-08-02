"""Train one experiment: YOLO26 size + augmentation recipe.

Planned CLI:
  python scripts/run_experiment.py \\
    --recipe experiments/classic/combined \\
    --model yolo26n \\
    --train-config configs/train_ablation.yaml

Applies recipe.yaml augmentations online; writes metrics under results/.
Not implemented in Phase 0.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Phase 0 stub — training comes later.")


if __name__ == "__main__":
    main()
