"""Train one experiment: YOLO26 size + augmentation recipe.

Example:
  python scripts/run_experiment.py \\
    --recipe experiments/classic/combined \\
    --model yolo26n \\
    --train-config configs/train_ablation.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO_ROOT / "experiments" / "summary.csv"
SUMMARY_FIELDS = [
    "family",
    "recipe",
    "model",
    "mAP50",
    "mAP50-95",
    "precision",
    "recall",
    "epochs",
    "results_dir",
    "confusion_matrix",
    "confusion_matrix_normalized",
    "timestamp",
]


def _die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _normalize_yolo26_weights(model_arg: str) -> tuple[str, str]:
    """Return (weights_filename, model_stem). Reject non-YOLO26 models."""
    name = Path(model_arg).name
    stem = Path(name).stem
    if not stem.startswith("yolo26"):
        _die(f"only YOLO26 models are allowed in v1, got '{model_arg}'")
    if not name.endswith(".pt"):
        name = f"{stem}.pt"
        stem = Path(name).stem
    return name, stem


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        _die("PyYAML is not installed; run: pip install -r requirements.txt")
    if not path.is_file():
        _die(f"missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        _die(f"expected a mapping in {path}")
    return data


def _infer_family_recipe(recipe_dir: Path) -> tuple[str, str]:
    """Infer family/recipe from .../experiments/<family>/<recipe>."""
    try:
        recipe = recipe_dir.name
        family = recipe_dir.parent.name
        experiments = recipe_dir.parent.parent.name
    except Exception:  # noqa: BLE001
        _die(f"could not infer family/recipe from path: {recipe_dir}")
    if experiments != "experiments":
        _die(
            "recipe path must look like experiments/<family>/<recipe>, "
            f"got: {recipe_dir.relative_to(REPO_ROOT) if recipe_dir.is_relative_to(REPO_ROOT) else recipe_dir}"
        )
    return family, recipe


def _merge_train_kwargs(
    train_config: dict[str, Any],
    recipe_config: dict[str, Any],
    device: str | None,
) -> dict[str, Any]:
    """Merge train knobs + recipe augs. Drop model (weights loaded separately)."""
    merged = dict(train_config)
    merged.pop("model", None)
    merged.update(recipe_config)
    merged.pop("model", None)
    merged["plots"] = True
    if device is not None:
        merged["device"] = device
    return merged


def _rel_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_detection_metrics(metrics: Any) -> dict[str, float | None]:
    """Read mAP / P / R from an Ultralytics validator metrics object."""
    out: dict[str, float | None] = {
        "mAP50": None,
        "mAP50-95": None,
        "precision": None,
        "recall": None,
    }
    if metrics is None:
        return out

    box = getattr(metrics, "box", None)
    if box is not None:
        out["mAP50"] = _as_float(getattr(box, "map50", None))
        out["mAP50-95"] = _as_float(getattr(box, "map", None))
        out["precision"] = _as_float(getattr(box, "mp", None))
        out["recall"] = _as_float(getattr(box, "mr", None))
        if all(v is not None for v in out.values()):
            return out

    results_dict = getattr(metrics, "results_dict", None)
    if not isinstance(results_dict, dict) and isinstance(metrics, dict):
        results_dict = metrics
    if isinstance(results_dict, dict):
        key_map = {
            "mAP50": ("map50",),
            "mAP50-95": ("map50-95", "map50_95"),
            "precision": ("precision",),
            "recall": ("recall",),
        }
        for field, needles in key_map.items():
            if out[field] is not None:
                continue
            for key, value in results_dict.items():
                key_l = str(key).lower().replace(" ", "")
                if any(n in key_l for n in needles) and (
                    field != "mAP50" or "map50-95" not in key_l and "map50_95" not in key_l
                ):
                    out[field] = _as_float(value)
                    break
    return out


def _find_plot(results_dir: Path, filename: str) -> Path | None:
    direct = results_dir / filename
    if direct.is_file():
        return direct
    matches = list(results_dir.rglob(filename))
    return matches[0] if matches else None


def _upsert_summary(row: dict[str, Any]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if SUMMARY_CSV.is_file():
        with SUMMARY_CSV.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                if (
                    existing.get("family") == row["family"]
                    and existing.get("recipe") == row["recipe"]
                    and existing.get("model") == row["model"]
                ):
                    continue
                rows.append(existing)
    rows.append({k: "" if row.get(k) is None else str(row.get(k)) for k in SUMMARY_FIELDS})
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one YOLO26 model with one augmentation recipe."
    )
    parser.add_argument(
        "--recipe",
        required=True,
        help="Path to recipe dir (must contain recipe.yaml and dataset/data.yaml)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="YOLO26 size or weights, e.g. yolo26n or yolo26n.pt",
    )
    parser.add_argument(
        "--train-config",
        default="configs/train_ablation.yaml",
        help="Train knobs YAML (default: configs/train_ablation.yaml)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional Ultralytics device override (e.g. 0, cpu, mps)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print merged train kwargs and exit without training",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    recipe_dir = _resolve_path(args.recipe)
    train_config_path = _resolve_path(args.train_config)
    weights, model_stem = _normalize_yolo26_weights(args.model)

    if not recipe_dir.is_dir():
        _die(f"recipe directory not found: {recipe_dir}")

    recipe_yaml = recipe_dir / "recipe.yaml"
    data_yaml = recipe_dir / "dataset" / "data.yaml"
    if not recipe_yaml.is_file():
        _die(f"missing recipe.yaml (create the experiment first): {recipe_yaml}")
    if not data_yaml.is_file():
        _die(
            f"missing dataset/data.yaml (run prepare_dataset + create_experiment first): {data_yaml}"
        )

    family, recipe_name = _infer_family_recipe(recipe_dir)
    train_config = _load_yaml(train_config_path)
    recipe_config = _load_yaml(recipe_yaml)
    train_kwargs = _merge_train_kwargs(train_config, recipe_config, args.device)

    results_dir = recipe_dir / "results" / model_stem
    train_call = {
        "data": str(data_yaml),
        "project": str(recipe_dir / "results"),
        "name": model_stem,
        "exist_ok": True,
        **train_kwargs,
    }

    if args.dry_run:
        print("dry-run: would train with")
        print(f"  weights: {weights}")
        print(f"  family/recipe: {family}/{recipe_name}")
        print(f"  results: {_rel_to_repo(results_dir)}")
        for key in sorted(train_call):
            print(f"  {key}: {train_call[key]}")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        _die("ultralytics is not installed; run: pip install -r requirements.txt")

    print(f"training {weights} on {family}/{recipe_name}")
    model = YOLO(weights)
    model.train(**train_call)

    best_weights = results_dir / "weights" / "best.pt"
    val_model = YOLO(str(best_weights)) if best_weights.is_file() else model
    metrics = val_model.val(
        data=str(data_yaml),
        project=str(recipe_dir / "results"),
        name=model_stem,
        exist_ok=True,
        plots=True,
        **({"device": args.device} if args.device is not None else {}),
    )

    det = _extract_detection_metrics(metrics)
    map50 = det["mAP50"]
    map5095 = det["mAP50-95"]
    precision = det["precision"]
    recall = det["recall"]
    epochs = train_kwargs.get("epochs", "")

    cm = _find_plot(results_dir, "confusion_matrix.png")
    cm_norm = _find_plot(results_dir, "confusion_matrix_normalized.png")

    row = {
        "family": family,
        "recipe": recipe_name,
        "model": model_stem,
        "mAP50": map50,
        "mAP50-95": map5095,
        "precision": precision,
        "recall": recall,
        "epochs": epochs,
        "results_dir": _rel_to_repo(results_dir),
        "confusion_matrix": _rel_to_repo(cm) if cm else "",
        "confusion_matrix_normalized": _rel_to_repo(cm_norm) if cm_norm else "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    _upsert_summary(row)

    print(f"results: {_rel_to_repo(results_dir)}")
    print(
        f"metrics: mAP50={map50} mAP50-95={map5095} "
        f"precision={precision} recall={recall}"
    )
    if cm:
        print(f"confusion matrix: {_rel_to_repo(cm)}")
    else:
        print("warning: confusion_matrix.png not found under results", file=sys.stderr)
    print(f"summary: {_rel_to_repo(SUMMARY_CSV)}")


if __name__ == "__main__":
    main()
