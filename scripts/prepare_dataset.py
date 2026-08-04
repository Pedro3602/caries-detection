"""Convert DENTEX disease COCO JSON into a shared YOLO caries split.

Example:
  python scripts/prepare_dataset.py

Writes datasets/caries/ with stratified train/val/test (70/15/15).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COCO = (
    REPO_ROOT
    / "data"
    / "archive"
    / "training_data"
    / "training_data"
    / "quadrant-enumeration-disease"
    / "train_quadrant_enumeration_disease.json"
)
DEFAULT_OUT = REPO_ROOT / "datasets" / "caries"
CARIES_CATEGORY_IDS = {1, 3}  # Caries, Deep Caries → YOLO class 0
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test


def _die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _link_or_copy(src: Path, dst: Path) -> str:
    """Prefer hardlink, then symlink, then copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        pass
    shutil.copy2(src, dst)
    return "copy"


def _coco_bbox_to_yolo(
    bbox: list[float], img_w: int, img_h: int
) -> tuple[float, float, float, float] | None:
    """COCO xywh (absolute) → YOLO xc yc w h (normalized)."""
    x, y, w, h = bbox
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    # clamp lightly for floating-point edge cases
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    if nw <= 0 or nh <= 0:
        return None
    return xc, yc, nw, nh


def _stratified_split(
    items: list[tuple[int, bool]],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[int]]:
    """Split image ids by has_caries flag. Returns train/val/test id lists."""
    rng = random.Random(seed)
    by_flag: dict[bool, list[int]] = {True: [], False: []}
    for image_id, has_caries in items:
        by_flag[has_caries].append(image_id)

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for flag in (True, False):
        ids = by_flag[flag][:]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        # remainder → test so totals match
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        # ensure tiny strata still get a test slot when possible
        if n >= 3 and n_test == 0 and n_val > 0:
            n_val -= 1
            n_test += 1
        if n >= 3 and n_val == 0 and n_train > 1:
            n_train -= 1
            n_val += 1
        splits["train"].extend(ids[:n_train])
        splits["val"].extend(ids[n_train : n_train + n_val])
        splits["test"].extend(ids[n_train + n_val : n_train + n_val + n_test])

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def _write_data_yaml(out_dir: Path) -> None:
    content = (
        f"path: {out_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: caries\n"
    )
    (out_dir / "data.yaml").write_text(content, encoding="utf-8")


def prepare_dataset(
    coco_json: Path,
    out_dir: Path,
    seed: int,
    force: bool,
) -> None:
    if not coco_json.is_file():
        _die(f"COCO JSON not found: {coco_json}")

    xrays_dir = coco_json.parent / "xrays"
    if not xrays_dir.is_dir():
        _die(f"xrays directory not found: {xrays_dir}")

    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            _die(
                f"output directory is not empty: {out_dir} "
                "(pass --force to overwrite)"
            )
        shutil.rmtree(out_dir)

    with coco_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    if not images:
        _die("no images found in COCO JSON")

    # image_id → list of YOLO lines
    labels: dict[int, list[str]] = defaultdict(list)
    skipped_boxes = 0
    for ann in coco.get("annotations", []):
        if ann.get("category_id_3") not in CARIES_CATEGORY_IDS:
            continue
        image_id = ann["image_id"]
        img = images.get(image_id)
        if img is None:
            skipped_boxes += 1
            continue
        yolo = _coco_bbox_to_yolo(ann["bbox"], img["width"], img["height"])
        if yolo is None:
            skipped_boxes += 1
            continue
        xc, yc, w, h = yolo
        labels[image_id].append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    items: list[tuple[int, bool]] = []
    missing_files = 0
    for image_id, img in images.items():
        src = xrays_dir / img["file_name"]
        if not src.is_file():
            missing_files += 1
            continue
        items.append((image_id, image_id in labels and len(labels[image_id]) > 0))

    if not items:
        _die("no usable images found (missing files?)")

    splits = _stratified_split(items, SPLIT_RATIOS, seed)

    link_mode_counts: dict[str, int] = defaultdict(int)
    box_counts: dict[str, int] = {k: 0 for k in splits}
    empty_counts: dict[str, int] = {k: 0 for k in splits}
    image_counts: dict[str, int] = {k: 0 for k in splits}

    for split_name, image_ids in splits.items():
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            img = images[image_id]
            src = xrays_dir / img["file_name"]
            dst_img = img_out / img["file_name"]
            mode = _link_or_copy(src, dst_img)
            link_mode_counts[mode] += 1

            lines = labels.get(image_id, [])
            stem = Path(img["file_name"]).stem
            (lbl_out / f"{stem}.txt").write_text(
                ("\n".join(lines) + ("\n" if lines else "")),
                encoding="utf-8",
            )
            image_counts[split_name] += 1
            box_counts[split_name] += len(lines)
            if not lines:
                empty_counts[split_name] += 1

    _write_data_yaml(out_dir)

    total_images = sum(image_counts.values())
    total_boxes = sum(box_counts.values())
    total_empty = sum(empty_counts.values())
    print(f"wrote {out_dir}")
    print(f"source: {coco_json}")
    print(f"seed: {seed}")
    print(
        "images: "
        + ", ".join(f"{k}={image_counts[k]}" for k in ("train", "val", "test"))
        + f" (total={total_images})"
    )
    print(
        "boxes: "
        + ", ".join(f"{k}={box_counts[k]}" for k in ("train", "val", "test"))
        + f" (total={total_boxes})"
    )
    print(
        "empty labels: "
        + ", ".join(f"{k}={empty_counts[k]}" for k in ("train", "val", "test"))
        + f" (total={total_empty})"
    )
    if missing_files:
        print(f"skipped missing image files: {missing_files}")
    if skipped_boxes:
        print(f"skipped invalid caries boxes: {skipped_boxes}")
    print(
        "file materialization: "
        + ", ".join(f"{k}={v}" for k, v in sorted(link_mode_counts.items()))
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert DENTEX disease COCO JSON to a YOLO caries dataset."
    )
    parser.add_argument(
        "--coco-json",
        default=str(DEFAULT_COCO.relative_to(REPO_ROOT)),
        help="Path to train_quadrant_enumeration_disease.json",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT.relative_to(REPO_ROOT)),
        help="Output directory (default: datasets/caries)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed (default: 42)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    prepare_dataset(
        coco_json=_resolve(args.coco_json),
        out_dir=_resolve(args.out),
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
