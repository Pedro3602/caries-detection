"""Scaffold experiments/<family>/<recipe>/ folder structure.

Examples:
  python scripts/create_experiment.py --all
  python scripts/create_experiment.py --family classic --recipe geometric
  python scripts/create_experiment.py --family none --recipe baseline --force
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "datasets" / "caries"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

AUG_KEYS = [
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "hsv_h",
    "hsv_s",
    "hsv_v",
]

ZERO_AUG = {k: 0.0 for k in AUG_KEYS}


@dataclass(frozen=True)
class RecipeSpec:
    family: str
    name: str
    intent: str
    aug: dict[str, float]

    @property
    def key(self) -> str:
        return f"{self.family}/{self.name}"


def _aug(**overrides: float) -> dict[str, float]:
    cfg = dict(ZERO_AUG)
    cfg.update(overrides)
    return cfg


RECIPES: dict[tuple[str, str], RecipeSpec] = {
    ("none", "baseline"): RecipeSpec(
        family="none",
        name="baseline",
        intent="No meaningful data augmentation. Control run for comparing classic and crops families.",
        aug=_aug(),
    ),
    ("classic", "photometric"): RecipeSpec(
        family="classic",
        name="photometric",
        intent="Intensity / appearance variation only (brightness, contrast via HSV), without geometric warps or mosaic-style crops.",
        aug=_aug(hsv_s=0.5, hsv_v=0.3),
    ),
    ("classic", "geometric"): RecipeSpec(
        family="classic",
        name="geometric",
        intent="Pose / framing variation only: small rotation, translation, scale, and horizontal flip.",
        aug=_aug(degrees=5.0, translate=0.1, scale=0.3, fliplr=0.5),
    ),
    ("classic", "combined"): RecipeSpec(
        family="classic",
        name="combined",
        intent="Full classic stack: photometric + geometric transforms together, without mosaic/mixup crops.",
        aug=_aug(
            degrees=5.0,
            translate=0.1,
            scale=0.3,
            fliplr=0.5,
            hsv_s=0.5,
            hsv_v=0.3,
        ),
    ),
    ("crops", "mosaic"): RecipeSpec(
        family="crops",
        name="mosaic",
        intent="Composition / crop-style training via 4-image mosaic. Isolates mosaic relative to classic augs.",
        aug=_aug(mosaic=1.0),
    ),
    ("crops", "mosaic_mixup"): RecipeSpec(
        family="crops",
        name="mosaic_mixup",
        intent="Mosaic plus light MixUp blending for stronger composition-style regularization.",
        aug=_aug(mosaic=1.0, mixup=0.15),
    ),
}


def _die(message: str, code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _allowed_list() -> str:
    return ", ".join(sorted(spec.key for spec in RECIPES.values()))


def _readme_has_analysis(text: str) -> bool:
    marker = "## Analysis"
    if marker not in text:
        return False
    after = text.split(marker, 1)[1].strip()
    if not after:
        return False
    # Placeholder lines from the template
    placeholders = {"_to be filled after training._", "to be filled after training."}
    lines = [ln.strip() for ln in after.splitlines() if ln.strip()]
    if not lines:
        return False
    return any(ln.lower() not in placeholders for ln in lines)


def _format_recipe_yaml(spec: RecipeSpec) -> str:
    lines = [f"# Family: {spec.family} — {spec.name}"]
    for key in AUG_KEYS:
        value = spec.aug[key]
        if float(value).is_integer():
            lines.append(f"{key}: {value:.1f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def _format_readme(spec: RecipeSpec) -> str:
    return (
        f"# Recipe: {spec.name} (`{spec.family}`)\n\n"
        f"## Intent\n\n"
        f"{spec.intent}\n\n"
        f"## Augmentation\n\n"
        f"See `recipe.yaml` for Ultralytics augmentation hyperparameters.\n\n"
        f"## Analysis\n\n"
        f"_To be filled after training._\n"
    )


def _format_data_yaml(data_dir: Path) -> str:
    return (
        f"path: {data_dir.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        "  0: caries\n"
    )


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_recipe(spec: RecipeSpec, data_dir: Path, force: bool) -> list[str]:
    """Create or refresh one recipe folder. Returns list of actions taken."""
    recipe_dir = EXPERIMENTS_DIR / spec.family / spec.name
    dataset_dir = recipe_dir / "dataset"
    previews_dir = dataset_dir / "previews"
    results_dir = recipe_dir / "results"

    actions: list[str] = []
    recipe_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    for keep in (previews_dir / ".gitkeep", results_dir / ".gitkeep"):
        if not keep.exists():
            keep.touch()
            actions.append(f"created {keep.relative_to(REPO_ROOT)}")

    recipe_yaml = recipe_dir / "recipe.yaml"
    if force or not recipe_yaml.exists():
        existed = recipe_yaml.exists()
        _write_text(recipe_yaml, _format_recipe_yaml(spec))
        actions.append(
            f"{'overwrote' if existed else 'wrote'} {recipe_yaml.relative_to(REPO_ROOT)}"
        )
    else:
        actions.append(f"kept {recipe_yaml.relative_to(REPO_ROOT)}")

    readme = recipe_dir / "README.md"
    if readme.exists() and _readme_has_analysis(readme.read_text(encoding="utf-8")):
        # Preserve filled analysis even with --force
        actions.append(f"kept {readme.relative_to(REPO_ROOT)} (has analysis)")
    else:
        _write_text(readme, _format_readme(spec))
        actions.append(f"wrote {readme.relative_to(REPO_ROOT)}")

    data_yaml = dataset_dir / "data.yaml"
    _write_text(data_yaml, _format_data_yaml(data_dir))
    actions.append(f"wrote {data_yaml.relative_to(REPO_ROOT)}")

    return actions


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scaffold augmentation experiment recipe folders."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create all known recipes",
    )
    parser.add_argument("--family", default=None, help="Recipe family (none|classic|crops)")
    parser.add_argument("--recipe", default=None, help="Recipe name within the family")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA.relative_to(REPO_ROOT)),
        help="Shared dataset root (default: datasets/caries)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite recipe.yaml and refresh README if Analysis is empty",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    has_pair = args.family is not None or args.recipe is not None
    if args.all and has_pair:
        _die("use either --all or --family/--recipe, not both")
    if not args.all and not has_pair:
        _die("specify --all or both --family and --recipe")
    if has_pair and (args.family is None or args.recipe is None):
        _die("--family and --recipe must be used together")

    data_dir = _resolve(args.data)
    if not (data_dir / "data.yaml").is_file():
        _die(
            f"shared dataset not found at {data_dir} "
            "(run: python scripts/prepare_dataset.py first)"
        )

    if args.all:
        specs = [RECIPES[key] for key in sorted(RECIPES)]
    else:
        key = (args.family, args.recipe)
        if key not in RECIPES:
            _die(
                f"unknown recipe '{args.family}/{args.recipe}'. "
                f"Allowed: {_allowed_list()}"
            )
        specs = [RECIPES[key]]

    for spec in specs:
        print(f"== {spec.key}")
        for action in create_recipe(spec, data_dir, force=args.force):
            print(f"  {action}")


if __name__ == "__main__":
    main()
