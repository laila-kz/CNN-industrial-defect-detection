#!/usr/bin/env python3
"""
organize_kaggle_data.py
========================
Organizes the Kaggle Severstal Steel Defect Detection dataset into the
train/val/test folder structure expected by the C++ CNN pipeline.

Expected raw Kaggle layout (extracted into data/):
    data/
    ├── train_images/   ← ~12,568 JPEG images
    ├── test_images/    ← unlabeled test images (optional)
    └── train.csv       ← RLE annotation file (ImageId, ClassId, EncodedPixels)

Output structure produced by this script:
    data/organized/
    ├── train/
    │   ├── OK/
    │   └── DEFECT/
    ├── val/
    │   ├── OK/
    │   └── DEFECT/
    ├── test/
    │   ├── OK/
    │   └── DEFECT/
    ├── dataset_summary.txt
    └── dataset_config.yaml

Usage:
    cd <project_root>
    python data/organize_kaggle_data.py [--data-dir PATH] [--output-dir PATH] [--seed N] [--yes]
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dependency guards — fail fast with a helpful message
# ---------------------------------------------------------------------------
try:
    import numpy as np
except ImportError:
    sys.exit("[ERROR] numpy is required. Install it with: pip install numpy")

try:
    import pandas as pd
except ImportError:
    sys.exit("[ERROR] pandas is required. Install it with: pip install pandas")

try:
    import yaml
except ImportError:
    sys.exit("[ERROR] pyyaml is required. Install it with: pip install pyyaml")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Organize the Kaggle Severstal Steel Defect dataset for CNN training."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Root directory containing train_images/ and train.csv (default: ./data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for organized splits (default: <data-dir>/organized)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val/test splits (default: 42)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of images used for training (default: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of images used for validation (default: 0.1; remainder is test)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt and run immediately",
    )
    return parser.parse_args()


def validate_inputs(data_dir: Path, train_images_dir: Path, train_csv: Path) -> None:
    """Raise a RuntimeError with a helpful message if any required path is missing."""
    missing = []
    if not data_dir.exists():
        missing.append(f"  • data directory  : {data_dir}")
    if not train_images_dir.exists():
        missing.append(f"  • train_images/   : {train_images_dir}")
    if not train_csv.is_file():
        missing.append(f"  • train.csv       : {train_csv}")

    if missing:
        msg = (
            "\n[ERROR] The following required paths do not exist:\n"
            + "\n".join(missing)
            + "\n\nPlease extract the Kaggle dataset into the data/ folder.\n"
            "Expected layout:\n"
            "  data/\n"
            "  ├── train_images/   ← extracted from severstal-steel-defect-detection.zip\n"
            "  └── train.csv\n"
        )
        raise RuntimeError(msg)


def identify_defective_images(train_csv: Path) -> dict[str, bool]:
    """
    Parse train.csv and return a mapping of {image_id: has_defect}.

    The Kaggle CSV has columns: ImageId, ClassId, EncodedPixels.
    An image is defective if *any* row for that ImageId has a non-null
    EncodedPixels value.
    """
    log.info("Reading annotation file: %s", train_csv)
    try:
        df = pd.read_csv(train_csv)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {train_csv}: {exc}") from exc

    required_cols = {"ImageId", "EncodedPixels"}
    if not required_cols.issubset(df.columns):
        raise RuntimeError(
            f"train.csv is missing required columns. "
            f"Found: {list(df.columns)}. Expected at least: {required_cols}"
        )

    log.info("Loaded %d annotation rows for %d unique images.", len(df), df["ImageId"].nunique())

    # An image is defective if any EncodedPixels entry is non-null and non-empty
    defect_map: dict[str, bool] = {}
    for image_id, group in df.groupby("ImageId"):
        has_defect = group["EncodedPixels"].apply(
            lambda v: pd.notna(v) and str(v).strip() != ""
        ).any()
        defect_map[str(image_id)] = bool(has_defect)

    n_defect = sum(defect_map.values())
    n_ok = len(defect_map) - n_defect
    log.info(
        "Image analysis: total=%d  with_defect=%d (%.1f%%)  ok=%d (%.1f%%)",
        len(defect_map),
        n_defect, n_defect / len(defect_map) * 100,
        n_ok,    n_ok    / len(defect_map) * 100,
    )
    return defect_map


def split_images(
    defect_map: dict[str, bool],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """Split image IDs into train / val / test with stratification on defect label."""
    rng = np.random.default_rng(seed)
    all_ids = np.array(list(defect_map.keys()))
    rng.shuffle(all_ids)

    n = len(all_ids)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio  * n)

    train_ids = all_ids[:n_train].tolist()
    val_ids   = all_ids[n_train : n_train + n_val].tolist()
    test_ids  = all_ids[n_train + n_val :].tolist()

    log.info(
        "Split: train=%d  val=%d  test=%d",
        len(train_ids), len(val_ids), len(test_ids),
    )
    return train_ids, val_ids, test_ids


def copy_split(
    image_ids: list[str],
    defect_map: dict[str, bool],
    src_dir: Path,
    dst_dir: Path,
    split_name: str,
) -> dict[str, int]:
    """
    Copy images from src_dir into dst_dir/{OK,DEFECT}/ according to defect_map.

    Returns a summary dict: {ok, defect, copied, skipped}.
    """
    ok_dir     = dst_dir / split_name / "OK"
    defect_dir = dst_dir / split_name / "DEFECT"
    ok_dir.mkdir(parents=True, exist_ok=True)
    defect_dir.mkdir(parents=True, exist_ok=True)

    counts = {"ok": 0, "defect": 0, "copied": 0, "skipped": 0}

    for img_id in image_ids:
        is_defect = defect_map[img_id]
        dst_folder = defect_dir if is_defect else ok_dir

        # Try exact path first, then common image extensions
        src = src_dir / img_id
        if not src.is_file():
            found = False
            for ext in (".jpg", ".jpeg", ".png"):
                candidate = src.with_suffix(ext)
                if candidate.is_file():
                    src = candidate
                    found = True
                    break
            if not found:
                log.warning("Image not found, skipping: %s", src_dir / img_id)
                counts["skipped"] += 1
                continue

        shutil.copy2(src, dst_folder / src.name)
        counts["copied"] += 1
        if is_defect:
            counts["defect"] += 1
        else:
            counts["ok"] += 1

    log.info(
        "[%s] Copied %d images (%d OK / %d DEFECT), skipped %d.",
        split_name.upper(),
        counts["copied"], counts["ok"], counts["defect"], counts["skipped"],
    )
    return counts


def write_summary(
    output_dir: Path,
    defect_map: dict[str, bool],
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    split_counts: dict[str, dict[str, int]],
) -> None:
    """Write both a human-readable .txt summary and a machine-readable .yaml config."""
    total = len(defect_map)
    n_defect = sum(defect_map.values())
    n_ok = total - n_defect

    # ---- Text summary ----
    summary_path = output_dir / "dataset_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Kaggle Steel Defect Dataset — Organized Structure\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated on : {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Total images    : {total}\n")
        f.write(f"  With defects    : {n_defect}  ({n_defect / total * 100:.1f}%)\n")
        f.write(f"  Without defects : {n_ok}  ({n_ok / total * 100:.1f}%)\n\n")

        for split, ids in [("TRAIN", train_ids), ("VAL", val_ids), ("TEST", test_ids)]:
            c = split_counts[split.lower()]
            f.write(f"{split} SET ({len(ids)} images):\n")
            f.write(f"  OK     : {c['ok']}\n")
            f.write(f"  DEFECT : {c['defect']}\n")
            f.write(f"  Skipped: {c['skipped']}\n\n")

        total_copied  = sum(v["copied"]  for v in split_counts.values())
        total_skipped = sum(v["skipped"] for v in split_counts.values())
        f.write(f"COPYING: copied={total_copied}  skipped={total_skipped}\n")
    log.info("Summary written → %s", summary_path)

    # ---- YAML config for C++ project ----
    yaml_cfg = {
        "dataset": {
            "name": "Kaggle Steel Defect Detection",
            "source": "Kaggle — Severstal Steel Defect Detection",
            "date_organized": pd.Timestamp.now().strftime("%Y-%m-%d"),
            "classes": ["OK", "DEFECT"],
            "image_extensions": [".jpg"],
        },
        "statistics": {
            "total_images": total,
            "defect_images": n_defect,
            "ok_images": n_ok,
            "train_ok":     split_counts["train"]["ok"],
            "train_defect": split_counts["train"]["defect"],
            "val_ok":       split_counts["val"]["ok"],
            "val_defect":   split_counts["val"]["defect"],
            "test_ok":      split_counts["test"]["ok"],
            "test_defect":  split_counts["test"]["defect"],
        },
        "paths": {
            "base": str(output_dir),
            "train": {"ok": "train/OK", "defect": "train/DEFECT"},
            "val":   {"ok": "val/OK",   "defect": "val/DEFECT"},
            "test":  {"ok": "test/OK",  "defect": "test/DEFECT"},
        },
        "split_ratios": {
            "train": round(len(train_ids) / total, 4),
            "val":   round(len(val_ids)   / total, 4),
            "test":  round(len(test_ids)  / total, 4),
        },
        "notes": "Organized for binary classification (OK vs DEFECT)",
    }

    yaml_path = output_dir / "dataset_config.yaml"
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        log.info("YAML config written → %s", yaml_path)
    except Exception as exc:
        log.warning("Could not write YAML config: %s", exc)


def main() -> None:
    args = parse_args()

    # Resolve paths
    data_dir        = args.data_dir.resolve()
    output_dir      = (args.output_dir or data_dir / "organized").resolve()
    train_images_dir = data_dir / "train_images"
    train_csv        = data_dir / "train.csv"

    log.info("Project root  : %s", Path.cwd())
    log.info("Data dir      : %s", data_dir)
    log.info("Output dir    : %s", output_dir)

    # Validate inputs before doing anything destructive
    validate_inputs(data_dir, train_images_dir, train_csv)

    n_images = sum(1 for f in train_images_dir.iterdir()
                   if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
    log.info("Found %d image files in %s", n_images, train_images_dir)

    if not args.yes:
        print(
            f"\nThis will organize {n_images} images into:\n"
            f"  {output_dir}/train/  |  val/  |  test/\n"
        )
        resp = input("Continue? [y/N] ").strip().lower()
        if resp != "y":
            print("Cancelled.")
            return

    # --- Core pipeline ---
    defect_map = identify_defective_images(train_csv)
    train_ids, val_ids, test_ids = split_images(
        defect_map,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_counts: dict[str, dict[str, int]] = {}
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        split_counts[split_name] = copy_split(
            ids, defect_map, train_images_dir, output_dir, split_name
        )

    write_summary(output_dir, defect_map, train_ids, val_ids, test_ids, split_counts)

    print("\n✅  Dataset organized successfully!")
    print(f"    → {output_dir}")
    print("\nNext steps:")
    print("  1. Update config/config.yaml → paths.train_img / val_img / test_img")
    print("  2. Run the C++ training binary: ./CNNIndustrialDefectsTraining")


if __name__ == "__main__":
    main()
