#!/usr/bin/env python3
"""
evaluate_dataset.py

Standalone script to compute dataset accuracy without modifying the project.

Usage examples:
  # If you have a metadata CSV already (AppConfig.meta_csv):
  python evaluate_dataset.py --use-meta

  # If you have images organized in class subfolders:
  python evaluate_dataset.py --images-dir datasets/mydataset/images --label-from-folder

  # If you have a csv mapping image->label:
  python evaluate_dataset.py --images-dir datasets/mydataset/images --labels-csv mapping.csv

The script will extract features using the project's `src.features.extract_features_from_image`
and evaluate a LogisticRegression classifier (the same wrapper used by the app).
This script does not modify any existing project files.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

try:
    from src.config import AppConfig
    from src.features import extract_features_from_image, extract_pose
    from src.model import AttireClassifier
except Exception as e:
    print(f"Error importing project modules: {e}")
    raise


def extract_features_for_image(image_path: Path, bins: int) -> Dict[str, float]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    pose = extract_pose(img)
    features = extract_features_from_image(img, pose_landmarks=pose, bins=bins)
    return features


def build_dataframe_from_image_list(image_paths: List[Path], labels: List[str], bins: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    all_keys = set()
    for p, lab in tqdm(list(zip(image_paths, labels)), desc="Extracting features", total=len(image_paths)):
        feats = extract_features_for_image(p, bins=bins)
        row = {"image": str(p), "label": lab}
        # keep numeric scalar features
        for k, v in feats.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                row[k] = float(v)
                all_keys.add(k)
        rows.append(row)

    # Build DataFrame with consistent column order
    cols = ["image", "label"] + sorted(all_keys)
    df = pd.DataFrame(rows, columns=cols)
    df = df.fillna(0.0)
    return df


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate dataset accuracy using project's feature extractor and classifier")
    p.add_argument("--use-meta", action="store_true", help="Use AppConfig.meta_csv (data/metadata.csv) if present")
    p.add_argument("--meta-csv", type=str, default=None, help="Path to metadata CSV (overrides --use-meta)")
    p.add_argument("--images-dir", type=str, default=None, help="Directory with images to process")
    p.add_argument("--labels-csv", type=str, default=None, help="CSV file with columns: image,label to map images to labels")
    p.add_argument("--label-from-folder", action="store_true", help="Infer labels from parent folder name of each image")
    p.add_argument("--bins", type=int, default=24, help="Histogram bins used by feature extractor")
    p.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    p.add_argument("--save-model", type=str, default=None, help="Optional path to save trained model")
    args = p.parse_args(argv)

    cfg = AppConfig()

    # Step 1: Try metadata CSV
    meta_path = None
    if args.meta_csv:
        meta_path = Path(args.meta_csv)
    elif args.use_meta:
        meta_path = Path(cfg.meta_csv)

    if meta_path and meta_path.exists():
        print(f"Loading metadata CSV: {meta_path}")
        df = pd.read_csv(meta_path)
        if "label" not in df.columns:
            raise RuntimeError("metadata CSV must contain a 'label' column")
        # Use classifier training helper
        clf = AttireClassifier()
        print("Training classifier with metadata CSV (this will run cross-validation)")
        X, feature_cols = clf._features_from_df(df)
        y = df["label"].astype(str).to_numpy()
        res = clf.fit(X, y, feature_names=feature_cols)
        print(f"Cross-val accuracy (cv=3): {res.get('cv_accuracy', 0.0):.4f}")
        if args.save_model:
            clf.save(args.save_model, cfg=cfg)
        return

    # Step 2: Build dataset from images
    if not args.images_dir:
        raise RuntimeError("No dataset found. Provide --images-dir or use --use-meta / --meta-csv")

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise RuntimeError(f"Images directory not found: {images_dir}")

    # Gather image paths
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]
    if not image_paths:
        raise RuntimeError(f"No images found under {images_dir}")

    labels = []
    if args.labels_csv:
        mapping = pd.read_csv(args.labels_csv)
        if not {"image", "label"}.issubset(mapping.columns):
            raise RuntimeError("labels CSV must contain 'image' and 'label' columns")
        map_dict = {str(Path(r["image"]).name): r["label"] for _, r in mapping.iterrows()}
        for p in image_paths:
            lab = map_dict.get(p.name)
            if lab is None:
                raise RuntimeError(f"No label found for image {p.name} in labels CSV")
            labels.append(str(lab))
    elif args.label_from_folder:
        for p in image_paths:
            labels.append(p.parent.name)
    else:
        raise RuntimeError("To use images dir you must pass --labels-csv or --label-from-folder")

    # Build DataFrame
    df = build_dataframe_from_image_list(image_paths, labels, bins=args.bins)

    # Train & evaluate
    clf = AttireClassifier()
    X, feature_cols = clf._features_from_df(df)
    y = df["label"].astype(str).to_numpy()
    print(f"Dataset samples: {X.shape[0]}, features: {X.shape[1]}")
    res = clf.fit(X, y, feature_names=feature_cols)
    print("Evaluation results:")
    print(f"  Cross-val accuracy (cv=3): {res.get('cv_accuracy', 0.0):.4f}")
    print(f"  Num samples: {res.get('num_samples')}")

    if args.save_model:
        clf.save(args.save_model, cfg=cfg)
        print(f"Saved model to: {args.save_model}")


if __name__ == "__main__":
    main()
