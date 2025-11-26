#!/usr/bin/env python3
"""
Script to preprocess the dataset with simple augmentations to improve accuracy.
"""

import os
import sys
from pathlib import Path
from typing import List
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AppConfig
from src.dataset import append_sample_to_dataset, ensure_dirs
from src.features import extract_features_from_image, extract_pose


def augment_image(image: np.ndarray) -> List[np.ndarray]:
    """Apply simple augmentations: flip, rotate, brightness."""
    augmented = [image]  # Original

    # Horizontal flip
    augmented.append(cv2.flip(image, 1))

    # Rotate ±10 degrees
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    for angle in [-10, 10]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented.append(rotated)

    # Brightness adjustment
    for factor in [0.8, 1.2]:
        bright = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        augmented.append(bright)

    return augmented


def preprocess_dataset(cfg: AppConfig, augmentation_factor: int = 3):
    """Augment the existing dataset to increase size and potentially improve accuracy."""
    ensure_dirs(cfg)

    # Load existing dataset
    if not cfg.meta_csv.exists():
        print("No existing dataset found. Run import_datasets.py first.")
        return 0

    df = pd.read_csv(cfg.meta_csv)
    print(f"Original dataset: {len(df)} samples")
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    augmented_count = 0

    for idx, row in df.iterrows():
        image_path = Path(row['image'])
        label = row['label']

        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        # Load image
        pil_img = Image.open(image_path).convert('RGB')
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Generate augmentations
        augmented_images = augment_image(bgr)

        # Skip original (already in dataset), add only augmentations
        for aug_img in augmented_images[1:]:
            try:
                # Extract features for augmented image
                pose = extract_pose(aug_img)
                features = extract_features_from_image(aug_img, pose_landmarks=pose, bins=cfg.hist_bins)

                # Append to dataset
                append_sample_to_dataset(aug_img, label, features, cfg)
                augmented_count += 1

                if augmented_count % 50 == 0:
                    print(f"Added {augmented_count} augmented samples...")

            except Exception as e:
                print(f"Error augmenting image {image_path}: {e}")
                continue

        # Limit total augmentations to prevent excessive growth
        if augmented_count >= len(df) * augmentation_factor:
            break

    print(f"Added {augmented_count} augmented samples")
    return augmented_count


def main():
    cfg = AppConfig()
    augmentation_factor = 3  # Create up to 3x more samples per original

    print("Starting dataset preprocessing with augmentations...")
    added = preprocess_dataset(cfg, augmentation_factor)

    # Reload and show final stats
    try:
        df = pd.read_csv(cfg.meta_csv)
        print(f"\nFinal dataset: {len(df)} samples")
        print(f"Labels: {df['label'].value_counts().to_dict()}")
    except Exception as e:
        print(f"Could not load final dataset: {e}")


if __name__ == "__main__":
    main()
