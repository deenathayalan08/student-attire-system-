#!/usr/bin/env python3
"""
Import all available datasets including the large footwears dataset to improve accuracy.
"""

import os
import sys
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AppConfig
from src.dataset import append_sample_to_dataset, ensure_dirs
from src.features import extract_features_from_image, extract_pose


def import_dataset_folder(folder_path: str, label: str, cfg: AppConfig, max_samples: int = None):
    """Import images from a dataset folder."""
    folder_path = Path(folder_path)
    if not folder_path.exists():
        print(f"Folder {folder_path} does not exist")
        return 0

    print(f"Importing from {folder_path} with label '{label}'...")

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(list(folder_path.rglob(ext)))

    if max_samples:
        image_files = image_files[:max_samples]

    imported = 0
    for img_path in tqdm(image_files, desc=f"Importing {label}"):
        try:
            # Load image
            pil_img = cv2.imread(str(img_path))
            if pil_img is None:
                continue

            # Convert to RGB for PIL compatibility
            rgb_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)

            # Extract features
            pose = extract_pose(pil_img)
            features = extract_features_from_image(pil_img, pose_landmarks=pose, bins=cfg.hist_bins)

            # Append to dataset
            append_sample_to_dataset(pil_img, label, features, cfg)
            imported += 1

        except Exception as e:
            print(f"Error importing {img_path}: {e}")
            continue

    print(f"Imported {imported} samples from {folder_path}")
    return imported


def main():
    cfg = AppConfig()

    print("Starting comprehensive dataset import...")

    # Import from datasets folder (large dataset)
    datasets_to_import = [
        ('datasets/train', 'compliant'),  # Assuming train has compliant samples
        ('datasets/valid', 'compliant'),  # Assuming valid has compliant samples
        ('datasets/test', 'non-compliant'),  # Assuming test has non-compliant samples
        ('datasets/footwears/train', 'compliant'),  # Footwear compliance
        ('datasets/uniform 1/train', 'compliant'),
        ('datasets/uniform 1/valid', 'compliant'),
        ('datasets/uniform 1/test', 'non-compliant'),
        ('datasets/uniform 2/train', 'compliant'),
        ('datasets/uniform 2/valid', 'compliant'),
        ('datasets/uniform 2/test', 'non-compliant'),
    ]

    total_imported = 0

    for folder, label in datasets_to_import:
        if os.path.exists(folder):
            imported = import_dataset_folder(folder, label, cfg, max_samples=1000)  # Limit to prevent excessive processing
            total_imported += imported
        else:
            print(f"Skipping {folder} - does not exist")

    # combined_dataset removed as it was redundant with datasets/

    print(f"\nTotal imported: {total_imported}")

    # Show final stats
    try:
        df = pd.read_csv(cfg.meta_csv)
        print(f"Final dataset: {len(df)} samples")
        print(f"Labels: {df['label'].value_counts().to_dict()}")
    except Exception as e:
        print(f"Could not load final dataset: {e}")


if __name__ == "__main__":
    main()
