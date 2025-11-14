#!/usr/bin/env python3
"""
Script to import existing datasets into the student attire verification app's dataset collection system.
This will extract features from images in the datasets/ and combined_dataset/ folders and add them
to the app's metadata.csv for ML training.

Usage:
    python import_datasets.py --individual  # Import from individual dataset folders
    python import_datasets.py --bulk        # Import from all available datasets (bulk import)
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AppConfig
from src.dataset import append_sample_to_dataset, ensure_dirs
from src.features import extract_features_from_image, extract_pose


def import_dataset_images(dataset_path: Path, label: str, cfg: AppConfig, max_images: int = None):
    """
    Import images from a dataset folder into the app's dataset system.

    Args:
        dataset_path: Path to the dataset folder containing images
        label: Label to assign to all images in this dataset
        cfg: AppConfig instance
        max_images: Maximum number of images to import (None for all)
    """
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist, skipping...")
        return 0

    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(dataset_path.rglob(ext)))

    if not image_files:
        print(f"No images found in {dataset_path}")
        return 0

    # Limit number of images if specified
    if max_images:
        image_files = image_files[:max_images]

    print(f"Importing {len(image_files)} images from {dataset_path} with label '{label}'")

    imported_count = 0
    for img_path in image_files:
        try:
            # Load image
            pil_img = Image.open(img_path).convert('RGB')
            bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Extract pose and features
            pose = extract_pose(bgr)
            features = extract_features_from_image(bgr, pose_landmarks=pose, bins=cfg.hist_bins)

            # Append to dataset
            append_sample_to_dataset(bgr, label, features, cfg)

            imported_count += 1

            if imported_count % 10 == 0:
                print(f"Imported {imported_count}/{len(image_files)} images...")

        except Exception as e:
            print(f"Error importing {img_path}: {e}")
            continue

    print(f"Successfully imported {imported_count} images from {dataset_path}")
    return imported_count


def import_individual_datasets(cfg: AppConfig):
    """Import from individual dataset folders with limited samples."""
    total_imported = 0

    # Import from combined_dataset
    combined_datasets = [
        ("combined_dataset/uniform_1/train", "compliant"),
        ("combined_dataset/uniform_1/test", "compliant"),
        ("combined_dataset/uniform_1/valid", "compliant"),
        ("combined_dataset/uniform_2/train", "compliant"),
        ("combined_dataset/uniform_2/test", "compliant"),
        ("combined_dataset/uniform_2/valid", "compliant"),
    ]

    for dataset_folder, label in combined_datasets:
        dataset_path = Path(dataset_folder)
        count = import_dataset_images(dataset_path, label, cfg, max_images=100)  # Limit for testing
        total_imported += count

    # Import from datasets folder
    datasets_folders = [
        ("datasets/train", "compliant"),  # Assuming train folder contains compliant examples
        ("datasets/test", "compliant"),   # Assuming test folder contains compliant examples
        ("datasets/valid", "compliant"),  # Assuming valid folder contains compliant examples
        ("datasets/footwears/train", "compliant"),  # Footwear training data
    ]

    for dataset_folder, label in datasets_folders:
        dataset_path = Path(dataset_folder)
        count = import_dataset_images(dataset_path, label, cfg, max_images=50)  # Limit for testing
        total_imported += count

    return total_imported


def import_bulk_datasets(cfg: AppConfig):
    """Import from all available datasets (bulk import)."""
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

    return total_imported


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
    parser = argparse.ArgumentParser(description="Import datasets for student attire verification")
    parser.add_argument('--individual', action='store_true', help='Import from individual dataset folders')
    parser.add_argument('--bulk', action='store_true', help='Import from all available datasets (bulk import)')

    args = parser.parse_args()

    if not args.individual and not args.bulk:
        print("Please specify --individual or --bulk")
        return

    cfg = AppConfig()
    ensure_dirs(cfg)

    if args.individual:
        print("Starting individual dataset import...")
        total_imported = import_individual_datasets(cfg)
    elif args.bulk:
        print("Starting bulk dataset import...")
        total_imported = import_bulk_datasets(cfg)

    print(f"\nTotal images imported: {total_imported}")

    # Load and display dataset summary
    try:
        df = pd.read_csv(cfg.meta_csv)
        print(f"\nDataset Summary:")
        print(f"Total samples: {len(df)}")
        print(f"Labels: {df['label'].value_counts().to_dict()}")
        print(f"Columns: {len(df.columns)}")
        print(f"Sample columns: {list(df.columns[:10])}...")
    except Exception as e:
        print(f"Could not load dataset summary: {e}")


if __name__ == "__main__":
    main()
