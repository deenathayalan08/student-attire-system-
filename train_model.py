#!/usr/bin/env python3
"""
Train the attire classifier model and save it.
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AppConfig
from src.dataset import load_dataset
from src.model import AttireClassifier


def main():
    cfg = AppConfig()

    print("Loading dataset...")
    df = load_dataset(cfg)
    if df is None or df.empty:
        print("No dataset found. Run import_datasets.py first.")
        return

    print(f"Dataset loaded: {len(df)} samples")
    print(f"Labels: {df['label'].value_counts().to_dict()}")

    print("Training model...")
    clf = AttireClassifier()
    result = clf.train_from_dataframe(df)

    print(".2%")
    print(f"Samples: {result['num_samples']}, Classes: {result['num_classes']}, CV folds: {result['cv_folds']}")

    print("Saving model...")
    clf.save(cfg.model_path, cfg)
    print(f"Model saved to {cfg.model_path}")


if __name__ == "__main__":
    main()
