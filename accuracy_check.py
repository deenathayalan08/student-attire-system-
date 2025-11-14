#!/usr/bin/env python3
"""
Simple script to check dataset accuracy and display results.
"""

import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import AppConfig
from src.dataset import load_dataset
from src.model import AttireClassifier


def main():
    print("=== Dataset Accuracy Check ===\n")

    # Load configuration and dataset
    cfg = AppConfig()
    df = load_dataset(cfg)

    print(f"Dataset: {len(df)} samples")
    print(f"Labels: {df['label'].value_counts().to_dict()}\n")

    # Load trained model
    clf = AttireClassifier()
    clf.load()

    # Extract features
    X, feature_cols = clf._features_from_df(df)
    y = df['label'].astype(str).to_numpy()

    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(set(y))}\n")

    # Cross-validation
    print("Cross-Validation (5-fold):")
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"Scores: {cv_scores}")
    print(".3f")
    print(".3f")
    print()

    # Test set evaluation
    print("Test Set Evaluation (20% holdout):")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.model.fit(X_train, y_train)
    y_pred = clf.model.predict(X_test)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
