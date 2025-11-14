#!/usr/bin/env python3
"""
Unified script to evaluate model accuracy with different levels of detail.

Usage:
  python evaluate_model.py --basic          # Basic accuracy check
  python evaluate_model.py --comprehensive  # Full metrics evaluation
  python evaluate_model.py --debug          # Debug version with warnings suppressed
"""

import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    precision_recall_fscore_support, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from src.config import AppConfig
from src.dataset import load_dataset
from src.model import AttireClassifier


def basic_accuracy_check():
    """Basic accuracy check - similar to accuracy_check.py"""
    print("=== Basic Dataset Accuracy Check ===\n")

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
    print(f"Mean: {cv_scores.mean():.3f}")
    print(f"Std: {cv_scores.std():.3f}")
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


def comprehensive_accuracy_check():
    """Comprehensive accuracy check - similar to comprehensive_accuracy_check.py"""
    print("=== Comprehensive Dataset Accuracy Check ===\n")

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

    # Cross-validation with multiple metrics
    print("=== CROSS-VALIDATION METRICS (5-fold) ===")
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    # Accuracy
    cv_accuracy = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  Individual folds: {cv_accuracy}")

    # Precision, Recall, F1
    cv_precision = cross_val_score(pipeline, X, y, cv=5, scoring='precision_macro')
    cv_recall = cross_val_score(pipeline, X, y, cv=5, scoring='recall_macro')
    cv_f1 = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')

    print(f"Precision (macro): {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"Recall (macro): {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"F1-Score (macro): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Balanced Accuracy
    cv_balanced = cross_val_score(pipeline, X, y, cv=5, scoring='balanced_accuracy')
    print(f"Balanced Accuracy: {cv_balanced.mean():.4f} ± {cv_balanced.std():.4f}")

    print()

    # Test set evaluation
    print("=== TEST SET EVALUATION (20% holdout) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Basic metrics
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
    print(f"Matthews Correlation: {matthews_corrcoef(y_test, y_pred):.4f}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    classes = pipeline.classes_
    print("\nPer-Class Metrics:")
    for i, cls in enumerate(classes):
        print(f"  {cls}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall: {recall[i]:.4f}")
        print(f"    F1-Score: {f1[i]:.4f}")
        print(f"    Support: {support[i]}")

    # Macro/Micro averages
    print("\nMacro Averages:")
    print(f"  Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\nMicro Averages:")
    print(f"  Precision: {precision_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred, average='micro'):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred, average='micro'):.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ROC-AUC (if binary classification)
    if len(classes) == 2:
        y_test_bin = label_binarize(y_test, classes=classes)
        roc_auc = roc_auc_score(y_test_bin, y_proba[:, 1])
        print(f"\nROC-AUC Score: {roc_auc:.4f}")

        # ROC Curve points
        fpr, tpr, _ = roc_curve(y_test_bin, y_proba[:, 1])
        print(f"AUC from curve: {auc(fpr, tpr):.4f}")

    # Feature Importance (top 10)
    if hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
        importances = pipeline.named_steps['clf'].feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print("\nTop 10 Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_cols[idx] if idx < len(feature_cols) else f'feat_{idx}'}: {importances[idx]:.4f}")

    print("\n=== END OF ACCURACY CHECK ===")


def debug_accuracy_check():
    """Debug accuracy check - similar to debug_accuracy.py"""
    print("=== Debug Dataset Accuracy Check ===\n")

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

    # Cross-validation with multiple metrics
    print("=== CROSS-VALIDATION METRICS (5-fold) ===")
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'))
    ])

    # Accuracy
    cv_accuracy = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
    print(f"Accuracy: {cv_accuracy.mean():.4f} ± {cv_accuracy.std():.4f}")
    print(f"  Individual folds: {cv_accuracy}")

    # Precision, Recall, F1
    cv_precision = cross_val_score(pipeline, X, y, cv=5, scoring='precision_macro')
    cv_recall = cross_val_score(pipeline, X, y, cv=5, scoring='recall_macro')
    cv_f1 = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')

    print(f"Precision (macro): {cv_precision.mean():.4f} ± {cv_precision.std():.4f}")
    print(f"Recall (macro): {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")
    print(f"F1-Score (macro): {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    # Balanced Accuracy
    cv_balanced = cross_val_score(pipeline, X, y, cv=5, scoring='balanced_accuracy')
    print(f"Balanced Accuracy: {cv_balanced.mean():.4f} ± {cv_balanced.std():.4f}")

    print()

    # Test set evaluation
    print("=== TEST SET EVALUATION (20% holdout) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Basic metrics
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"Cohen's Kappa: {cohen_kappa_score(y_test, y_pred):.4f}")
    print(f"Matthews Correlation: {matthews_corrcoef(y_test, y_pred):.4f}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)
    classes = pipeline.classes_
    print("\nPer-Class Metrics:")
    for i, cls in enumerate(classes):
        print(f"  {cls}:")
        print(f"    Precision: {precision[i]:.4f}")
        print(f"    Recall: {recall[i]:.4f}")
        print(f"    F1-Score: {f1[i]:.4f}")
        print(f"    Support: {support[i]}")

    # Macro/Micro averages
    print("\nMacro Averages:")
    print(f"  Precision: {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred, average='macro'):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

    print("\nMicro Averages:")
    print(f"  Precision: {precision_score(y_test, y_pred, average='micro', zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_test, y_pred, average='micro'):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred, average='micro'):.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # ROC-AUC (if binary classification)
    if len(classes) == 2:
        y_test_bin = label_binarize(y_test, classes=classes)
        roc_auc = roc_auc_score(y_test_bin, y_proba[:, 1])
        print(f"\nROC-AUC Score: {roc_auc:.4f}")

        # ROC Curve points
        fpr, tpr, _ = roc_curve(y_test_bin, y_proba[:, 1])
        print(f"AUC from curve: {auc(fpr, tpr):.4f}")

    # Feature Importance (top 10)
    if hasattr(pipeline.named_steps['clf'], 'feature_importances_'):
        importances = pipeline.named_steps['clf'].feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        print("\nTop 10 Important Features:")
        for i, idx in enumerate(indices):
            print(f"  {i+1}. {feature_cols[idx] if idx < len(feature_cols) else f'feat_{idx}'}: {importances[idx]:.4f}")

    print("\n=== END OF ACCURACY CHECK ===")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model accuracy")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--basic', action='store_true', help='Basic accuracy check')
    group.add_argument('--comprehensive', action='store_true', help='Comprehensive accuracy check')
    group.add_argument('--debug', action='store_true', help='Debug accuracy check with warnings suppressed')

    args = parser.parse_args()

    if args.debug:
        # Suppress warnings for debug mode
        warnings.filterwarnings('ignore')

    if args.basic:
        basic_accuracy_check()
    elif args.comprehensive:
        comprehensive_accuracy_check()
    elif args.debug:
        debug_accuracy_check()


if __name__ == "__main__":
    main()
