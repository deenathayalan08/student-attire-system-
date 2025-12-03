"""Model evaluation utilities."""
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from pathlib import Path

# Optional matplotlib import
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        class_names: Names of classes
    
    Returns:
        Dictionary with evaluation metrics
    """
    if class_names is None:
        class_names = ["Formal", "Semi-Formal", "Casual"]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "per_class": {
            class_names[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1_score": float(f1_per_class[i])
            }
            for i in range(len(class_names))
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Path = None,
    normalize: bool = False
) -> None:
    """Plot confusion matrix."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping plot generation.")
        return
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def calculate_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20
) -> Dict[str, float]:
    """Calculate and return feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'named_steps') and hasattr(model.named_steps['clf'], 'feature_importances_'):
        importances = model.named_steps['clf'].feature_importances_
    else:
        return {}
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    
    feature_importance = {
        feature_names[i]: float(importances[i])
        for i in indices
    }
    
    return feature_importance


def print_evaluation_summary(results: Dict[str, Any]) -> None:
    """Print formatted evaluation summary."""
    print("\n" + "=" * 60)
    print("Model Evaluation Summary")
    print("=" * 60)
    print(f"Overall Accuracy: {results['accuracy']:.1%}")
    print(f"Precision: {results['precision']:.1%}")
    print(f"Recall: {results['recall']:.1%}")
    print(f"F1-Score: {results['f1_score']:.1%}")
    
    print("\nPer-Class Performance:")
    print("-" * 60)
    for class_name, metrics in results['per_class'].items():
        print(f"{class_name:15s} | Precision: {metrics['precision']:.1%} | "
              f"Recall: {metrics['recall']:.1%} | F1: {metrics['f1_score']:.1%}")
    
    print("\nConfusion Matrix:")
    print("-" * 60)
    cm = np.array(results['confusion_matrix'])
    print(cm)
    print("=" * 60)
