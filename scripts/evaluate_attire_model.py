"""Evaluate the trained attire classifier on test set."""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attire_classifier import EnhancedAttireClassifier
from src.model_evaluator import evaluate_model, print_evaluation_summary, plot_confusion_matrix


def main():
    """Evaluate the model on test set."""
    print("=" * 60)
    print("Enhanced Attire Classifier Evaluation")
    print("=" * 60)
    
    # Load trained model
    print("\n1. Loading trained model...")
    model_path = Path("models/attire_classifier_enhanced.joblib")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using: python scripts/train_attire_classifier.py")
        return
    
    clf = EnhancedAttireClassifier()
    clf.load(model_path)
    
    # Prepare test data
    print("\n2. Preparing test data...")
    dataset_root = Path("datasets")
    X_test, y_test, _ = clf.prepare_dataset(
        dataset_root,
        split="test",
        max_samples=200  # Limit for faster evaluation
    )
    
    # Make predictions
    print("\n3. Making predictions...")
    y_pred = clf.model.predict(X_test)
    y_proba = clf.model.predict_proba(X_test)
    
    # Evaluate
    print("\n4. Evaluating model...")
    results = evaluate_model(
        y_test, y_pred, y_proba,
        class_names=clf.classes_present
    )
    
    # Print summary
    print_evaluation_summary(results)
    
    # Plot confusion matrix
    print("\n5. Generating confusion matrix plot...")
    cm = np.array(results["confusion_matrix"])
    plot_path = Path("models/confusion_matrix.png")
    plot_confusion_matrix(cm, clf.classes_present, save_path=plot_path)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Test Accuracy: {results['accuracy']:.1%}")
    print(f"Confusion matrix saved to: {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
