"""Training script for enhanced attire classifier."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attire_classifier import EnhancedAttireClassifier
from src.dataset_analyzer import analyze_dataset_structure


def main():
    """Train the enhanced attire classifier."""
    print("=" * 60)
    print("Enhanced Attire Classifier Training")
    print("=" * 60)
    
    # Dataset path
    dataset_root = Path("datasets")
    
    # Analyze dataset
    print("\n1. Analyzing dataset structure...")
    stats = analyze_dataset_structure(dataset_root)
    print(f"   Train: {stats['train']['count']} images")
    print(f"   Valid: {stats['valid']['count']} images")
    print(f"   Test: {stats['test']['count']} images")
    print(f"   Total: {stats['total_images']} images")
    
    # Initialize classifier
    print("\n2. Initializing classifier...")
    clf = EnhancedAttireClassifier()
    
    # Prepare training data
    print("\n3. Preparing training data...")
    X_train, y_train, feature_names = clf.prepare_dataset(
        dataset_root,
        split="train",
        max_samples=500  # Limit for faster training, remove for full dataset
    )
    
    # Prepare validation data
    print("\n4. Preparing validation data...")
    X_val, y_val, _ = clf.prepare_dataset(
        dataset_root,
        split="valid",
        max_samples=200  # Limit for faster validation
    )
    
    # Train model
    print("\n5. Training model...")
    training_stats = clf.train(X_train, y_train, X_val, y_val)
    
    # Save model
    print("\n6. Saving model...")
    model_path = Path("models/attire_classifier_enhanced.joblib")
    clf.save(model_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"CV Accuracy: {training_stats['cv_accuracy']:.1%}")
    print(f"Train Accuracy: {training_stats['train_accuracy']:.1%}")
    if "val_accuracy" in training_stats:
        print(f"Validation Accuracy: {training_stats['val_accuracy']:.1%}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
