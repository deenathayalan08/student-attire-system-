"""
Train Enhanced Attire Classification Model
Uses dataset to train Formal/Semi-Formal/Casual classifier
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attire_classifier import AttireClassifier
from src.dataset_analyzer import DatasetAnalyzer


def main():
    print("=" * 70)
    print("PHASE 3: ENHANCED ATTIRE CLASSIFICATION TRAINING")
    print("=" * 70)
    print()
    
    # Step 1: Analyze dataset
    print("Step 1: Analyzing Dataset...")
    print("-" * 70)
    analyzer = DatasetAnalyzer()
    report = analyzer.generate_report()
    print(report)
    print()
    
    # Step 2: Train classifier
    print("Step 2: Training Attire Classifier...")
    print("-" * 70)
    classifier = AttireClassifier()
    
    training_results = classifier.train()
    
    if 'error' in training_results:
        print(f"❌ Training failed: {training_results['error']}")
        return
    
    print()
    print("✅ Training Complete!")
    print(f"   Samples trained: {training_results['num_samples']}")
    print(f"   Classes: {training_results['num_classes']}")
    print(f"   Class names: {', '.join(training_results['class_names'])}")
    print(f"   Cross-validation accuracy: {training_results['cv_mean_accuracy']:.2%} ± {training_results['cv_std_accuracy']:.2%}")
    print()
    
    # Step 3: Evaluate on test set
    print("Step 3: Evaluating on Test Set...")
    print("-" * 70)
    eval_results = classifier.evaluate()
    
    if 'error' in eval_results:
        print(f"⚠️  Evaluation skipped: {eval_results['error']}")
    else:
        print()
        print("✅ Evaluation Complete!")
        print(f"   Test samples: {eval_results['num_test_samples']}")
        print(f"   Test accuracy: {eval_results['accuracy']:.2%}")
        print()
        
        # Show per-class metrics
        print("   Per-Class Performance:")
        report = eval_results['classification_report']
        for class_name in eval_results['class_names']:
            if class_name in report:
                metrics = report[class_name]
                print(f"     {class_name}:")
                print(f"       Precision: {metrics['precision']:.2%}")
                print(f"       Recall: {metrics['recall']:.2%}")
                print(f"       F1-Score: {metrics['f1-score']:.2%}")
        print()
    
    # Summary
    print("=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    print()
    print("✅ Model trained and saved successfully!")
    print(f"✅ Model location: models/attire_classifier.joblib")
    print(f"✅ Training accuracy: {training_results['cv_mean_accuracy']:.2%}")
    if 'error' not in eval_results:
        print(f"✅ Test accuracy: {eval_results['accuracy']:.2%}")
    print()
    print("Next steps:")
    print("  1. Integrate classifier into verification flow")
    print("  2. Update student dashboard with enhanced reports")
    print("  3. Test with real student images")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
