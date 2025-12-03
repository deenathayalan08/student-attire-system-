"""Test Phase 3 integration with the complete system."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attire_classifier import EnhancedAttireClassifier
from src.dataset_analyzer import analyze_dataset_structure
import cv2
import numpy as np


def test_model_loading():
    """Test if model can be loaded."""
    print("\n1. Testing Model Loading...")
    print("-" * 60)
    
    model_path = Path("models/attire_classifier_enhanced.joblib")
    
    if not model_path.exists():
        print("❌ Model not found. Please train first:")
        print("   python scripts/train_attire_classifier.py")
        return False
    
    try:
        clf = EnhancedAttireClassifier()
        clf.load(model_path)
        print(f"✅ Model loaded successfully")
        print(f"   Features: {len(clf.feature_names)}")
        print(f"   Classes: {clf.classes_present}")
        print(f"   Training stats: {clf.training_stats}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_dataset_structure():
    """Test dataset structure."""
    print("\n2. Testing Dataset Structure...")
    print("-" * 60)
    
    dataset_root = Path("datasets")
    
    if not dataset_root.exists():
        print("❌ Dataset not found at datasets/")
        return False
    
    try:
        stats = analyze_dataset_structure(dataset_root)
        print(f"✅ Dataset found")
        print(f"   Train: {stats['train']['count']} images")
        print(f"   Valid: {stats['valid']['count']} images")
        print(f"   Test: {stats['test']['count']} images")
        print(f"   Total: {stats['total_images']} images")
        return True
    except Exception as e:
        print(f"❌ Error analyzing dataset: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction."""
    print("\n3. Testing Feature Extraction...")
    print("-" * 60)
    
    try:
        from src.features import extract_features_from_image, extract_pose
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Extract features
        pose = extract_pose(dummy_image)
        features = extract_features_from_image(dummy_image, pose, bins=24)
        
        print(f"✅ Feature extraction works")
        print(f"   Total features: {len(features)}")
        print(f"   Pose detected: {'Yes' if pose else 'No'}")
        print(f"   Sample features: {list(features.keys())[:5]}")
        return True
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return False


def test_prediction():
    """Test prediction on sample image."""
    print("\n4. Testing Prediction...")
    print("-" * 60)
    
    model_path = Path("models/attire_classifier_enhanced.joblib")
    
    if not model_path.exists():
        print("⚠️ Model not found, skipping prediction test")
        return True
    
    try:
        clf = EnhancedAttireClassifier()
        clf.load(model_path)
        
        # Create dummy features
        from src.features import extract_features_from_image, extract_pose
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pose = extract_pose(dummy_image)
        features = extract_features_from_image(dummy_image, pose, bins=24)
        
        # Predict
        category, probabilities = clf.predict(features)
        
        print(f"✅ Prediction works")
        print(f"   Predicted category: {category}")
        print(f"   Probabilities:")
        for cat, prob in probabilities.items():
            print(f"      {cat}: {prob:.1%}")
        return True
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dashboard_integration():
    """Test dashboard integration."""
    print("\n5. Testing Dashboard Integration...")
    print("-" * 60)
    
    try:
        from src.ui.student_dashboard import classify_attire_type
        
        # Create dummy features and result
        features = {
            "torso_mean_h": 90.0,
            "torso_mean_s": 50.0,
            "torso_mean_v": 150.0,
            "id_card_detected": 0.8
        }
        result = {
            "status": "PASS",
            "score": 0.85,
            "violations": {"violations": []}
        }
        
        # Test classification
        category, probabilities = classify_attire_type(features, result, "M")
        
        print(f"✅ Dashboard integration works")
        print(f"   Category: {category}")
        print(f"   Probabilities: {probabilities}")
        return True
    except Exception as e:
        print(f"❌ Error testing dashboard: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase 3 Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Dataset Structure", test_dataset_structure),
        ("Feature Extraction", test_feature_extraction),
        ("Prediction", test_prediction),
        ("Dashboard Integration", test_dashboard_integration)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    print("=" * 60)
    
    if passed == total:
        print("\n🎉 All tests passed! Phase 3 is ready to use.")
        print("\nNext steps:")
        print("1. Run: streamlit run app/streamlit_app.py")
        print("2. Navigate to 'Student Verification'")
        print("3. Test the enhanced classification")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
