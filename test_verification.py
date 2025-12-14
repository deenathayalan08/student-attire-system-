#!/usr/bin/env python3
"""
Test script to verify the enhanced attire verification system
Tests with actual images to ensure realistic results
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath('.'))

from src.config import AppConfig
from src.features import extract_features_from_image, extract_pose
from src.verify import verify_attire_and_safety
from src.model import AttireClassifier

def test_verification_system():
    """Test the enhanced verification system with sample images"""
    
    # Initialize configuration with enhanced settings
    cfg = AppConfig()
    print(f"Configuration loaded:")
    print(f"  - Confidence threshold: {cfg.confidence_threshold}")
    print(f"  - Rule-based checks: {cfg.enable_rules}")
    print(f"  - ID card detection: {cfg.enable_id_card_detection}")
    print(f"  - ID card required: {cfg.id_card_required}")
    print(f"  - Black shoes required (male): {cfg.require_black_shoes_male}")
    print(f"  - Allow any color pants (male): {cfg.allow_any_color_pants_male}")
    print()
    
    # Initialize classifier (may not have trained model)
    clf = AttireClassifier()
    try:
        clf.load(cfg=cfg)
        print("✓ Trained model loaded")
    except:
        print("⚠ No trained model found - using rule-based verification only")
        clf = None
    
    # Test with sample images
    images_dir = Path("data/images")
    if not images_dir.exists():
        print("❌ No test images found in data/images")
        return
    
    # Get first 5 images for testing
    image_files = list(images_dir.glob("*.jpg"))[:5]
    
    print(f"\nTesting with {len(image_files)} sample images:")
    print("=" * 60)
    
    results_summary = []
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n{i}. Testing: {img_path.name}")
        print("-" * 40)
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print("❌ Failed to load image")
            continue
        
        h, w = image.shape[:2]
        print(f"   Image size: {w}x{h} (aspect ratio: {h/w:.2f})")
        
        # Extract pose landmarks
        pose_landmarks = extract_pose(image)
        if pose_landmarks:
            print("   ✓ Pose landmarks detected")
        else:
            print("   ⚠ No pose landmarks detected")
        
        # Extract features
        features = extract_features_from_image(image, pose_landmarks, bins=cfg.hist_bins)
        
        # Verify attire
        result = verify_attire_and_safety(features, cfg, clf)
        
        # Display results
        status = result['status']
        score = result['success_score']
        violations = result['violations']
        
        print(f"   Status: {status}")
        print(f"   Compliance Score: {score:.1%}")
        print(f"   Violations: {violations['total_violations']} total")
        
        if violations['total_violations'] > 0:
            print(f"     - Critical: {violations['critical']}")
            print(f"     - High: {violations['high']}")
            print(f"     - Medium: {violations['medium']}")
            
            # Show first few violations
            for j, violation in enumerate(violations['violations'][:3]):
                severity = violation['severity']
                item = violation['item']
                reason = violation['reason']
                print(f"     {j+1}. [{severity.upper()}] {item}: {reason}")
        
        # Store summary
        results_summary.append({
            'image': img_path.name,
            'status': status,
            'score': score,
            'violations': violations['total_violations']
        })
    
    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS SUMMARY")
    print("=" * 60)
    
    status_counts = {'PASS': 0, 'WARNING': 0, 'FAIL': 0}
    total_score = 0
    
    for result in results_summary:
        status_counts[result['status']] += 1
        total_score += result['score']
        print(f"{result['image']:<40} {result['status']:<8} {result['score']:.1%} ({result['violations']} violations)")
    
    print("-" * 60)
    print(f"PASS: {status_counts['PASS']}, WARNING: {status_counts['WARNING']}, FAIL: {status_counts['FAIL']}")
    print(f"Average compliance score: {total_score/len(results_summary):.1%}")
    
    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if status_counts['PASS'] == len(results_summary):
        print("⚠ WARNING: All images passed verification!")
        print("   This suggests the system may still be too lenient.")
        print("   Consider:")
        print("   - Increasing confidence threshold further")
        print("   - Adding more strict color detection")
        print("   - Enhancing violation penalties")
    elif status_counts['FAIL'] == len(results_summary):
        print("⚠ WARNING: All images failed verification!")
        print("   This suggests the system may be too strict.")
        print("   Consider:")
        print("   - Decreasing confidence threshold")
        print("   - Relaxing some color detection criteria")
        print("   - Reducing violation penalties")
    else:
        print("✓ GOOD: Mixed results detected!")
        print("   The system appears to be working realistically.")
        print(f"   {status_counts['PASS']} passed, {status_counts['WARNING']} warnings, {status_counts['FAIL']} failed")

if __name__ == "__main__":
    test_verification_system()