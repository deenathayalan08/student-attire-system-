"""Analyze dataset structure and statistics."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset_analyzer import (
    analyze_dataset_structure,
    load_dataset_with_labels,
    get_category_distribution
)


def main():
    """Analyze the dataset."""
    print("=" * 60)
    print("Dataset Analysis")
    print("=" * 60)
    
    dataset_root = Path("datasets")
    
    # Overall structure
    print("\n1. Dataset Structure:")
    print("-" * 60)
    stats = analyze_dataset_structure(dataset_root)
    print(f"   Train images: {stats['train']['count']}")
    print(f"   Valid images: {stats['valid']['count']}")
    print(f"   Test images: {stats['test']['count']}")
    print(f"   Total images: {stats['total_images']}")
    
    # Analyze each split
    for split in ["train", "valid", "test"]:
        print(f"\n2. {split.capitalize()} Split Analysis:")
        print("-" * 60)
        
        dataset = load_dataset_with_labels(dataset_root, split)
        print(f"   Loaded {len(dataset)} images with labels")
        
        if dataset:
            # Category distribution
            distribution = get_category_distribution(dataset)
            print(f"\n   Attire Category Distribution:")
            for category, count in sorted(distribution.items()):
                percentage = (count / len(dataset)) * 100
                print(f"      {category:15s}: {count:4d} ({percentage:5.1f}%)")
            
            # Original label distribution
            original_labels = {}
            for _, orig_label, _ in dataset:
                original_labels[orig_label] = original_labels.get(orig_label, 0) + 1
            
            print(f"\n   Original Label Distribution:")
            for label, count in sorted(original_labels.items()):
                percentage = (count / len(dataset)) * 100
                print(f"      {label:25s}: {count:4d} ({percentage:5.1f}%)")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
