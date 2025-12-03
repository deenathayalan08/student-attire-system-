"""Dataset analysis utilities for attire classification."""
from pathlib import Path
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np
from collections import Counter


def analyze_dataset_structure(dataset_root: Path) -> Dict[str, Any]:
    """Analyze the dataset structure and return statistics."""
    stats = {
        "train": {"count": 0, "categories": {}},
        "valid": {"count": 0, "categories": {}},
        "test": {"count": 0, "categories": {}},
        "total_images": 0,
        "categories": set()
    }
    
    for split in ["train", "valid", "test"]:
        split_path = dataset_root / split / "images"
        if split_path.exists():
            images = list(split_path.glob("*.jpg")) + list(split_path.glob("*.png"))
            stats[split]["count"] = len(images)
            stats["total_images"] += len(images)
            
            # Try to extract categories from filenames or labels
            labels_path = dataset_root / split / "labels"
            if labels_path.exists():
                labels = list(labels_path.glob("*.txt"))
                stats[split]["labels_count"] = len(labels)
    
    return stats


def map_to_attire_category(class_name: str, gender: str = None) -> str:
    """Map dataset class names to Formal/Semi-Formal/Casual categories."""
    class_name = class_name.lower()
    
    # Men's attire mapping
    if "men" in class_name or "man" in class_name:
        if "pant-shirt" in class_name or "shirt" in class_name:
            # Men in pant-shirt: Formal (proper shirt + pants)
            return "Formal"
        elif "shalwar" in class_name or "kameez" in class_name:
            # Traditional wear: Semi-Formal
            return "Semi-Formal"
        else:
            return "Casual"
    
    # Women's attire mapping
    elif "women" in class_name:
        if "salwar-kameez" in class_name or "kameez" in class_name:
            # Traditional formal wear: Formal
            return "Formal"
        elif "pant-shirt" in class_name:
            # Western wear: Semi-Formal
            return "Semi-Formal"
        else:
            return "Casual"
    
    # Default
    return "Casual"


def load_dataset_with_labels(
    dataset_root: Path,
    split: str = "train",
    category_mapping: Dict[int, str] = None
) -> List[Tuple[Path, str, str]]:
    """
    Load dataset images with their labels and attire categories.
    
    Returns:
        List of (image_path, original_label, attire_category) tuples
    """
    if category_mapping is None:
        # Default mapping from data.yaml
        category_mapping = {
            0: "Women-Pant-Shirt",
            1: "Women-Salwar-Kameez",
            2: "man-shalwar-kameez",
            3: "men-pant-shirt"
        }
    
    images_path = dataset_root / split / "images"
    labels_path = dataset_root / split / "labels"
    
    dataset = []
    
    if not images_path.exists():
        return dataset
    
    for img_file in images_path.glob("*.jpg"):
        # Find corresponding label file
        label_file = labels_path / f"{img_file.stem}.txt"
        
        if label_file.exists():
            # Read YOLO format label (class_id x y w h)
            with open(label_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Take first line (primary class)
                    class_id = int(lines[0].split()[0])
                    original_label = category_mapping.get(class_id, "unknown")
                    attire_category = map_to_attire_category(original_label)
                    dataset.append((img_file, original_label, attire_category))
    
    return dataset


def get_category_distribution(dataset: List[Tuple[Path, str, str]]) -> Dict[str, int]:
    """Get distribution of attire categories in dataset."""
    categories = [item[2] for item in dataset]
    return dict(Counter(categories))


def validate_image_quality(image_path: Path, min_size: Tuple[int, int] = (100, 100)) -> bool:
    """Validate if image meets quality requirements."""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False
        
        h, w = img.shape[:2]
        if h < min_size[0] or w < min_size[1]:
            return False
        
        return True
    except (IOError, cv2.error):
        return False
