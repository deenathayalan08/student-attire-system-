"""Enhanced attire classifier with Formal/Semi-Formal/Casual detection."""
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib

from .features import extract_pose, extract_features_from_image
from .dataset_analyzer import load_dataset_with_labels, get_category_distribution


class EnhancedAttireClassifier:
    """Enhanced classifier for Formal/Semi-Formal/Casual attire detection."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = {"Formal": 0, "Semi-Formal": 1, "Casual": 2}
        self.label_decoder = {0: "Formal", 1: "Semi-Formal", 2: "Casual"}
        self.feature_names = None
        self.training_stats = {}
        self.classes_present = []  # Track which classes are actually in the dataset
    
    def extract_features_from_path(self, image_path: Path, bins: int = 24) -> Dict[str, float]:
        """Extract features from image file."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        pose = extract_pose(img)
        features = extract_features_from_image(img, pose, bins=bins)
        return features
    
    def prepare_dataset(
        self,
        dataset_root: Path,
        split: str = "train",
        max_samples: int = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare dataset for training.
        
        Returns:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
        """
        # Load dataset with labels
        dataset = load_dataset_with_labels(dataset_root, split)
        
        if max_samples:
            dataset = dataset[:max_samples]
        
        print(f"Loading {len(dataset)} images from {split} split...")
        
        X_list = []
        y_list = []
        feature_names = None
        
        for i, (img_path, original_label, attire_category) in enumerate(dataset):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(dataset)} images...")
            
            try:
                features = self.extract_features_from_path(img_path)
                
                # Store feature names from first sample
                if feature_names is None:
                    feature_names = sorted(features.keys())
                
                # Extract feature values in consistent order
                feature_vector = [features.get(name, 0.0) for name in feature_names]
                X_list.append(feature_vector)
                
                # Encode label
                y_list.append(self.label_encoder[attire_category])
                
            except (IOError, cv2.error, ValueError, KeyError) as e:
                import logging
                logging.getLogger(__name__).error(f"Error processing {img_path}: {e}")
                continue
        
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        
        print(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Category distribution: {get_category_distribution(dataset)}")
        
        self.feature_names = feature_names
        return X, y, feature_names
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, Any]:
        """Train the classifier."""
        print("\nTraining Enhanced Attire Classifier...")
        
        # Create pipeline with scaling and classifier
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            ))
        ])
        
        # Cross-validation
        print("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv, scoring='accuracy')
        
        print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        # Train on full training set
        print("Training on full training set...")
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        stats = {
            "cv_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "train_accuracy": float(train_acc),
            "n_samples": int(len(y_train)),
            "n_features": int(X_train.shape[1])
        }
        
        # Determine which classes are present
        unique_classes = np.unique(y_train)
        self.classes_present = [self.label_decoder[c] for c in unique_classes]
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = np.mean(val_pred == y_val)
            stats["val_accuracy"] = float(val_acc)
            
            print(f"\nValidation Accuracy: {val_acc:.3f}")
            print("\nClassification Report (Validation):")
            print(classification_report(
                y_val, val_pred,
                target_names=self.classes_present,
                labels=unique_classes
            ))
            
            print("\nConfusion Matrix (Validation):")
            cm = confusion_matrix(y_val, val_pred, labels=unique_classes)
            print(cm)
            stats["confusion_matrix"] = cm.tolist()
        
        self.training_stats = stats
        return stats
    
    def predict(self, features: Dict[str, float]) -> Tuple[str, Dict[str, float]]:
        """
        Predict attire category from features.
        
        Returns:
            category: Predicted category (Formal/Semi-Formal/Casual)
            probabilities: Dict of category -> probability
        """
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        # Extract feature vector in consistent order
        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        X = np.array([feature_vector], dtype=np.float32)
        
        # Predict
        pred_class = self.model.predict(X)[0]
        pred_proba = self.model.predict_proba(X)[0]
        
        category = self.label_decoder[pred_class]
        
        # Build probabilities dict - handle case where not all classes are present
        probabilities = {"Formal": 0.0, "Semi-Formal": 0.0, "Casual": 0.0}
        for i, class_name in enumerate(self.classes_present):
            probabilities[class_name] = float(pred_proba[i])
        
        return category, probabilities
    
    def predict_from_image(self, image_path: Path) -> Tuple[str, Dict[str, float]]:
        """Predict attire category from image file."""
        features = self.extract_features_from_path(image_path)
        return self.predict(features)
    
    def save(self, path: Path):
        """Save trained model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "training_stats": self.training_stats,
            "classes_present": self.classes_present
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load trained model."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.label_encoder = data.get("label_encoder", self.label_encoder)
        self.label_decoder = data.get("label_decoder", self.label_decoder)
        self.training_stats = data.get("training_stats", {})
        self.classes_present = data.get("classes_present", ["Formal", "Semi-Formal", "Casual"])
        print(f"Model loaded from {path}")
