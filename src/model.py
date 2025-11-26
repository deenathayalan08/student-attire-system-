from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .config import AppConfig


@dataclass
class AttireClassifier:
	model: Pipeline | None = None
	label_names: List[str] | None = None
	feature_names: List[str] | None = None  # Persist feature column order

	def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] | None = None) -> Dict[str, Any]:
		# Store feature names for consistent inference
		if feature_names is not None:
			self.feature_names = feature_names
		elif self.feature_names is None:
			# Fallback: auto-generate if not provided
			self.feature_names = [f"feat_{i}" for i in range(X.shape[1])]
		
		from sklearn.model_selection import StratifiedKFold
		
		# Check class balance
		n_classes = len(np.unique(y))
		if n_classes < 2:
			raise ValueError(f"Dataset must have at least 2 classes; found {n_classes}")
		
		self.model = Pipeline([
			("scaler", StandardScaler(with_mean=False)),
			("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
		])
		
		# Determine adaptive CV folds based on class distribution
		n_samples = len(y)
		max_folds = min(3, n_samples // max(1, n_classes))
		cv_folds = max(2, max_folds)
		
		# Check for extreme class imbalance
		from collections import Counter
		class_counts = Counter(y)
		min_class_count = min(class_counts.values())
		
		# If minority class is very small, reduce cv_folds further
		if min_class_count < cv_folds:
			cv_folds = max(2, min_class_count)
			print(f"Detected extreme class imbalance: min class count {min_class_count}; reducing cv_folds to {cv_folds}")
		
		try:
			# Use StratifiedKFold to maintain class proportions in each fold
			skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
			cv = cross_val_score(self.model, X, y, cv=skf)
		except Exception as e:
			# Fallback to simple k-fold with reduced folds
			print(f"StratifiedKFold failed: {e}; using regular k-fold with {cv_folds} folds")
			cv = cross_val_score(self.model, X, y, cv=min(cv_folds, 2))
		
		# If still NaN (all values are nan), replace with mean of non-nan or 0.5
		cv = np.array(cv)
		if np.all(np.isnan(cv)):
			print(f"Warning: all CV scores are NaN due to extreme class imbalance; setting to 0.5")
			cv = np.array([0.5])
		# Use nanmean to skip any NaN values
		mean_cv = float(np.nanmean(cv))
		if np.isnan(mean_cv):
			print(f"Warning: CV accuracy is NaN due to extreme class imbalance; setting to 0.5")
			mean_cv = 0.5
		
		self.model.fit(X, y)
		self.label_names = sorted(list({str(v) for v in y}))
		return {
			"cv_accuracy": mean_cv,
			"num_samples": int(len(y)),
			"num_classes": int(n_classes),
			"cv_folds": len(cv)
		}

	def predict_proba(self, X: np.ndarray) -> np.ndarray:
		if self.model is None:
			raise RuntimeError("Model not trained")
		return self.model.predict_proba(X)

	def predict(self, X: np.ndarray) -> np.ndarray:
		if self.model is None:
			raise RuntimeError("Model not trained")
		return self.model.predict(X)

	def save(self, path: Path | str | None = None, cfg: AppConfig | None = None) -> None:
		if self.model is None:
			return
		cfg = cfg or AppConfig()
		p = Path(path) if path is not None else cfg.model_path
		p.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump({
			"model": self.model,
			"labels": self.label_names,
			"feature_names": self.feature_names
		}, p)

	def load(self, path: Path | str | None = None, cfg: AppConfig | None = None) -> None:
		cfg = cfg or AppConfig()
		p = Path(path) if path is not None else cfg.model_path
		obj = joblib.load(p)
		self.model = obj.get("model")
		self.label_names = obj.get("labels")
		self.feature_names = obj.get("feature_names")  # Load feature names

	def _features_from_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
		"""Extract features from DataFrame and return both array and column names."""
		feature_cols = [c for c in df.columns if c not in ("image", "label")]
		self.feature_names = feature_cols  # Store for consistency
		return df[feature_cols].fillna(0).astype(float).to_numpy(), feature_cols

	def train_from_dataframe(self, df: pd.DataFrame, label_col: str = "label", bins: int = 24) -> Dict[str, Any]:
		if df is None or df.empty:
			raise ValueError("Dataset is empty")
		X, feature_cols = self._features_from_df(df)
		y = df[label_col].astype(str).to_numpy()
		return self.fit(X, y, feature_names=feature_cols)
