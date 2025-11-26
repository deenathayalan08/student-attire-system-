from pathlib import Path
from typing import Dict, Any
import uuid
import cv2
import pandas as pd
from .config import AppConfig


def ensure_dirs(cfg: AppConfig | None = None) -> None:
	cfg = cfg or AppConfig()
	cfg.data_dir.mkdir(parents=True, exist_ok=True)
	cfg.images_dir.mkdir(parents=True, exist_ok=True)
	cfg.models_dir.mkdir(parents=True, exist_ok=True)


def append_sample_to_dataset(bgr_image, label: str, features: Dict[str, Any], cfg: AppConfig | None = None) -> Path:
	cfg = cfg or AppConfig()
	ensure_dirs(cfg)
	image_id = str(uuid.uuid4())
	image_path = cfg.images_dir / f"{image_id}.jpg"
	cv2.imwrite(str(image_path), bgr_image)

	# Extract only numeric features and maintain consistent column order
	row = {"image": str(image_path), "label": label}
	numeric_features = {}
	for k, v in features.items():
		if isinstance(v, (int, float)):
			numeric_features[k] = float(v)
	
	# Sort keys for consistency across rows
	for k in sorted(numeric_features.keys()):
		row[k] = numeric_features[k]

	# Load existing CSV to get column order
	if cfg.meta_csv.exists():
		existing_df = pd.read_csv(cfg.meta_csv)
		existing_cols = existing_df.columns.tolist()
		# Ensure new row has all existing columns (fill missing with NaN)
		for col in existing_cols:
			if col not in row:
				row[col] = None
		# Create DataFrame with consistent column order
		df = pd.DataFrame([row], columns=existing_cols)
		df.to_csv(cfg.meta_csv, mode="a", header=False, index=False)
	else:
		# First row: use current row's columns
		df = pd.DataFrame([row])
		df.to_csv(cfg.meta_csv, index=False)
	return image_path


def load_dataset(cfg: AppConfig | None = None) -> pd.DataFrame:
	cfg = cfg or AppConfig()
	ensure_dirs(cfg)
	if cfg.meta_csv.exists():
		return pd.read_csv(cfg.meta_csv)
	return pd.DataFrame(columns=["image", "label"])
