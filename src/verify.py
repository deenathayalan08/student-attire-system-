from typing import Dict, Any, List
import numpy as np
from datetime import datetime

from .config import AppConfig
from .model import AttireClassifier
from .biometric import get_biometric_system


def _keyword_score_from_hue(mean_h: float, keyword: str) -> float:
	# Very rough mapping from hue to color keyword
	# hue in [0,180]: 0 red, 30 yellow, 60 green, 90 cyan, 120 blue, 150 magenta
	kw = (keyword or "").lower()
	if kw in ("white", "light"):
		return 0.7
	if kw in ("dark", "black"):
		return 0.7
	if kw == "blue":
		return float(np.exp(-((mean_h - 120.0) ** 2) / (2 * 12.0 ** 2)))
	if kw == "green":
