from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class AppConfig:
	# Expected attire hints (keywords used in simple color heuristics)
	expected_top: str = "white"
	expected_bottom: str = "dark"

	# Model and features
	hist_bins: int = 24
	confidence_threshold: float = 0.7
	enable_rules: bool = True
	enable_model: bool = True
	max_video_fps: int = 10

	# Dataset labeling
	current_label: str = "compliant"

	# ID Card detection
	enable_id_card_detection: bool = True
	id_card_required: bool = True
	id_card_confidence_threshold: float = 0.6

	# Policy profiles
	policy_profile: str = "regular"  # regular, sports, lab
	zones: List[str] = None  # filled in __post_init__
	# Uniform policy extensions
	policy_gender: str = "male"  # male or female
	require_shirt_for_male: bool = True
	require_kurti_dupatta_for_female: bool = True
	require_footwear_male: bool = True
	require_footwear_female: bool = False  # sandals or shoes acceptable; not mandatory
	require_black_shoes_male: bool = True  # Require black shoes for males
	allow_any_color_pants_male: bool = True  # Allow any color pants for males (instead of only dark)
	
	# Security & Safety Features
	enable_unauthorized_entry_alerts: bool = True
	enable_late_entry_tracking: bool = True
	enable_early_exit_tracking: bool = True
	enable_emergency_alerts: bool = True
	enable_geofencing: bool = False  # Optional for IoT/Smart Campus
	normal_entry_start_time: str = "08:00"
	normal_entry_end_time: str = "09:00"
	normal_exit_start_time: str = "15:00"
	normal_exit_end_time: str = "16:00"
	campus_latitude: float = 0.0  # GPS coordinates for geofencing
	campus_longitude: float = 0.0
	campus_radius_meters: float = 500.0  # Radius in meters

	# Paths
	data_dir: Path = Path("data")
	images_dir: Path = Path("data/images")
	meta_csv: Path = Path("data/metadata.csv")
	models_dir: Path = Path("models")
	model_path: Path = Path("models/attire_clf.joblib")

	def __post_init__(self):
		if self.zones is None:
			self.zones = ["Gate", "Classroom", "Lab", "Sports"]
