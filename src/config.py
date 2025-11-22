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

	# Enhanced Attire Verification Features
	enable_gender_specific_checks: bool = True
	enable_haircut_beard_detection: bool = True
	enable_haircut_detection: bool = True
	enable_beard_detection: bool = True
	enable_jewelry_detection: bool = True
	enable_biometric_verification: bool = True
	enable_casual_uniform_policy: bool = True
	enable_pdf_reports: bool = True

	# Casual/Uniform Day Policy
	uniform_days: List[str] = None  # ["wednesday"] - filled in __post_init__
	casual_days: List[str] = None  # ["friday"] - filled in __post_init__
	
	# Admin-editable policy settings
	policy_uniform_day: str = "wednesday"  # Day(s) when uniform is mandatory
	policy_casual_day: str = "friday"      # Day(s) when casual is allowed
	policy_other_days: str = "uniform"     # uniform or casual for other days

	# Gender-specific policy settings
	# Male attire requirements
	require_haircut_check_male: bool = True
	require_beard_check_male: bool = True
	require_tuck_in_male: bool = True
	allow_chains_male: bool = False  # No silver/gold chains allowed

	# Female attire requirements
	require_sandals_check_female: bool = True
	allow_jewelry_female: bool = True

	# Grooming policy settings
	haircut_required: bool = True
	beard_policy: str = "clean_shaven"  # clean_shaven, trimmed, no_policy
	jewelry_policy: str = "no_jewelry"  # no_jewelry, allowed
	haircut_compliance_threshold: float = 0.8
	beard_tolerance_threshold: float = 0.2
	jewelry_tolerance_threshold: float = 0.2

	# Biometric settings
	biometric_registration_required: bool = True
	biometric_verification_threshold: float = 0.8
	biometric_threshold: float = 0.8  # Threshold for verification in verify_attire_and_safety
	biometric_required: bool = True  # Whether biometric verification is mandatory

	# College information
	college_name: str = "Student College"
	college_address: str = "College Address"
	college_phone: str = "College Phone"
	college_email: str = "college@example.com"

	# Report settings
	report_template_path: Path = Path("templates/report_template.html")
	reports_dir: Path = Path("reports")
	
	# Uniform images storage
	uniform_images_dir: Path = Path("data/uniforms")
	male_uniform_image: Path = Path("data/uniforms/male_uniform.jpg")
	female_uniform_image: Path = Path("data/uniforms/female_uniform.jpg")
	male_casual_image: Path = Path("data/uniforms/male_casual.jpg")
	female_casual_image: Path = Path("data/uniforms/female_casual.jpg")

	# Paths
	data_dir: Path = Path("data")
	images_dir: Path = Path("data/images")
	meta_csv: Path = Path("data/metadata.csv")
	models_dir: Path = Path("models")
	model_path: Path = Path("models/attire_clf.joblib")

	# Attire color settings
	male_uniform_color: str = "white"
	male_shoe_color: str = "black"
	female_uniform_color: str = "blue"
	male_pants_color: str = "black"

	# Thresholds
	warning_threshold: float = 0.5

	def __post_init__(self):
		if self.zones is None:
			self.zones = ["Gate", "Classroom", "Lab", "Sports"]
		if self.uniform_days is None:
			self.uniform_days = [self.policy_uniform_day.lower()]
		if self.casual_days is None:
			self.casual_days = [self.policy_casual_day.lower()]
		
		# Ensure directories exist
		self.uniform_images_dir.mkdir(parents=True, exist_ok=True)
		self.reports_dir.mkdir(parents=True, exist_ok=True)
