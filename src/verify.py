from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

from .config import AppConfig
from .model import AttireClassifier
from .biometric import get_biometric_system
from .features import extract_features_from_image, extract_pose
from .id_card_detector import detect_id_card
from .db import get_student, insert_event
from .security import check_and_alert_unauthorized_student, check_and_log_entry_time, check_and_alert_emergency_violations


def _keyword_score_from_hue(mean_h: float, keyword: str) -> float:
	"""Very rough mapping from hue to color keyword"""
	# hue in [0,180]: 0 red, 30 yellow, 60 green, 90 cyan, 120 blue, 150 magenta
	kw = (keyword or "").lower()
	if kw in ("white", "light"):
		return 0.7
	if kw in ("dark", "black"):
		return 0.7
	if kw == "blue":
		return float(np.exp(-((mean_h - 120.0) ** 2) / (2 * 12.0 ** 2)))
	if kw == "green":
		return float(np.exp(-((mean_h - 60.0) ** 2) / (2 * 15.0 ** 2)))
	if kw == "red":
		return float(np.exp(-((mean_h - 0.0) ** 2) / (2 * 15.0 ** 2)))
	if kw == "yellow":
		return float(np.exp(-((mean_h - 30.0) ** 2) / (2 * 15.0 ** 2)))
	if kw == "purple":
		return float(np.exp(-((mean_h - 150.0) ** 2) / (2 * 15.0 ** 2)))
	return 0.0


def _get_current_day_policy(cfg: AppConfig) -> str:
	"""Get current day policy (uniform/casual) based on day-of-week settings"""
	from datetime import datetime
	
	current_day = datetime.now().strftime("%A").lower()
	
	# Check if it's a casual day
	if current_day in [d.lower() for d in cfg.casual_days]:
		return "casual"
	
	# Check if it's a uniform day
	if current_day in [d.lower() for d in cfg.uniform_days]:
		return "uniform"
	
	# Default policy for other days
	return cfg.policy_other_days


def _check_gender_specific_attire(features: Dict[str, Any], gender: str, cfg: AppConfig, day_policy: str = "uniform") -> Dict[str, Any]:
	"""Check gender-specific attire requirements based on day policy"""
	violations = []
	score = 1.0

	if gender.lower() == "male":
		# MEN'S ATTIRE CHECKS
		
		# 1. BEARD CHECK
		if cfg.enable_beard_detection and cfg.beard_policy != "no_policy":
			beard_presence = features.get("beard_presence_score", 0.0)
			if cfg.beard_policy == "clean_shaven":
				if beard_presence > cfg.beard_tolerance_threshold:
					violations.append({
						"item": "Beard",
						"detected": f"Beard detected (score: {beard_presence:.1%})",
						"severity": "high",
						"score": beard_presence,
						"required": "Clean shaven",
						"reason": "Policy requires clean-shaven face"
					})
					score *= 0.65
			elif cfg.beard_policy == "trimmed":
				if beard_presence > 0.5:
					violations.append({
						"item": "Beard",
						"detected": f"Beard too long (score: {beard_presence:.1%})",
						"severity": "medium",
						"score": beard_presence,
						"required": "Well-trimmed beard",
						"reason": "Beard should be well-trimmed"
					})
					score *= 0.75

		# 2. HAIRCUT CHECK
		if cfg.enable_haircut_detection and cfg.haircut_required:
			haircut_compliance = features.get("haircut_compliance_score", 1.0)
			if haircut_compliance < cfg.haircut_compliance_threshold:
				violations.append({
					"item": "Haircut",
					"detected": f"Hair not well-groomed (score: {haircut_compliance:.1%})",
					"severity": "medium",
					"score": 1.0 - haircut_compliance,
					"required": "Well-groomed haircut",
					"reason": "Hair should be well-groomed and neat"
				})
				score *= 0.8

		# 3. SHIRT/DRESS CHECK
		if day_policy.lower() == "uniform" and cfg.require_shirt_for_male:
			dress_color_score = _keyword_score_from_hue(features.get("mean_hue_torso", 0), cfg.male_uniform_color)
			if dress_color_score < 0.5:
				violations.append({
					"item": "Uniform Shirt",
					"detected": f"Color mismatch or not wearing uniform",
					"severity": "high",
					"score": 1.0 - dress_color_score,
					"required": f"{cfg.male_uniform_color.title()} formal shirt",
					"reason": f"Uniform day requires {cfg.male_uniform_color} formal shirt"
				})
				score *= 0.6
		elif day_policy.lower() == "casual":
			# Casual: any shirt is acceptable
			pass

		# 4. PANTS/TROUSERS CHECK
		if day_policy.lower() == "uniform":
			pants_color_score = _keyword_score_from_hue(features.get("mean_hue_legs", 0), cfg.male_shoe_color)
			if not cfg.allow_any_color_pants_male and pants_color_score < 0.5:
				violations.append({
					"item": "Trousers",
					"detected": f"Color mismatch (expected dark/black)",
					"severity": "high",
					"score": 1.0 - pants_color_score,
					"required": "Dark/Black trousers",
					"reason": "Uniform day requires dark/black formal trousers"
				})
				score *= 0.65

		# 5. SHOES CHECK
		if cfg.require_footwear_male:
			shoe_color_score = _keyword_score_from_hue(features.get("mean_hue_feet", 0), cfg.male_shoe_color)
			if day_policy.lower() == "uniform" and cfg.require_black_shoes_male:
				if shoe_color_score < 0.5:
					violations.append({
						"item": "Shoes",
						"detected": f"Color mismatch or missing shoes",
						"severity": "high",
						"score": 1.0 - shoe_color_score,
						"required": "Black formal shoes",
						"reason": "Uniform day requires black formal shoes"
					})
					score *= 0.6
			elif day_policy.lower() == "casual":
				# Casual: any shoes or sandals acceptable
				pass

		# 6. SHIRT TUCK-IN CHECK (for uniform day)
		if day_policy.lower() == "uniform" and cfg.require_tuck_in_male:
			tuck_in_score = features.get("tuck_in_score", 0.5)
			if tuck_in_score < 0.6:
				violations.append({
					"item": "Shirt Tuck-in",
					"detected": "Shirt not properly tucked in",
					"severity": "medium",
					"score": 1.0 - tuck_in_score,
					"required": "Shirt should be tucked in",
					"reason": "Uniform day requires shirt to be tucked into trousers"
				})
				score *= 0.8

		# 7. CHAIN DETECTION (Men should not wear chains)
		if cfg.enable_jewelry_detection and not cfg.allow_chains_male:
			chain_score = features.get("neck_jewelry_presence_score", 0.0)
			if chain_score > cfg.jewelry_tolerance_threshold:
				violations.append({
					"item": "Neck Chain/Jewelry",
					"detected": f"Jewelry detected (score: {chain_score:.1%})",
					"severity": "medium",
					"score": chain_score,
					"required": "No visible chains or jewelry",
					"reason": "Policy does not allow visible chains or neck jewelry"
				})
				score *= 0.85

	elif gender.lower() == "female":
		# WOMEN'S ATTIRE CHECKS
		
		# 1. DRESS/KURTI CHECK
		if day_policy.lower() == "uniform":
			dress_color_score = _keyword_score_from_hue(features.get("mean_hue_torso", 0), cfg.female_uniform_color)
			if dress_color_score < 0.5:
				violations.append({
					"item": "Uniform Dress/Kurti",
					"detected": f"Color mismatch or inappropriate dress",
					"severity": "high",
					"score": 1.0 - dress_color_score,
					"required": f"{cfg.female_uniform_color.title()} formal dress/kurti",
					"reason": f"Uniform day requires {cfg.female_uniform_color} formal dress or kurti"
				})
				score *= 0.6
		elif day_policy.lower() == "casual":
			# Casual: any dress acceptable
			pass

		# 2. FOOTWEAR CHECK (Sandals/Shoes)
		if cfg.require_footwear_female:
			feet_visible = features.get("feet_mask_area", 0.0) > 0.1
			if not feet_visible:
				violations.append({
					"item": "Footwear",
					"detected": "Feet not visible or no footwear detected",
					"severity": "medium",
					"score": 0.5,
					"required": "Visible sandals or shoes",
					"reason": "Footwear (sandals or shoes) should be visible"
				})
				score *= 0.8

		# 4. JEWELRY CHECK (Women - excessive jewelry check)
		if cfg.enable_jewelry_detection and not cfg.allow_jewelry_female:
			jewelry_score = features.get("neck_jewelry_presence_score", 0.0)
			if jewelry_score > cfg.jewelry_tolerance_threshold:
				violations.append({
					"item": "Excessive Jewelry",
					"detected": f"Jewelry detected (score: {jewelry_score:.1%})",
					"severity": "low",
					"score": jewelry_score,
					"required": "Minimal/modest jewelry",
					"reason": "Excessive jewelry may not be allowed"
				})
				score *= 0.9
		elif cfg.allow_jewelry_female:
			# Jewelry is allowed for women, just note if excessive
			jewelry_score = features.get("neck_jewelry_presence_score", 0.0)
			if jewelry_score > 0.7:
				violations.append({
					"item": "Jewelry",
					"detected": f"Excessive jewelry (score: {jewelry_score:.1%})",
					"severity": "low",
					"score": 0.0,  # Not a violation, just a warning
					"required": "Excessive jewelry warning",
					"reason": "Consider limiting jewelry amount for better appearance"
				})
				score *= 0.95

	# ID CARD CHECK (Critical for both genders)
	id_card_detected = features.get("id_card_detected", False)
	id_card_confidence = features.get("id_card_confidence", 0.0)
	
	if cfg.id_card_required and not id_card_detected:
		violations.append({
			"item": "ID Card",
			"detected": "No ID card detected",
			"severity": "critical",
			"score": 0.0,
			"required": "Visible student ID card",
			"reason": "Student ID card is mandatory and must be visible"
		})
		score *= 0.2
	elif id_card_detected and id_card_confidence < cfg.id_card_confidence_threshold:
		violations.append({
			"item": "ID Card",
			"detected": f"ID card detected with low confidence ({id_card_confidence:.1%})",
			"severity": "high",
			"score": id_card_confidence,
			"required": "Clear, visible student ID card",
			"reason": f"ID card confidence below threshold ({cfg.id_card_confidence_threshold:.0%})"
		})
		score *= 0.5

	return {
		"score": score,
		"violations": violations,
		"gender": gender
	}


def verify_attire_and_safety(image: np.ndarray, student_id: str, zone: str = "Main Gate",
						   cfg: AppConfig = None, classifier: AttireClassifier = None) -> Dict[str, Any]:
	"""
	Main function to verify student attire and safety compliance
	This function performs full computer vision analysis after biometric verification

	Args:
		image: Student image (BGR format)
		student_id: Verified student ID from phone biometric
		zone: Verification zone/location
		cfg: Application configuration
		classifier: Trained attire classifier

	Returns:
		Dictionary with comprehensive verification results
	"""
	if cfg is None:
		cfg = AppConfig()

	# Get student information
	student_info = get_student(student_id, cfg)
	if not student_info:
		return {
			"status": "ERROR",
			"error": f"Student {student_id} not found in database",
			"student_id": student_id,
			"zone": zone
		}

	# Extract features from image
	pose = extract_pose(image)
	features = extract_features_from_image(image, pose_landmarks=pose, bins=cfg.hist_bins)

	# Detect ID card
	id_card_result = detect_id_card(image)
	features["id_card_detected"] = id_card_result["detected"]
	features["id_card_confidence"] = id_card_result["confidence"]

	# Determine gender from student info or features
	gender = student_info.get("gender", "unknown")
	if gender == "unknown":
		# Try to detect from features (rough approximation)
		gender = "male" if features.get("shoulder_width_ratio", 1.0) > 1.05 else "female"

	# Get current day policy
	day_policy = _get_current_day_policy(cfg)

	# Check gender-specific attire
	attire_check = _check_gender_specific_attire(features, gender, cfg, day_policy)

	# ML-based classification (if classifier available)
	ml_prediction = None
	if classifier:
		try:
			ml_prediction = classifier.predict_proba(features)
			ml_label = classifier.predict(features)[0]
		except Exception as e:
			print(f"ML prediction failed: {e}")
			ml_prediction = None

	# Calculate overall scores
	rule_based_score = attire_check["score"]
	ml_score = ml_prediction[0][1] if ml_prediction is not None else rule_based_score

	# Weighted combination
	overall_score = (rule_based_score * 0.7) + (ml_score * 0.3)
	success_score = overall_score
	fail_score = 1.0 - overall_score

	# Determine status
	if overall_score >= cfg.compliance_threshold:
		status = "PASS"
	elif overall_score >= cfg.warning_threshold:
		status = "WARNING"
	else:
		status = "FAIL"

	# Security checks
	security_checks = {}

	# Check for unauthorized access
	auth_result = check_and_alert_unauthorized_student(student_id, zone, cfg)
	security_checks["authorized"] = auth_result

	# Check entry time
	time_result = check_and_log_entry_time(student_id, zone, cfg)
	security_checks["time_check"] = time_result

	# Emergency violations check
	emergency_result = check_and_alert_emergency_violations(student_id, zone, attire_check["violations"], cfg)
	security_checks["emergency_check"] = emergency_result

	# Prepare violations summary
	all_violations = attire_check["violations"]
	total_violations = len(all_violations)
	critical_violations = len([v for v in all_violations if v["severity"] == "critical"])
	high_violations = len([v for v in all_violations if v["severity"] == "high"])
	medium_violations = len([v for v in all_violations if v["severity"] == "medium"])

	# Log event to database
	event_id = insert_event({
		"student_id": student_id,
		"zone": zone,
		"status": status,
		"score": float(overall_score),
		"label": ml_label if ml_prediction is not None else status,
		"details": f"Gender: {gender}, Policy: {day_policy}, Violations: {total_violations}",
		"image_path": None  # Could save image if needed
	}, cfg)

	# Prepare comprehensive result
	result = {
		"status": status,
		"student_id": student_id,
		"student_name": student_info.get("name", "Unknown"),
		"zone": zone,
		"timestamp": datetime.now().isoformat(),

		# Scores
		"success_score": float(success_score),
		"fail_score": float(fail_score),
		"overall_score": float(overall_score),
		"rule_based_score": float(rule_based_score),
		"ml_score": float(ml_score) if ml_prediction is not None else None,

		# Student info
		"gender": gender,
		"class": student_info.get("class", "Unknown"),
		"department": student_info.get("department", "Unknown"),

		# Policy info
		"day_policy": day_policy,
		"current_day": datetime.now().strftime("%A"),

		# Violations
		"violations": {
			"total_violations": total_violations,
			"critical": critical_violations,
			"high": high_violations,
			"medium": medium_violations,
			"details": all_violations
		},

		# Security
		"security_checks": security_checks,

		# Technical details
		"face_verified": True,
		"id_card_detected": id_card_result["detected"],
		"id_card_confidence": id_card_result["confidence"],
		"pose_detected": pose is not None,

		# Event logging
		"event_id": event_id,

		# Recommendations
		"recommendations": _generate_recommendations(all_violations, status, day_policy, gender)
	}

	return result


def _generate_recommendations(violations: List[Dict[str, Any]], status: str, day_policy: str, gender: str = None) -> List[str]:
	"""Generate gender-specific recommendations based on violations and status"""
	recommendations = []

	if status == "FAIL":
		recommendations.append("⚠️ CRITICAL: Immediate attention required - multiple policy violations detected")
	elif status == "WARNING":
		recommendations.append("⚠️ Minor adjustments needed to meet full compliance")
	elif status == "PASS":
		recommendations.append("✓ All attire requirements met - excellent compliance")
		return recommendations

	# Process violations with gender-specific guidance
	for violation in violations:
		item = violation["item"]
		severity = violation.get("severity", "medium")
		required = violation.get("required", "")
		
		if item == "ID Card":
			recommendations.append(f"[{severity.upper()}] Ensure student ID card is clearly visible and properly displayed on chest area")
		
		elif item == "Uniform Shirt" and gender == "male":
			recommendations.append(f"[{severity.upper()}] For uniform day: Wear prescribed {required.lower()}")
		
		elif item == "Uniform Dress/Kurti" and gender == "female":
			recommendations.append(f"[{severity.upper()}] For uniform day: Wear prescribed {required.lower()}")
		
		elif item == "Trousers":
			recommendations.append(f"[{severity.upper()}] For uniform day: Wear dark/black formal trousers (not casual)")
		
		elif item == "Shoes":
			if gender == "male":
				recommendations.append(f"[{severity.upper()}] For uniform day: Wear black formal shoes (not casual/sports shoes)")
			else:
				recommendations.append(f"[{severity.upper()}] Ensure appropriate formal footwear")
		
		elif item == "Beard":
			recommendations.append("[HIGH] Policy requires clean-shaven face - arrange grooming before next entry")
		
		elif item == "Haircut":
			recommendations.append("[MEDIUM] Hairstyle does not meet grooming standards - arrange haircut before next entry")
		
		elif item == "Shirt Tuck-in":
			recommendations.append("[MEDIUM] For uniform day: Shirt must be tucked into trousers throughout the day")
		
		elif item == "Neck Chain/Jewelry" and gender == "male":
			recommendations.append("[MEDIUM] Remove visible chains/jewelry - they are not permitted")
		
		elif item == "Excessive Jewelry" and gender == "female":
			recommendations.append("[LOW] Consider reducing jewelry amount for better professional appearance")
		
		elif item == "Jewelry" and gender == "female":
			recommendations.append("[INFO] Consider limiting excessive jewelry for cleaner appearance")
		
		elif item == "Footwear" and gender == "female":
			recommendations.append("[MEDIUM] Ensure sandals or shoes are visible and appropriate")

	# Add day-specific guidance
	if day_policy == "uniform":
		if gender == "male":
			recommendations.append("\n[UNIFORM DAY CHECKLIST - MEN]\n  ✓ Clean shaven\n  ✓ Well-groomed hair\n  ✓ Formal uniform shirt (color as prescribed)\n  ✓ Black formal trousers\n  ✓ Black formal shoes\n  ✓ Shirt tucked in\n  ✓ Student ID card visible\n  ✓ No chains or excessive jewelry")
		elif gender == "female":
			recommendations.append("\n[UNIFORM DAY CHECKLIST - WOMEN]\n  ✓ Formal dress/kurti (color as prescribed)\n  ✓ Appropriate footwear (sandals/shoes)\n  ✓ Student ID card visible\n  ✓ Modest jewelry\n  ✓ Well-groomed appearance")
	elif day_policy == "casual":
		recommendations.append(f"\n[CASUAL DAY] More relaxed dress code applies - jeans, casual shirts acceptable")

	if not recommendations:
		recommendations.append("All attire requirements met - excellent compliance")

	return recommendations
