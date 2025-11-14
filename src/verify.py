from typing import Dict, Any, List
import numpy as np

from .config import AppConfig
from .model import AttireClassifier


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
		return float(np.exp(-((mean_h - 60.0) ** 2) / (2 * 12.0 ** 2)))
	if kw in ("yellow", "hi-vis", "high-visibility"):
		return float(np.exp(-((mean_h - 30.0) ** 2) / (2 * 12.0 ** 2)))
	return 0.5


def rule_based_checks(features: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
	# Torso expected color
	torso_h = float(features.get("torso_mean_h", 90.0))
	top_score = _keyword_score_from_hue(torso_h, cfg.expected_top)
	# Legs expected darker value
	legs_v = float(features.get("legs_mean_v", 60.0))
	bottom_score = 1.0 - (legs_v / 255.0) if (cfg.expected_bottom or "").lower() in ("dark", "black") else (legs_v / 255.0)
	bottom_score = float(np.clip(bottom_score, 0.0, 1.0))

	# Shoes vs barefoot heuristic: feet brightness low implies shoes
	feet_v = float(features.get("feet_mean_v", 60.0))
	shoes_score = 1.0 - (feet_v / 255.0)
	shoes_score = float(np.clip(shoes_score, 0.0, 1.0))

	# Aggregate
	rule_score = float(np.clip(0.4 * top_score + 0.4 * bottom_score + 0.2 * shoes_score, 0.0, 1.0))
	return {
		"top_score": top_score,
		"bottom_score": bottom_score,
		"shoes_score": shoes_score,
		"rule_score": rule_score,
	}


def _infer_missing_items(features: Dict[str, Any], cfg: AppConfig) -> List[Dict[str, Any]]:
	"""Return detailed violation information with scores and reasons"""
	violations = []
	profile = (cfg.policy_profile or "regular").lower()

	# Enhanced feature extraction
	torso_h = float(features.get("torso_mean_h", 90.0))
	torso_s = float(features.get("torso_mean_s", 50.0))
	torso_v = float(features.get("torso_mean_v", 60.0))
	torso_brightness = float(features.get("torso_brightness", 128.0))
	torso_texture = float(features.get("torso_texture", 0.0))
	
	legs_v = float(features.get("legs_mean_v", 60.0))
	legs_brightness = float(features.get("legs_brightness", 128.0))
	legs_texture = float(features.get("legs_texture", 0.0))
	
	feet_v = float(features.get("feet_mean_v", 60.0))
	feet_brightness = float(features.get("feet_brightness", 128.0))
	feet_texture = float(features.get("feet_texture", 0.0))
	feet_h = float(features.get("feet_mean_h", 90.0))
	feet_s = float(features.get("feet_mean_s", 50.0))
	
	# Pants length analysis (get this first as it helps determine visibility)
	pants_length_ratio = float(features.get("pants_length_ratio", 0.0))
	pants_length_appropriate = float(features.get("pants_length_appropriate", 0.0))
	
	# Get image dimensions FIRST - this is the most reliable indicator
	image_height = float(features.get("image_height", 0.0))
	image_width = float(features.get("image_width", 0.0))
	
	# SIMPLIFIED APPROACH: If image is portrait (height > width), it's ALWAYS a full-body image
	# Portrait images (height > width) virtually always show full body in student verification
	# Calculate aspect ratio - if width is 0, assume it's a valid image (aspect ratio will be high)
	if image_width > 0:
		aspect_ratio = image_height / image_width
	else:
		aspect_ratio = 2.0  # Assume portrait if width is missing/zero
	
	# Portrait = height > width (even slightly)
	# For student verification, portrait images are ALWAYS full-body
	is_portrait_image = aspect_ratio > 1.0  # Any height > width = portrait = full body
	
	# Also check if image has reasonable dimensions (not zero)
	has_valid_dimensions = image_height > 0 and image_width > 0
	
	# Additional signals
	has_full_body_pose = float(features.get("has_full_body_pose", 0.0)) > 0.5
	legs_mask_area = float(features.get("legs_mask_area", 0.0))
	feet_mask_area = float(features.get("feet_mask_area", 0.0))
	
	# CRITICAL FIX: Default to assuming legs and feet are visible UNLESS image is clearly a headshot
	# For student verification, most images are full-body portrait photos
	# Only very small square/landscape images are likely headshots
	
	# Determine if image is clearly a headshot (small square/landscape)
	is_clearly_headshot = (
		has_valid_dimensions and  # Have dimensions
		aspect_ratio <= 1.0 and  # Square or landscape (width >= height)
		image_height < 400 and  # Small height
		image_width < 400 and  # Small width
		not has_full_body_pose  # No pose landmarks
	)
	
	# If clearly a headshot, use feature-based detection
	# Otherwise, assume full-body and set visible = True
	if is_clearly_headshot:
		# Small square/landscape image - might be headshot, use feature detection
		legs_visible = (
			legs_texture > 0.0 or 
			legs_mask_area > 0.0 or
			legs_brightness > 0.0 or
			pants_length_ratio > 0.0 or
			legs_v > 0.0 or
			has_full_body_pose
		)
		
		feet_visible = (
			feet_texture > 0.0 or 
			feet_mask_area > 0.0 or
			feet_brightness > 0.0 or
			pants_length_ratio > 0.0 or
			feet_v > 0.0 or
			has_full_body_pose or
			legs_visible
		)
	else:
		# NOT clearly a headshot - assume full-body image
		# Portrait images, large images, or images with pose = full body
		legs_visible = True
		feet_visible = True
	
	# ID Card detection analysis
	id_card_detected = float(features.get("id_card_detected", 0.0))
	id_card_confidence = float(features.get("id_card_confidence", 0.0))
	id_card_area = float(features.get("id_card_area", 0.0))
	
	# Enhanced detection logic
	# Shoes detection (more tolerant): consider present if any of these signals
	# Improved: if feet region has texture, it's likely shoes (not bare feet)
	# Bare feet typically have higher brightness and lower texture
	shoes_present = (
		feet_texture > 20 or  # Texture indicates shoes (more reliable)
		(feet_brightness < 160 and feet_texture > 10) or  # Darker with some texture
		(feet_v < 160 and feet_texture > 10)  # Low value with texture
	)
	shoes_score = float(np.clip(
		(feet_texture / 100.0) * 0.5 +  # Texture is most reliable (50% weight)
		((160 - min(feet_brightness, 160)) / 160.0) * 0.3 +  # Brightness (30% weight)
		((160 - min(feet_v, 160)) / 160.0) * 0.2,  # Value (20% weight)
		0.0, 1.0
	))
	
	# Black shoe detection: black/dark shoes have low saturation (S) in HSV - this is the key indicator
	# Black shoes can have reflections/lighting that increase brightness/V significantly, so we focus on saturation
	# Low saturation (< 80) with moderate V/brightness indicates black/dark gray colors
	# More lenient: if saturation is very low (< 60), it's almost certainly black regardless of brightness
	# If saturation is moderate (60-80) and V/brightness are not too high, it's likely dark/black
	# This accounts for lighting variations and reflections on black shoes
	shoes_black = (
		(feet_s < 60) or  # Very low saturation = black/dark gray
		(feet_s < 80 and feet_v < 140 and feet_brightness < 140)  # Low saturation with moderate brightness = dark colors
	)
	# Scoring: prioritize low saturation as the main indicator of black
	black_shoes_score = float(np.clip(
		((80 - max(feet_s, 0)) / 80.0) * 0.6 +  # Saturation is the key (60% weight)
		((140 - max(feet_v, 0)) / 140.0) * 0.2 +  # Value (20% weight)
		((140 - max(feet_brightness, 0)) / 140.0) * 0.2,  # Brightness (20% weight)
		0.0, 1.0
	))
	
	# Bottom wear detection: look for appropriate color and texture
	# For males: allow any color pants if configured
	gender = (cfg.policy_gender or "male").lower()
	allow_any_color_pants = (gender == "male" and getattr(cfg, "allow_any_color_pants_male", True))
	bottom_dark = legs_v < 100 and legs_brightness < 120
	bottom_score = float(np.clip((100 - legs_v) / 100.0 + (120 - legs_brightness) / 120.0, 0.0, 1.0)) if (cfg.expected_bottom or "").lower() in ("dark", "black") and not allow_any_color_pants else float(np.clip(legs_v / 100.0 + legs_brightness / 120.0, 0.0, 1.0))
	
	# Top wear detection: enhanced with saturation and texture
	lab_coat_like = (torso_h > 0 and torso_h < 30) and (torso_s < 50) and (torso_v > 200) and (torso_brightness > 200)
	
	# Enhanced high-visibility detection
	hi_vis_like = (
		(torso_h > 20 and torso_h < 40) and (torso_s > 100) and (torso_v > 150) and (torso_brightness > 150)
	) or (
		# Also detect bright fluorescent colors (high saturation + high brightness)
		(torso_s > 150) and (torso_v > 180) and (torso_brightness > 180)
	) or (
		# Detect orange/red high-vis colors
		((torso_h > 0 and torso_h < 20) or (torso_h > 160 and torso_h < 180)) and (torso_s > 120) and (torso_v > 150)
	)
	
	top_appropriate = _keyword_score_from_hue(torso_h, cfg.expected_top) > 0.5
	
	# Calculate top score based on expected color
	if cfg.expected_top.lower() in ("white", "light"):
		top_score = float(np.clip((torso_h < 30 and torso_s < 50 and torso_v > 200) * 1.0, 0.0, 1.0))
	elif cfg.expected_top.lower() in ("yellow", "hi-vis", "high-visibility"):
		top_score = float(np.clip((torso_h > 20 and torso_h < 40 and torso_s > 100) * 1.0, 0.0, 1.0))
	else:
		top_score = _keyword_score_from_hue(torso_h, cfg.expected_top)

	if profile == "regular":
		# Bottom wear check - for males, allow any color pants if configured
		gender = (cfg.policy_gender or "male").lower()
		allow_any_color_pants = (gender == "male" and getattr(cfg, "allow_any_color_pants_male", True))
		
		if legs_visible:
			# For males with any color pants allowed: only check if pants are present (not shorts)
			# For others: check for dark color
			if allow_any_color_pants:
				# Males: any color pants are acceptable - no color-based violation
				# Only check for pants length (shorts vs full pants) which is done below
				# If legs are visible and have texture, pants are likely present
				# No violation needed here for color
				pass
			elif not bottom_dark:
				# For non-males or when dark pants required: check color
				# But only if we have valid color data (not default values)
				if legs_v > 0.0 or legs_brightness > 0.0:
					violations.append({
						"item": "Proper Bottom Wear",
						"required": "Dark trousers/skirt",
						"detected": "Light colored or inappropriate bottom wear",
						"score": bottom_score,
						"severity": "high",
						"reason": f"Expected dark bottom wear but detected light color (brightness: {legs_brightness:.1f}, value: {legs_v:.1f})"
					})
				# If color data is default/zero, skip violation (detection may have failed)
		elif not legs_visible:
			# Only add "not visible" violation if image is clearly a headshot
			# This should rarely happen now since we default to legs_visible = True
			# Only very small square/landscape images without pose will trigger this
			if is_clearly_headshot:
				# Very small square/landscape image - likely a headshot
				violations.append({
					"item": "Bottom Wear Check",
					"required": "Pants (any color)" if allow_any_color_pants else "Dark trousers/skirt",
					"detected": "Not visible in image (headshot/passport photo)",
					"score": 1.0,  # Neutral score - can't verify
					"severity": "low",
					"reason": "Bottom wear not visible in this image type - verification skipped"
				})
			# For all other cases, skip violation
			# legs_visible should be True for most images, so this block should rarely execute
		
		# Footwear check - enhanced for males to check for black shoes
		require_footwear = cfg.require_footwear_male if gender == "male" else cfg.require_footwear_female
		require_black_shoes = (gender == "male" and getattr(cfg, "require_black_shoes_male", True))
		
		if require_footwear and feet_visible:
			if not shoes_present:
				violations.append({
					"item": "Footwear",
					"required": "Black shoes" if require_black_shoes else ("Closed shoes" if gender == "male" else "Footwear (shoes or sandals)"),
					"detected": "Barefoot or inappropriate footwear",
					"score": shoes_score,
					"severity": "high",
					"reason": f"Expected shoes but detected bare feet (brightness: {feet_brightness:.1f}, texture: {feet_texture:.1f})"
				})
			elif require_black_shoes and shoes_present and not shoes_black:
				# Only flag as violation if saturation is high (S > 80) indicating colored shoes
				# High saturation means the shoes have a distinct color (not black/gray)
				# Low saturation (< 80) even with high brightness could be black shoes with reflections
				if feet_s > 80:
					violations.append({
						"item": "Shoe Color",
						"required": "Black shoes only",
						"detected": f"Non-black shoes detected (high saturation indicates colored shoes)",
						"score": black_shoes_score,
						"severity": "high",
						"reason": f"Male students must wear black shoes. Detected shoe has high saturation (S: {feet_s:.1f}) indicating colored shoes, not black. (HSV - H: {feet_h:.1f}, S: {feet_s:.1f}, V: {feet_v:.1f}, Brightness: {feet_brightness:.1f})"
					})
				# If saturation is low but brightness/V are high, it might be black shoes with strong reflections
				# In this case, don't flag as violation to avoid false positives
		elif not feet_visible:
			# Only add "not visible" violation if image is clearly a headshot
			# This should rarely happen now since we default to feet_visible = True
			# Only very small square/landscape images without pose will trigger this
			if is_clearly_headshot:
				# Very small square/landscape image - likely a headshot
				violations.append({
					"item": "Footwear Check",
					"required": "Black shoes" if require_black_shoes else ("Closed shoes" if gender == "male" else "Footwear (optional)"),
					"detected": "Not visible in image (headshot/passport photo)",
					"score": 1.0,  # Neutral score - can't verify
					"severity": "low",
					"reason": "Footwear not visible in this image type - verification skipped"
				})
			# For all other cases, skip violation
			# feet_visible should be True for most images, so this block should rarely execute
		
		# Always check top wear as it should be visible in most photos
		# Enforce shirt for male; kurti+dupatta for female when configured
		# Get torso contrast for shirt detection
		torso_contrast = float(features.get("torso_contrast", 0.0))
		
		def _looks_like_shirt() -> bool:
			# Shirt detection: should have structured appearance (texture), moderate saturation
			# Shirts typically have: texture > 20 (structured fabric), saturation < 120 (not too vibrant)
			# Should not look like a t-shirt (which might have less texture)
			# Should have reasonable brightness (not too dark, not too bright)
			has_structure = torso_texture > 20
			appropriate_saturation = torso_s < 120  # Shirts are usually less saturated than t-shirts
			has_clothing_texture = torso_contrast > 10  # Should have some contrast/patterns
			return has_structure and appropriate_saturation

		def _looks_like_kurti_dupatta() -> bool:
			return (torso_brightness > 120) and (torso_s < 140)
		
		top_ok = top_appropriate
		if gender == "male" and getattr(cfg, "require_shirt_for_male", True):
			top_ok = top_ok and _looks_like_shirt()
		elif gender == "female" and getattr(cfg, "require_kurti_dupatta_for_female", True):
			top_ok = top_ok and _looks_like_kurti_dupatta()

		if not top_ok:
			violations.append({
				"item": "Top Wear",
				"required": (
					"Shirt only (male)" if gender == "male" and getattr(cfg, "require_shirt_for_male", True) else (
						"Kurti with dupatta (female)" if gender == "female" and getattr(cfg, "require_kurti_dupatta_for_female", True) else f"Appropriate top ({cfg.expected_top})"
					)
				),
				"detected": "Inappropriate or missing required top (not a proper shirt)",
				"score": top_score,
				"severity": "high",
				"reason": f"Male students must wear shirts only. Top wear did not meet policy (hue: {torso_h:.1f}, saturation: {torso_s:.1f}, texture: {torso_texture:.1f}, contrast: {torso_contrast:.1f})"
			})
		
		# Check pants length if legs are visible (accept full pants)
		# Only check if pants are too short (shorts), not if they're full-length
		# If pants_length_ratio is 0.0, it means pose detection couldn't calculate it - don't flag as violation
		if legs_visible and pants_length_ratio > 0.0:
			# Use a more lenient threshold - only flag if clearly shorts (ratio < 0.20 instead of 0.25)
			# This prevents false positives when pants are present but ratio calculation is slightly off
			# Full-length pants typically have ratio > 0.25, so 0.20 gives more tolerance
			if pants_length_ratio < 0.20:
				violations.append({
					"item": "Bottom Wear",
					"required": "Full-length pants (any color for males)",
					"detected": "Pants appear too short (shorts not allowed)",
					"score": pants_length_appropriate,
					"severity": "high",
					"reason": f"Pants length ratio: {pants_length_ratio:.1%} (< 20% indicates shorts). Full-length pants required."
				})
			# If pants_length_ratio >= 0.20, pants are acceptable (no violation)
		elif legs_visible and pants_length_ratio == 0.0:
			# Pants length couldn't be calculated (pose detection issue), but legs are visible
			# Assume pants are present if legs are visible with texture - don't flag as violation
			# This prevents false positives when pose detection is incomplete
			pass
	elif profile == "sports":
		if not hi_vis_like:
			violations.append({
				"item": "Sports Attire",
				"required": "High-visibility/bright sports top",
				"detected": "Inappropriate sports top",
				"score": _keyword_score_from_hue(torso_h, "yellow"),
				"severity": "high",
				"reason": f"Expected bright sports top but detected different color (hue: {torso_h:.1f}, saturation: {torso_s:.1f})"
			})
		if not shoes_present:
			violations.append({
				"item": "Sports Footwear",
				"required": "Sports shoes",
				"detected": "Barefoot or inappropriate footwear",
				"score": shoes_score,
				"severity": "high",
				"reason": f"Expected sports shoes but detected bare feet (brightness: {feet_brightness:.1f}, texture: {feet_texture:.1f})"
			})
	elif profile == "lab":
		if not lab_coat_like:
			violations.append({
				"item": "Lab Safety",
				"required": "Lab coat or white protective clothing",
				"detected": "Missing lab coat",
				"score": _keyword_score_from_hue(torso_h, "white"),
				"severity": "critical",
				"reason": f"Expected lab coat but detected different color (hue: {torso_h:.1f}, brightness: {torso_brightness:.1f})"
			})
		if not shoes_present:
			violations.append({
				"item": "Lab Footwear",
				"required": "Closed-toe shoes",
				"detected": "Barefoot or open-toe shoes",
				"score": shoes_score,
				"severity": "critical",
				"reason": f"Expected closed-toe shoes for lab safety but detected bare feet (brightness: {feet_brightness:.1f}, texture: {feet_texture:.1f})"
			})
	else:
		# Default to regular profile logic
		# Only check bottom wear if legs are actually visible
		if legs_visible and not bottom_dark:
			violations.append({
				"item": "Proper Bottom Wear",
				"required": "Dark trousers/skirt",
				"detected": "Light colored or inappropriate bottom wear",
				"score": bottom_score,
				"severity": "high",
				"reason": f"Expected dark bottom wear but detected light color (brightness: {legs_brightness:.1f}, value: {legs_v:.1f})"
			})
		elif not legs_visible:
			# For headshots/passport photos, add an informational note instead of a violation
			violations.append({
				"item": "Bottom Wear Check",
				"required": "Dark trousers/skirt",
				"detected": "Not visible in image (headshot/passport photo)",
				"score": 1.0,  # Neutral score - can't verify
				"severity": "low",
				"reason": "Bottom wear not visible in this image type - verification skipped"
			})
		
		# Only check footwear if feet are actually visible
		if feet_visible and not shoes_present:
			violations.append({
				"item": "Footwear",
				"required": "Closed shoes",
				"detected": "Barefoot or inappropriate footwear",
				"score": shoes_score,
				"severity": "high",
				"reason": f"Expected closed shoes but detected bare feet (brightness: {feet_brightness:.1f}, texture: {feet_texture:.1f})"
			})
		elif not feet_visible:
			# For headshots/passport photos, add an informational note instead of a violation
			violations.append({
				"item": "Footwear Check",
				"required": "Closed shoes",
				"detected": "Not visible in image (headshot/passport photo)",
				"score": 1.0,  # Neutral score - can't verify
				"severity": "low",
				"reason": "Footwear not visible in this image type - verification skipped"
			})
	
	# Check ID card for all profiles if enabled (outside profile-specific logic)
	if cfg.enable_id_card_detection and cfg.id_card_required:
		if not id_card_detected or id_card_confidence < cfg.id_card_confidence_threshold:
			violations.append({
				"item": "ID Card",
				"required": "Valid student ID card visible",
				"detected": "No ID card detected or confidence too low",
				"score": id_card_confidence,
				"severity": "high",
				"reason": f"ID card detection confidence: {id_card_confidence:.1%} (minimum required: {cfg.id_card_confidence_threshold:.1%})"
			})

	return violations


def verify_attire_and_safety(features: Dict[str, Any], cfg: AppConfig, clf: AttireClassifier | None = None) -> Dict[str, Any]:
	# Rule-based
	rule = rule_based_checks(features, cfg) if cfg.enable_rules else {"rule_score": 0.5}
	combined_score = rule.get("rule_score", 0.5)
	label = "unknown"

	# ML model probability for best class if available
	if cfg.enable_model and clf is not None and clf.model is not None:
		feature_values = [v for k, v in features.items() if k not in ("image", "label")]
		X = np.array(feature_values, dtype=float).reshape(1, -1)
		proba = clf.predict_proba(X)[0]
		p = float(np.max(proba))
		combined_score = float(0.5 * combined_score + 0.5 * p)
		if clf.label_names:
			label = clf.label_names[int(np.argmax(proba))]

	# Get detailed violations
	violations = _infer_missing_items(features, cfg)
	
	# Calculate detailed scores
	success_score = combined_score
	fail_score = 1.0 - combined_score
	
	# Calculate violation penalty based on severity
	violation_penalty = 0.0
	critical_violations = 0
	high_violations = 0
	medium_violations = 0
	
	for violation in violations:
		severity = violation.get("severity", "medium")
		if severity == "critical":
			critical_violations += 1
			violation_penalty += 0.3
		elif severity == "high":
			high_violations += 1
			violation_penalty += 0.2
		elif severity == "medium":
			medium_violations += 1
			violation_penalty += 0.1
	
	# Apply penalty to scores
	success_score = float(np.clip(success_score - violation_penalty, 0.0, 1.0))
	fail_score = float(np.clip(fail_score + violation_penalty, 0.0, 1.0))
	
	# Determine overall status
	has_critical = critical_violations > 0
	has_high = high_violations > 0
	has_medium = medium_violations > 0
	
	if has_critical or (has_high and success_score < 0.6) or success_score < cfg.confidence_threshold:
		status = "FAIL"
	elif has_high or has_medium:
		status = "WARNING"
	else:
		status = "PASS"
	
	# Create summary of violations
	violation_summary = {
		"total_violations": len(violations),
		"critical": critical_violations,
		"high": high_violations,
		"medium": medium_violations,
		"violations": violations
	}
	
	return {
		"status": status,
		"success_score": success_score,
		"fail_score": fail_score,
		"score": success_score,  # Keep for backward compatibility
		"label": label,
		"violations": violation_summary,
		"details": rule,
		"summary": {
			"overall_compliance": f"{success_score:.1%}",
			"violation_count": len(violations),
			"severity_breakdown": {
				"critical": critical_violations,
				"high": high_violations,
				"medium": medium_violations
			}
		}
	}
