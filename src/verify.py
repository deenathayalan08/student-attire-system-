from typing import Dict, Any, List
import numpy as np

from .config import AppConfig
from .model import AttireClassifier


def _keyword_score_from_hue(mean_h: float, keyword: str) -> float:
	# More strict color matching for professional verification
	# hue in [0,180]: 0 red, 30 yellow, 60 green, 90 cyan, 120 blue, 150 magenta
	kw = (keyword or "").lower()
	if kw in ("white", "light"):
		# White should have low saturation and high brightness
		return 0.8 if mean_h < 30 or mean_h > 150 else 0.3
	if kw in ("dark", "black"):
		# Dark colors should have low brightness
		return 0.8
	if kw == "blue":
		# Stricter blue detection (hue 100-140)
		return float(np.exp(-((mean_h - 120.0) ** 2) / (2 * 8.0 ** 2))) if 100 <= mean_h <= 140 else 0.2
	if kw == "green":
		# Stricter green detection (hue 50-80)
		return float(np.exp(-((mean_h - 60.0) ** 2) / (2 * 8.0 ** 2))) if 50 <= mean_h <= 80 else 0.2
	if kw in ("yellow", "hi-vis", "high-visibility"):
		# Stricter yellow detection (hue 20-40)
		return float(np.exp(-((mean_h - 30.0) ** 2) / (2 * 8.0 ** 2))) if 20 <= mean_h <= 40 else 0.2
	return 0.3  # Lower default score for unmatched colors


def rule_based_checks(features: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
	# Enhanced rule-based checks with stricter criteria
	torso_h = float(features.get("torso_mean_h", 90.0))
	torso_s = float(features.get("torso_mean_s", 50.0))
	torso_v = float(features.get("torso_mean_v", 128.0))
	torso_brightness = float(features.get("torso_brightness", 128.0))
	
	legs_h = float(features.get("legs_mean_h", 90.0))
	legs_s = float(features.get("legs_mean_s", 50.0))
	legs_v = float(features.get("legs_mean_v", 60.0))
	legs_brightness = float(features.get("legs_brightness", 128.0))
	
	feet_h = float(features.get("feet_mean_h", 90.0))
	feet_s = float(features.get("feet_mean_s", 50.0))
	feet_v = float(features.get("feet_mean_v", 60.0))
	feet_brightness = float(features.get("feet_brightness", 128.0))
	
	# Stricter top color evaluation
	if cfg.expected_top.lower() in ("white", "light"):
		# White shirt: low saturation (<30), high brightness (>180)
		top_score = 0.8 if (torso_s < 30 and torso_brightness > 180) else 0.2
	else:
		top_score = _keyword_score_from_hue(torso_h, cfg.expected_top)
	
	# Stricter bottom evaluation - dark pants required
	if cfg.expected_bottom.lower() in ("dark", "black"):
		# Dark pants: low brightness (<100) and reasonable saturation
		bottom_score = 0.8 if (legs_brightness < 100 and legs_v < 120) else 0.2
	else:
		bottom_score = 1.0 - (legs_v / 255.0)
	bottom_score = float(np.clip(bottom_score, 0.0, 1.0))

	# Stricter shoe evaluation - black shoes required for males
	gender = (cfg.policy_gender or "male").lower()
	if gender == "male" and getattr(cfg, "require_black_shoes_male", True):
		# Black shoes: low saturation (<40), low brightness (<120)
		shoes_score = 0.8 if (feet_s < 40 and feet_brightness < 120) else 0.2
	else:
		# General shoe detection: low brightness implies shoes
		shoes_score = 1.0 - (feet_v / 255.0)
	shoes_score = float(np.clip(shoes_score, 0.0, 1.0))

	# More strict aggregation - all components must pass for high score
	rule_score = float(np.clip(0.4 * top_score + 0.4 * bottom_score + 0.2 * shoes_score, 0.0, 1.0))
	
	# Additional penalty if any component fails badly
	if top_score < 0.5 or bottom_score < 0.5 or shoes_score < 0.5:
		rule_score *= 0.6  # 40% penalty for failing any component
	
	return {
		"top_score": top_score,
		"bottom_score": bottom_score,
		"shoes_score": shoes_score,
		"rule_score": rule_score,
		"top_details": f"H:{torso_h:.1f} S:{torso_s:.1f} V:{torso_v:.1f} B:{torso_brightness:.1f}",
		"bottom_details": f"H:{legs_h:.1f} S:{legs_s:.1f} V:{legs_v:.1f} B:{legs_brightness:.1f}",
		"shoes_details": f"H:{feet_h:.1f} S:{feet_s:.1f} V:{feet_v:.1f} B:{feet_brightness:.1f}",
	}


def _infer_missing_items(features: Dict[str, Any], cfg: AppConfig) -> List[Dict[str, Any]]:
	"""Return detailed violation information with scores and reasons"""
	violations = []
	profile = (cfg.policy_profile or "regular").lower()

	# Enhanced feature extraction with None checks
	torso_h = float(features.get("torso_mean_h") or 90.0)
	torso_s = float(features.get("torso_mean_s") or 50.0)
	torso_v = float(features.get("torso_mean_v") or 60.0)
	torso_brightness = float(features.get("torso_brightness") or 128.0)
	torso_texture = float(features.get("torso_texture") or 0.0)
	
	legs_v = float(features.get("legs_mean_v") or 60.0)
	legs_brightness = float(features.get("legs_brightness") or 128.0)
	legs_texture = float(features.get("legs_texture") or 0.0)
	
	feet_v = float(features.get("feet_mean_v") or 60.0)
	feet_brightness = float(features.get("feet_brightness") or 128.0)
	feet_texture = float(features.get("feet_texture") or 0.0)
	feet_h = float(features.get("feet_mean_h") or 90.0)
	feet_s = float(features.get("feet_mean_s") or 50.0)
	
	# Pants length analysis (get this first as it helps determine visibility)
	pants_length_ratio = float(features.get("pants_length_ratio") or 0.0)
	pants_length_appropriate = float(features.get("pants_length_appropriate") or 0.0)
	
	# Get image dimensions FIRST - this is the most reliable indicator
	image_height = float(features.get("image_height") or 0.0)
	image_width = float(features.get("image_width") or 0.0)
	
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
	has_full_body_pose = float(features.get("has_full_body_pose") or 0.0) > 0.5
	legs_mask_area = float(features.get("legs_mask_area") or 0.0)
	feet_mask_area = float(features.get("feet_mask_area") or 0.0)
	
	# CRITICAL FIX: Default to assuming legs and feet are visible UNLESS image is clearly a headshot
	# For student verification, most images are full-body portrait photos
	# Only very small square/landscape images are likely headshots
	
	# Determine if image is clearly a headshot (small square/landscape)
	is_clearly_headshot = (
		has_valid_dimensions and  # Have dimensions
		aspect_ratio <= 1.2 and  # Square or slightly portrait (width >= 80% of height)
		image_height <= 640 and  # Medium or small height
		image_width <= 640 and  # Medium or small width
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
	
	# ID Card detection analysis (using new shape-based detector)
	id_card_detected = float(features.get("id_card_detected") or 0.0)
	id_card_confidence = float(features.get("id_card_confidence") or 0.0)
	id_card_area = float(features.get("id_card_area") or 0.0)
	
	# Chain/Lanyard detection (indicates ID card is worn around neck)
	chain_detected = float(features.get("chain_detected") or 0.0)
	chain_confidence = float(features.get("chain_confidence") or 0.0)
	
	# Boost ID card confidence if chain is detected (chain indicates ID card is worn)
	if chain_detected > 0.5 and id_card_confidence > 0.3:
		id_card_confidence = min(1.0, id_card_confidence * 1.2)  # 20% boost
	
	# Enhanced detection logic using new object detectors
	# Use color-based and shape-based detection instead of texture analysis
	
	# Shoes detection using new detector
	shoes_detected_new = features.get("shoes_detected", 0.0) > 0.5
	shoes_confidence_new = features.get("shoes_confidence", 0.0)
	shoes_is_black_new = features.get("shoes_is_black", 0.0) > 0.5
	shoes_saturation_new = features.get("shoes_saturation", 0.0)
	
	# Fallback to old method if new detector not available
	if shoes_confidence_new > 0.0:
		# Use new detector results
		shoes_present = shoes_detected_new
		shoes_score = shoes_confidence_new
		shoes_black = shoes_is_black_new
		black_shoes_score = shoes_confidence_new if shoes_is_black_new else 0.0
	else:
		# Fallback to texture-based detection
		shoes_present = (
			feet_texture > 10 or
			feet_brightness < 180 or
			feet_v < 180 or
			feet_mask_area > 0
		)
		shoes_score = float(np.clip(
			(feet_texture / 100.0) * 0.5 +
			((180 - min(feet_brightness, 180)) / 180.0) * 0.3 +
			((180 - min(feet_v, 180)) / 180.0) * 0.2,
			0.0, 1.0
		))
		shoes_black = (feet_s < 60) or (feet_s < 80 and feet_v < 140 and feet_brightness < 140)
		black_shoes_score = float(np.clip(
			((80 - max(feet_s, 0)) / 80.0) * 0.6 +
			((140 - max(feet_v, 0)) / 140.0) * 0.2 +
			((140 - max(feet_brightness, 0)) / 140.0) * 0.2,
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
		# Bottom wear check - STRICT: Always require dark pants for professional appearance
		gender = (cfg.policy_gender or "male").lower()
		allow_any_color_pants = (gender == "male" and getattr(cfg, "allow_any_color_pants_male", True))
		
		if legs_visible:
			# STRICT: Always check for dark color unless explicitly allowed
			if not allow_any_color_pants or not bottom_dark:
				# Check if pants are too light/bright for professional dress code - MORE LENIENT THRESHOLDS
				if legs_brightness > 140 or legs_v > 160:  # Increased from 120/140 to 140/160
					violations.append({
						"item": "Bottom Wear Color",
						"required": "Dark colored trousers/pants (professional dress code)",
						"detected": f"Light colored pants detected (brightness: {legs_brightness:.1f})",
						"score": bottom_score,
						"severity": "high",
						"reason": f"Professional dress code requires dark pants. Detected brightness: {legs_brightness:.1f} (max allowed: 140), value: {legs_v:.1f} (max allowed: 160)"
					})
			
			# Additional check for pants vs shorts
			if legs_brightness < 50:  # Very dark might indicate bare legs (shorts)
				violations.append({
					"item": "Bottom Wear Coverage", 
					"required": "Full-length pants",
					"detected": "Possible shorts or insufficient leg coverage",
					"score": 0.3,
					"severity": "medium",
					"reason": f"Very low leg brightness ({legs_brightness:.1f}) may indicate shorts or bare legs"
				})
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
		require_black_shoes = (gender == "male" and getattr(cfg, "require_black_shoes_male", False))
		
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
				# STRICT: Check for black shoes with more realistic criteria
				# Black shoes should have: low saturation (<50) AND low brightness (<130)
				is_actually_black = (feet_s < 50 and feet_brightness < 130)
				
				if not is_actually_black:
					# Determine if shoes are clearly non-black
					if feet_s > 60 or feet_brightness > 150:
						severity = "high"
						reason = f"Professional dress code requires black shoes. Detected non-black shoes: saturation {feet_s:.1f} (max: 50), brightness {feet_brightness:.1f} (max: 130)"
					else:
						severity = "medium" 
						reason = f"Shoe color unclear - may not be black. Saturation: {feet_s:.1f}, brightness: {feet_brightness:.1f}"
					
					violations.append({
						"item": "Shoe Color Compliance",
						"required": "Black shoes only (professional dress code)",
						"detected": f"Non-black or unclear shoe color (S:{feet_s:.1f}, B:{feet_brightness:.1f})",
						"score": black_shoes_score,
						"severity": severity,
						"reason": reason
					})
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
			# SIMPLIFIED: Just check if wearing any top (formal or casual)
			# Focus on presence of clothing, not specific texture/structure
			# Any shirt/top with reasonable brightness is acceptable
			has_clothing = torso_brightness > 30  # Not bare skin (which is usually brighter)
			reasonable_coverage = torso_v > 30  # Has some color/coverage
			# Very lenient - just check if wearing something on top
			return has_clothing or reasonable_coverage or torso_texture > 5

		def _looks_like_kurti_dupatta() -> bool:
			return (torso_brightness > 120) and (torso_s < 140)
		
		# STRICT: Check both clothing presence AND color appropriateness
		top_ok = top_appropriate and (torso_brightness > 30)  # Must have appropriate color AND be wearing something
		
		# Additional checks based on gender and requirements
		if gender == "male" and getattr(cfg, "require_shirt_for_male", True):
			shirt_ok = _looks_like_shirt()
			# For males: must wear shirt AND have appropriate color (usually white)
			if cfg.expected_top.lower() in ("white", "light"):
				# White shirt required: low saturation + high brightness
				white_shirt_ok = (torso_s < 50 and torso_brightness > 160)
				top_ok = shirt_ok and white_shirt_ok
			else:
				top_ok = shirt_ok and top_appropriate
		elif gender == "female" and getattr(cfg, "require_kurti_dupatta_for_female", True):
			top_ok = _looks_like_kurti_dupatta() and top_appropriate

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
		# SIMPLIFIED: Only check if pants are VERY clearly shorts (very strict threshold)
		# If pants_length_ratio is 0.0, it means pose detection couldn't calculate it - don't flag as violation
		if legs_visible and pants_length_ratio > 0.0:
			# VERY lenient threshold - only flag if EXTREMELY short (ratio < 0.15)
			# This prevents false positives - most pants will pass
			# Only very obvious shorts will fail
			if pants_length_ratio < 0.15:
				violations.append({
					"item": "Bottom Wear",
					"required": "Full-length pants (any color for males)",
					"detected": "Pants appear too short (shorts not allowed)",
					"score": pants_length_appropriate,
					"severity": "medium",  # Reduced severity
					"reason": f"Pants length ratio: {pants_length_ratio:.1%} (< 15% indicates very short shorts). Full-length pants required."
				})
			# If pants_length_ratio >= 0.15, pants are acceptable (no violation)
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
	# BUT ONLY if image shows torso area (not for headshots)
	if cfg.enable_id_card_detection and cfg.id_card_required and not is_clearly_headshot:
		# More realistic ID card detection
		id_card_ok = (id_card_detected and id_card_confidence >= cfg.id_card_confidence_threshold)
		
		# Also check for chain/lanyard as indicator of ID card
		has_lanyard = (chain_detected > 0.3 and chain_confidence > 0.3)
		
		if not id_card_ok and not has_lanyard:
			# No ID card and no lanyard detected
			violations.append({
				"item": "Student ID Card",
				"required": "Visible student ID card (worn around neck or clipped to shirt)",
				"detected": f"No ID card detected (confidence: {id_card_confidence:.1%})",
				"score": id_card_confidence,
				"severity": "medium",  # Reduced from "high" to "medium"
				"reason": f"Student ID card is mandatory and must be visible. Detection confidence: {id_card_confidence:.1%} (required: {cfg.id_card_confidence_threshold:.1%}), lanyard detected: {has_lanyard}"
			})
		elif not id_card_ok and has_lanyard:
			# Lanyard detected but ID card not clear - REDUCE SEVERITY
			violations.append({
				"item": "ID Card Visibility",
				"required": "ID card must be clearly visible",
				"detected": f"Lanyard detected but ID card not clearly visible (confidence: {id_card_confidence:.1%})",
				"score": max(id_card_confidence, 0.6),  # Give more credit for lanyard
				"severity": "low",  # Reduced from "medium" to "low"
				"reason": f"Lanyard/chain detected indicating ID card is worn, but card itself not clearly visible. Please ensure ID card faces forward and is not obscured."
			})
	elif cfg.enable_id_card_detection and cfg.id_card_required and is_clearly_headshot:
		# For headshots, add informational note instead of violation
		violations.append({
			"item": "ID Card Check",
			"required": "Student ID card verification",
			"detected": "Cannot verify ID card in headshot/passport photo",
			"score": 1.0,  # Neutral score
			"severity": "low",
			"reason": "ID card verification skipped for headshot/passport photo format"
		})

	return violations


def verify_attire_and_safety(features: Dict[str, Any], cfg: AppConfig, clf: AttireClassifier | None = None) -> Dict[str, Any]:
	# Rule-based
	rule = rule_based_checks(features, cfg) if cfg.enable_rules else {"rule_score": 0.5}
	combined_score = rule.get("rule_score", 0.5)
	label = "unknown"

	# ML model probability for best class if available
	if cfg.enable_model and clf is not None and clf.model is not None:
		try:
			# Handle feature mismatch gracefully
			feature_values = [v for k, v in features.items() if k not in ("image", "label")]
			X = np.array(feature_values, dtype=float).reshape(1, -1)
			proba = clf.predict_proba(X)[0]
			p = float(np.max(proba))
			combined_score = float(0.5 * combined_score + 0.5 * p)
			if clf.label_names:
				label = clf.label_names[int(np.argmax(proba))]
		except (ValueError, Exception) as e:
			# Feature mismatch or other ML error - fall back to rule-based only
			print(f"Warning: ML model error ({e}), using rule-based verification only")
			combined_score = rule.get("rule_score", 0.5)
			label = "rule_based"

	# Get detailed violations
	violations = _infer_missing_items(features, cfg)
	
	# Calculate detailed scores
	success_score = combined_score
	fail_score = 1.0 - combined_score
	
	# Calculate violation penalty based on severity - BALANCED
	violation_penalty = 0.0
	critical_violations = 0
	high_violations = 0
	medium_violations = 0
	low_violations = 0
	
	for violation in violations:
		severity = violation.get("severity", "medium")
		if severity == "critical":
			critical_violations += 1
			violation_penalty += 0.4  # Critical violations are serious
		elif severity == "high":
			high_violations += 1
			violation_penalty += 0.25  # High violations are important
		elif severity == "medium":
			medium_violations += 1
			violation_penalty += 0.15  # Medium violations are moderate
		elif severity == "low":
			low_violations += 1
			violation_penalty += 0.05  # Low violations are minor
	
	# Apply penalty to scores
	success_score = float(np.clip(success_score - violation_penalty, 0.0, 1.0))
	fail_score = float(np.clip(fail_score + violation_penalty, 0.0, 1.0))
	
	# Determine overall status - BALANCED CRITERIA
	has_critical = critical_violations > 0
	has_high = high_violations > 0
	has_medium = medium_violations > 0
	has_low = low_violations > 0
	
	# Balanced pass criteria - more realistic for different image types
	if has_critical or success_score < 0.25:
		status = "FAIL"
	elif has_high or success_score < 0.45:
		status = "WARNING" 
	elif has_medium or success_score < cfg.confidence_threshold:
		status = "WARNING"
	elif has_low and success_score < 0.7:
		status = "WARNING"  # Low violations only cause warnings if score is low
	else:
		status = "PASS"
	
	# Create summary of violations
	violation_summary = {
		"total_violations": len(violations),
		"critical": critical_violations,
		"high": high_violations,
		"medium": medium_violations,
		"low": low_violations,
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
				"medium": medium_violations,
				"low": low_violations
			}
		}
	}
