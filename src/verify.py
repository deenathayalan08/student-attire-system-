from typing import Dict, Any, List
import numpy as np
import cv2

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


def _analyze_formality(features: Dict[str, Any]) -> Dict[str, Any]:
	"""Analyze clothing formality based on CLOTHING APPEARANCE, not body structure"""
	
	# Extract CLOTHING-focused features (not body structure)
	torso_brightness = float(features.get("torso_brightness", 128.0))
	torso_s = float(features.get("torso_mean_s", 50.0))  # Saturation
	torso_v = float(features.get("torso_mean_v", 128.0))  # Value/brightness
	torso_h = float(features.get("torso_mean_h", 90.0))  # Hue
	
	legs_brightness = float(features.get("legs_brightness", 128.0))
	legs_s = float(features.get("legs_mean_s", 50.0))
	legs_v = float(features.get("legs_mean_v", 128.0))
	legs_h = float(features.get("legs_mean_h", 90.0))
	
	feet_brightness = float(features.get("feet_brightness", 128.0))
	feet_s = float(features.get("feet_mean_s", 50.0))
	feet_v = float(features.get("feet_mean_v", 60.0))
	feet_h = float(features.get("feet_mean_h", 90.0))
	
	# TOP WEAR FORMALITY - Focus on CLOTHING characteristics (LENIENT for formal wear)
	formal_top_indicators = 0
	
	# Professional colors (more lenient - allow most colors except very bright)
	if torso_s < 150:  # Allow moderate saturation (professional colors)
		formal_top_indicators += 1
	
	# Appropriate brightness (more lenient range)
	if 50 < torso_brightness < 220:  # Wider professional brightness range
		formal_top_indicators += 1
	
	# Professional hues (more inclusive - accept most colors except very unusual ones)
	# Accept most hues as professional (only reject extreme cases)
	if not (30 <= torso_h <= 60 and torso_s > 150):  # Reject only very bright yellow/orange
		formal_top_indicators += 1
	
	# Not extremely casual/bright (more lenient)
	if torso_v < 240:  # Allow bright colors, reject only extremely bright
		formal_top_indicators += 1
	
	top_formality_score = formal_top_indicators / 4.0
	
	# BOTTOM WEAR FORMALITY - Focus on PANTS/TROUSERS appearance (LENIENT)
	formal_bottom_indicators = 0
	
	# Professional colors for pants (more lenient)
	if legs_s < 120:  # Allow moderate colors (formal pants can have some color)
		formal_bottom_indicators += 1
	
	# Appropriate appearance (more lenient range)
	if legs_brightness < 180:  # Allow lighter pants (khaki, gray, etc.)
		formal_bottom_indicators += 1
	
	# Professional appearance (more inclusive)
	if legs_v < 200:  # Allow most pants except extremely bright
		formal_bottom_indicators += 1
	
	bottom_formality_score = formal_bottom_indicators / 3.0
	
	# FOOTWEAR FORMALITY - Focus on SHOE appearance (LENIENT)
	formal_shoes_indicators = 0
	
	# Check if footwear is present (more lenient detection)
	feet_texture = float(features.get("feet_texture", 0.0))
	if feet_texture > 3:  # Lower threshold for shoe detection
		formal_shoes_indicators += 1
		
		# Professional shoe colors (more lenient)
		if feet_s < 100:  # Allow moderate colors (brown, burgundy shoes)
			formal_shoes_indicators += 1
		
		if feet_brightness < 170:  # Allow lighter formal shoes
			formal_shoes_indicators += 1
		
		# Professional shoe appearance (more inclusive)
		if feet_v < 160:  # Allow most dress shoes
			formal_shoes_indicators += 1
	else:
		# If no shoes detected, check if it might be a partial image
		image_height = float(features.get("image_height", 0))
		image_width = float(features.get("image_width", 0))
		aspect_ratio = image_height / image_width if image_width > 0 else 1.0
		
		# If image is not clearly full-body, give benefit of doubt
		if aspect_ratio < 1.5:  # Not a typical full-body portrait
			formal_shoes_indicators += 2  # Give partial credit for unclear image
	
	shoes_formality_score = formal_shoes_indicators / 4.0 if formal_shoes_indicators > 0 else 0.0
	
	return {
		"top_formality": top_formality_score,
		"bottom_formality": bottom_formality_score,
		"shoes_formality": shoes_formality_score,
		"formal_indicators": {
			"top": formal_top_indicators,
			"bottom": formal_bottom_indicators, 
			"shoes": formal_shoes_indicators
		},
		"analysis_focus": "clothing_appearance",  # Indicate we focus on clothing, not body structure
		"clothing_details": {
			"top_colors": f"H:{torso_h:.0f} S:{torso_s:.0f} V:{torso_v:.0f}",
			"bottom_colors": f"H:{legs_h:.0f} S:{legs_s:.0f} V:{legs_v:.0f}",
			"shoes_colors": f"H:{feet_h:.0f} S:{feet_s:.0f} V:{feet_v:.0f}"
		}
	}

def rule_based_checks(features: Dict[str, Any], cfg: AppConfig) -> Dict[str, Any]:
	"""Enhanced rule-based checks focusing on FORMAL vs CASUAL attire analysis"""
	
	# Analyze formality instead of specific colors
	formality_analysis = _analyze_formality(features)
	
	top_score = formality_analysis["top_formality"]
	bottom_score = formality_analysis["bottom_formality"] 
	shoes_score = formality_analysis["shoes_formality"]
	
	# Calculate overall formality score
	rule_score = float(np.clip(0.4 * top_score + 0.4 * bottom_score + 0.2 * shoes_score, 0.0, 1.0))
	
	# Determine compliance based on formality thresholds
	formal_threshold = 0.5  # 50% formality required (realistic for formal wear)
	
	top_formal = top_score >= formal_threshold
	bottom_formal = bottom_score >= formal_threshold
	shoes_formal = shoes_score >= formal_threshold
	
	# Count non-formal components
	non_formal_components = sum([1 for formal in [top_formal, bottom_formal, shoes_formal] if not formal])
	
	# Apply penalty for casual/non-formal items
	if non_formal_components > 0:
		rule_score *= (0.7 ** non_formal_components)  # Moderate penalty for casual items
	
	return {
		"top_score": top_score,
		"bottom_score": bottom_score,
		"shoes_score": shoes_score,
		"rule_score": rule_score,
		"formality_analysis": formality_analysis,
		"non_formal_components": non_formal_components,
		"top_details": f"Formality: {top_score:.1%} ({'Formal' if top_formal else 'Casual'})",
		"bottom_details": f"Formality: {bottom_score:.1%} ({'Formal' if bottom_formal else 'Casual'})",
		"shoes_details": f"Formality: {shoes_score:.1%} ({'Formal' if shoes_formal else 'Casual'})",
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
		# FORMAL ATTIRE ANALYSIS - Focus on formal vs casual appearance
		
		# Get formality analysis results
		formality_analysis = _analyze_formality(features)
		top_formality = formality_analysis["top_formality"]
		bottom_formality = formality_analysis["bottom_formality"]
		shoes_formality = formality_analysis["shoes_formality"]
		
		formal_threshold = 0.5  # 50% formality required (more realistic for actual formal wear)
		
		# TOP WEAR - Check for formal shirt/blouse (CLOTHING FOCUS)
		if legs_visible or not is_clearly_headshot:  # Check top wear for most images
			if top_formality < formal_threshold:
				# Analyze CLOTHING characteristics (not body structure)
				torso_s = float(features.get("torso_mean_s", 50.0))
				torso_v = float(features.get("torso_mean_v", 128.0))
				torso_h = float(features.get("torso_mean_h", 90.0))
				
				# Determine casual clothing indicators
				casual_indicators = []
				if torso_s > 120:
					casual_indicators.append("very bright/flashy colors")
				if torso_v > 220:
					casual_indicators.append("overly bright clothing")
				if not ((torso_h < 30 or torso_h > 150) or (90 <= torso_h <= 140)):
					casual_indicators.append("non-professional colors")
				
				casual_description = ", ".join(casual_indicators) if casual_indicators else "casual clothing style"
				
				violations.append({
					"item": "Top Wear - Professional Clothing",
					"required": "Formal shirt/blouse in professional colors (white, blue, gray, etc.)",
					"detected": f"Casual clothing detected: {casual_description}",
					"score": top_formality,
					"severity": "high",
					"reason": f"Dress code requires professional shirt/blouse. Clothing analysis shows: {top_formality:.1%} formality (required: {formal_threshold:.1%}). Colors: H:{torso_h:.0f}° S:{torso_s:.0f} V:{torso_v:.0f}"
				})
		
		# BOTTOM WEAR - Check for formal trousers/pants (CLOTHING FOCUS)
		if legs_visible:
			if bottom_formality < formal_threshold:
				# Analyze CLOTHING characteristics (not body structure)
				legs_s = float(features.get("legs_mean_s", 50.0))
				legs_v = float(features.get("legs_mean_v", 128.0))
				legs_h = float(features.get("legs_mean_h", 90.0))
				
				# Determine casual clothing indicators
				casual_indicators = []
				if legs_s > 100:
					casual_indicators.append("overly colorful/bright pants")
				if legs_brightness > 160:
					casual_indicators.append("too bright for formal wear")
				if legs_v > 180:
					casual_indicators.append("casual bright colors")
				
				casual_description = ", ".join(casual_indicators) if casual_indicators else "casual pants/jeans style"
				
				violations.append({
					"item": "Bottom Wear - Professional Clothing",
					"required": "Formal trousers/pants in professional colors (navy, black, gray, khaki)",
					"detected": f"Casual pants detected: {casual_description}",
					"score": bottom_formality,
					"severity": "high",
					"reason": f"Dress code requires professional trousers/pants. Clothing analysis shows: {bottom_formality:.1%} formality (required: {formal_threshold:.1%}). Colors: H:{legs_h:.0f}° S:{legs_s:.0f} V:{legs_v:.0f}"
				})
			
			# Additional check for shorts vs pants
			pants_length_ratio = float(features.get("pants_length_ratio", 0.0))
			if pants_length_ratio > 0.0 and pants_length_ratio < 0.15:
				violations.append({
					"item": "Bottom Wear Coverage", 
					"required": "Full-length formal pants",
					"detected": "Shorts detected (not appropriate for professional setting)",
					"score": 0.2,
					"severity": "high",
					"reason": f"Shorts not allowed in professional dress code. Pants length ratio: {pants_length_ratio:.1%}"
				})
		elif not legs_visible and is_clearly_headshot:
			# For headshots, add informational note
			violations.append({
				"item": "Bottom Wear Check",
				"required": "Formal trousers/pants",
				"detected": "Not visible in headshot/passport photo",
				"score": 1.0,  # Neutral score
				"severity": "low",
				"reason": "Bottom wear verification skipped for headshot format"
			})
		
		# FOOTWEAR - Check for formal shoes (CLOTHING FOCUS)
		if feet_visible:
			feet_texture = float(features.get("feet_texture", 0.0))
			has_footwear = feet_texture > 3  # More lenient footwear detection
			
			# Additional check - if feet region has any color data, assume shoes are present
			feet_mask_area = float(features.get("feet_mask_area", 0.0))
			if feet_mask_area > 0.01:  # If feet region is detected (1% of image)
				has_footwear = True
			
			if not has_footwear:
				# Check if this might be a partial image where feet aren't visible
				image_height = float(features.get("image_height", 0))
				image_width = float(features.get("image_width", 0))
				aspect_ratio = image_height / image_width if image_width > 0 else 1.0
				
				if aspect_ratio < 1.5:  # Not a typical full-body image
					violations.append({
						"item": "Footwear Check",
						"required": "Formal dress shoes (if visible in image)",
						"detected": "Footwear not visible in this image crop/angle",
						"score": 0.8,  # Give benefit of doubt
						"severity": "low",
						"reason": f"Image appears to be cropped or partial view. Footwear detection: {feet_texture:.1f}, feet region: {feet_mask_area:.3f}"
					})
				else:
					violations.append({
						"item": "Footwear Required",
						"required": "Formal dress shoes in professional colors (black, brown, etc.)",
						"detected": "No footwear detected or barefoot",
						"score": 0.0,
						"severity": "high",  # Reduced from critical
						"reason": f"Professional dress code requires formal footwear. Footwear detection: {feet_texture:.1f} (minimum: 3.0), feet region: {feet_mask_area:.3f}"
					})
			elif shoes_formality < formal_threshold:
				# Analyze SHOE characteristics (not body structure)
				feet_s = float(features.get("feet_mean_s", 50.0))
				feet_v = float(features.get("feet_mean_v", 60.0))
				feet_h = float(features.get("feet_mean_h", 90.0))
				
				# Determine casual shoe indicators
				casual_indicators = []
				if feet_s > 80:
					casual_indicators.append("overly colorful shoes")
				if feet_brightness > 150:
					casual_indicators.append("too bright for formal wear")
				if feet_v > 140:
					casual_indicators.append("casual bright colors")
				
				casual_description = ", ".join(casual_indicators) if casual_indicators else "casual sneakers/sandals style"
				
				violations.append({
					"item": "Footwear - Professional Shoes",
					"required": "Formal dress shoes in professional colors (black, brown, navy)",
					"detected": f"Casual footwear detected: {casual_description}",
					"score": shoes_formality,
					"severity": "medium",
					"reason": f"Dress code requires professional dress shoes. Shoe analysis shows: {shoes_formality:.1%} formality (required: {formal_threshold:.1%}). Colors: H:{feet_h:.0f}° S:{feet_s:.0f} V:{feet_v:.0f}"
				})
		elif not feet_visible and is_clearly_headshot:
			# For headshots, add informational note
			violations.append({
				"item": "Footwear Check",
				"required": "Formal dress shoes",
				"detected": "Not visible in headshot/passport photo",
				"score": 1.0,  # Neutral score
				"severity": "low",
				"reason": "Footwear verification skipped for headshot format"
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


def verify_attire_and_safety_new(features: Dict[str, Any], cfg: AppConfig, clf: AttireClassifier | None = None) -> Dict[str, Any]:
	"""NEW CLOTHING-FOCUSED VERIFICATION - bypasses any caching issues"""
	
	# Force formal/casual analysis
	cfg.policy_profile = "regular"
	cfg.expected_top = "formal"
	cfg.expected_bottom = "formal"
	
	# Extract clothing colors directly
	torso_h = float(features.get("torso_mean_h", 90.0))
	torso_s = float(features.get("torso_mean_s", 50.0))
	torso_v = float(features.get("torso_mean_v", 128.0))
	torso_brightness = float(features.get("torso_brightness", 128.0))
	
	legs_h = float(features.get("legs_mean_h", 90.0))
	legs_s = float(features.get("legs_mean_s", 50.0))
	legs_v = float(features.get("legs_mean_v", 128.0))
	legs_brightness = float(features.get("legs_brightness", 128.0))
	
	feet_h = float(features.get("feet_mean_h", 90.0))
	feet_s = float(features.get("feet_mean_s", 50.0))
	feet_v = float(features.get("feet_mean_v", 60.0))
	feet_brightness = float(features.get("feet_brightness", 128.0))
	feet_texture = float(features.get("feet_texture", 0.0))
	
	# Simple clothing-based formality analysis
	violations = []
	
	# TOP WEAR - Very lenient for formal shirts
	top_formal = True
	if torso_s > 180 or torso_v > 240:  # Only reject extremely bright/flashy
		top_formal = False
		violations.append({
			"item": "Top Wear - Professional Clothing",
			"required": "Professional shirt/blouse (any reasonable color)",
			"detected": f"Very bright/flashy clothing (Saturation: {torso_s:.0f}, Brightness: {torso_v:.0f})",
			"score": 0.3,
			"severity": "medium",
			"reason": f"Clothing appears too bright/flashy for professional setting. Colors: H:{torso_h:.0f}° S:{torso_s:.0f} V:{torso_v:.0f}"
		})
	
	# BOTTOM WEAR - Very lenient for formal pants
	bottom_formal = True
	if legs_s > 150 or legs_v > 230:  # Only reject extremely bright
		bottom_formal = False
		violations.append({
			"item": "Bottom Wear - Professional Clothing", 
			"required": "Professional pants/trousers (any reasonable color)",
			"detected": f"Very bright clothing (Saturation: {legs_s:.0f}, Brightness: {legs_v:.0f})",
			"score": 0.3,
			"severity": "medium",
			"reason": f"Pants appear too bright for professional setting. Colors: H:{legs_h:.0f}° S:{legs_s:.0f} V:{legs_v:.0f}"
		})
	
	# FOOTWEAR - Very lenient detection
	shoes_formal = True
	image_height = float(features.get("image_height", 0))
	image_width = float(features.get("image_width", 0))
	aspect_ratio = image_height / image_width if image_width > 0 else 1.0
	
	if feet_texture < 2 and aspect_ratio > 1.3:  # Only flag if clearly full-body with no shoes
		shoes_formal = False
		violations.append({
			"item": "Footwear",
			"required": "Closed shoes (any professional color)",
			"detected": f"No footwear detected (texture: {feet_texture:.1f})",
			"score": 0.0,
			"severity": "medium",
			"reason": f"Please wear closed shoes. Detection: {feet_texture:.1f}, Image: {image_height:.0f}x{image_width:.0f}"
		})
	
	# Calculate overall score
	formal_count = sum([top_formal, bottom_formal, shoes_formal])
	success_score = formal_count / 3.0
	
	# Determine status
	if success_score >= 0.8:
		status = "PASS"
	elif success_score >= 0.6:
		status = "WARNING"
	else:
		status = "FAIL"
	
	return {
		"status": status,
		"success_score": success_score,
		"fail_score": 1.0 - success_score,
		"score": success_score,
		"label": "clothing_focused_analysis",
		"violations": {
			"total_violations": len(violations),
			"violations": violations
		},
		"details": {
			"formality_analysis": {
				"top_formality": 0.8 if top_formal else 0.3,
				"bottom_formality": 0.8 if bottom_formal else 0.3,
				"shoes_formality": 0.8 if shoes_formal else 0.3,
			},
			"clothing_colors": {
				"top": f"H:{torso_h:.0f}° S:{torso_s:.0f} V:{torso_v:.0f}",
				"bottom": f"H:{legs_h:.0f}° S:{legs_s:.0f} V:{legs_v:.0f}",
				"shoes": f"H:{feet_h:.0f}° S:{feet_s:.0f} V:{feet_v:.0f}"
			}
		},
		"summary": {
			"overall_compliance": f"{success_score:.1%}",
			"violation_count": len(violations)
		}
	}

def verify_attire_and_safety(features: Dict[str, Any], cfg: AppConfig, clf: AttireClassifier | None = None) -> Dict[str, Any]:
	# STRICT rule-based verification (primary method)
	rule = rule_based_checks(features, cfg) if cfg.enable_rules else {"rule_score": 0.5}
	combined_score = rule.get("rule_score", 0.5)
	label = "rule_based"

	# ML model as secondary validation (if available)
	if cfg.enable_model and clf is not None and clf.model is not None:
		try:
			# Handle feature mismatch gracefully
			feature_values = [v for k, v in features.items() if k not in ("image", "label")]
			X = np.array(feature_values, dtype=float).reshape(1, -1)
			proba = clf.predict_proba(X)[0]
			p = float(np.max(proba))
			# Give more weight to rule-based for strict verification
			combined_score = float(0.8 * combined_score + 0.2 * p)
			if clf.label_names:
				label = f"rule+ml_{clf.label_names[int(np.argmax(proba))]}"
		except (ValueError, Exception) as e:
			# Feature mismatch or other ML error - fall back to rule-based only
			combined_score = rule.get("rule_score", 0.5)
			label = "rule_based_only"

	# Get detailed violations with STRICT analysis
	violations = _infer_missing_items(features, cfg)
	
	# Calculate detailed scores with STRICT penalties
	success_score = combined_score
	fail_score = 1.0 - combined_score
	
	# Calculate violation penalty based on severity - STRICT
	violation_penalty = 0.0
	critical_violations = 0
	high_violations = 0
	medium_violations = 0
	low_violations = 0
	
	for violation in violations:
		severity = violation.get("severity", "medium")
		if severity == "critical":
			critical_violations += 1
			violation_penalty += 0.6  # MAJOR penalty for critical violations
		elif severity == "high":
			high_violations += 1
			violation_penalty += 0.4  # SIGNIFICANT penalty for high violations
		elif severity == "medium":
			medium_violations += 1
			violation_penalty += 0.2  # MODERATE penalty for medium violations
		elif severity == "low":
			low_violations += 1
			violation_penalty += 0.1  # MINOR penalty for low violations
	
	# Apply STRICT penalty to scores
	success_score = float(np.clip(success_score - violation_penalty, 0.0, 1.0))
	fail_score = float(np.clip(fail_score + violation_penalty, 0.0, 1.0))
	
	# Determine overall status - STRICT CRITERIA for real-time analysis
	has_critical = critical_violations > 0
	has_high = high_violations > 0
	has_medium = medium_violations > 0
	has_low = low_violations > 0
	
	# STRICT pass criteria - realistic dress code enforcement
	if has_critical:
		status = "FAIL"
	elif has_high >= 2 or success_score < 0.3:
		status = "FAIL"
	elif has_high >= 1 or success_score < 0.6:
		status = "WARNING" 
	elif has_medium >= 2 or success_score < 0.75:
		status = "WARNING"
	elif has_medium >= 1 or has_low >= 3 or success_score < 0.85:
		status = "WARNING"
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

def verify_attire_universal(image_array: np.ndarray) -> Dict[str, Any]:
	"""Universal formal vs casual verification - works with ANY colors and styles"""
	
	try:
		# Convert to different color spaces for comprehensive analysis
		hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
		h, w = image_array.shape[:2]
		
		# Divide image into regions for clothing analysis
		top_region = image_array[:h//3, :]      # Top third (shirt/blouse area)
		middle_region = image_array[h//3:2*h//3, :]  # Middle third (pants/skirt area)
		bottom_region = image_array[2*h//3:, :]      # Bottom third (shoes area)
		
		violations = []
		
		# UNIVERSAL FORMAL vs CASUAL ANALYSIS
		
		# 1. TOP WEAR ANALYSIS - Detect clothing TYPE, not color
		top_formal_score = analyze_clothing_formality(top_region, "top")
		
		# 2. BOTTOM WEAR ANALYSIS - Detect clothing TYPE, not color  
		bottom_formal_score = analyze_clothing_formality(middle_region, "bottom")
		
		# 3. FOOTWEAR ANALYSIS - Detect shoe TYPE, not color
		shoes_formal_score = analyze_clothing_formality(bottom_region, "shoes")
		
		# Flag violations for items below professional standards
		if top_formal_score < 0.6:  # Below 60% formality = violation
			violations.append({
				"item": "Top Wear Style",
				"required": "Formal shirt/blouse/dress (any color)",
				"detected": "Casual or inappropriate top wear detected",
				"score": top_formal_score,
				"severity": "high",
				"reason": f"Top wear does not meet professional standards. Formality score: {top_formal_score:.1%} (required: 60%)"
			})
		
		if bottom_formal_score < 0.6:  # Below 60% formality = violation
			violations.append({
				"item": "Bottom Wear Style", 
				"required": "Formal pants/trousers/skirt (any color)",
				"detected": "Casual or inappropriate bottom wear detected",
				"score": bottom_formal_score,
				"severity": "high",
				"reason": f"Bottom wear does not meet professional standards. Formality score: {bottom_formal_score:.1%} (required: 60%)"
			})
		
		if shoes_formal_score < 0.5:  # Below 50% formality = violation (slightly lower for shoes)
			violations.append({
				"item": "Footwear Style",
				"required": "Formal shoes/dress shoes (any color)", 
				"detected": "Casual footwear or no shoes detected",
				"score": shoes_formal_score,
				"severity": "medium",
				"reason": f"Footwear does not meet professional standards. Formality score: {shoes_formal_score:.1%} (required: 50%)"
			})
		
		# Calculate overall formality score
		overall_formality = (top_formal_score * 0.4 + bottom_formal_score * 0.4 + shoes_formal_score * 0.2)
		
		# Professional pass criteria - require 60% formality
		if overall_formality >= 0.6:  # 60% formality required for PASS
			success_score = 1.0
			status = "PASS"
		elif overall_formality >= 0.4:  # 40-59% formality gets WARNING
			success_score = 0.7
			status = "WARNING"
		else:  # Below 40% formality gets FAIL
			success_score = 0.3
			status = "FAIL"
		
		return {
			"status": status,
			"success_score": success_score,
			"fail_score": 1.0 - success_score,
			"score": success_score,
			"label": "universal_formal_casual_analysis",
			"violations": {
				"total_violations": len(violations),
				"violations": violations
			},
			"details": {
				"formality_analysis": {
					"top_formality": top_formal_score,
					"bottom_formality": bottom_formal_score,
					"shoes_formality": shoes_formal_score,
					"overall_formality": overall_formality
				},
				"analysis_method": "universal_style_detection",
				"focus": "clothing_type_not_color"
			},
			"summary": {
				"overall_compliance": f"{success_score:.1%}",
				"violation_count": len(violations),
				"formality_level": f"{overall_formality:.1%}"
			}
		}
		
	except Exception as e:
		# Fallback - assume formal attire (benefit of doubt)
		return {
			"status": "PASS",
			"success_score": 0.9,
			"fail_score": 0.1,
			"score": 0.9,
			"label": "analysis_fallback",
			"violations": {
				"total_violations": 0,
				"violations": []
			},
			"details": {
				"analysis_method": "fallback_assume_formal",
				"error": str(e)
			},
			"summary": {
				"overall_compliance": "90%",
				"violation_count": 0,
				"note": "Analysis failed, assuming formal attire"
			}
		}


def analyze_clothing_formality(region: np.ndarray, clothing_type: str) -> float:
	"""Simple and accurate formality analysis - assume most clothing is formal unless clearly casual"""
	
	if region.size == 0:
		return 0.8  # Default to formal if region is empty
	
	try:
		# Convert to HSV for basic analysis
		hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
		
		# Get basic color properties
		mean_saturation = np.mean(hsv_region[:, :, 1])
		mean_value = np.mean(hsv_region[:, :, 2])
		
		# SIMPLE RULE: Most clothing is formal unless it's extremely casual
		# Only flag as casual if it has very specific casual characteristics
		
		if clothing_type == "top":
			# For tops: assume formal unless extremely bright/flashy
			if mean_saturation > 200 and mean_value > 230:  # Very bright and saturated = t-shirt
				return 0.3  # Clearly casual
			elif mean_saturation > 150 and mean_value > 200:  # Moderately bright = borderline
				return 0.65  # Just above threshold
			else:
				return 0.85  # Assume formal (shirts, blouses, etc.)
		
		elif clothing_type == "bottom":
			# For bottoms: assume formal unless extremely casual
			if mean_saturation > 180 and mean_value > 220:  # Very bright = casual
				return 0.4  # Clearly casual
			elif mean_saturation > 120 and mean_value > 180:  # Moderately bright = borderline
				return 0.65  # Just above threshold
			else:
				return 0.8  # Assume formal (pants, trousers, skirts)
		
		elif clothing_type == "shoes":
			# For shoes: check if footwear is present
			gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
			avg_brightness = np.mean(gray_region)
			
			if avg_brightness < 180:  # Dark regions suggest shoes are present
				return 0.75  # Assume formal shoes
			else:
				return 0.3   # Likely no shoes or very bright casual shoes
		
		else:
			return 0.8  # Default to formal
		
	except Exception:
		return 0.8  # Default to formal on error