import cv2
import numpy as np
from typing import Dict, List, Any

try:
	import mediapipe as mp
	_mp_ok = True
except Exception:
	_mp_ok = False


def draw_pose_annotations(bgr: np.ndarray, landmarks) -> np.ndarray:
	img = bgr
	if not _mp_ok or landmarks is None:
		return img

	h, w = img.shape[:2]
	drawing = mp.solutions.drawing_utils
	style = mp.solutions.drawing_styles
	drawing.draw_landmarks(
		img,
		landmarks,
		mp.solutions.pose.POSE_CONNECTIONS,
		landmark_drawing_spec=style.get_default_pose_landmarks_style(),
	)
	return img


def overlay_badge(bgr: np.ndarray, text: str, ok: bool = True) -> np.ndarray:
	img = bgr.copy()
	color = (46, 125, 50) if ok else (30, 30, 200)
	cv2.rectangle(img, (10, 10), (10 + 260, 60), color, thickness=-1)
	cv2.putText(img, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
	return img


def draw_violation_indicators(bgr: np.ndarray, landmarks, violations: List[Dict[str, Any]]) -> np.ndarray:
	"""Draw visual indicators on the image showing where violations occur"""
	img = bgr.copy()
	h, w = img.shape[:2]
	
	if landmarks is None or not violations:
		return img
	
	# Get landmark points
	pts = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
	
	# Define regions based on landmarks
	regions = {
		"torso": {
			"center": (int((pts[11][0] + pts[12][0]) / 2), int((pts[11][1] + pts[12][1]) / 2)) if len(pts) > 12 else (w//2, h//3),
			"color": (0, 255, 255)  # Yellow
		},
		"legs": {
			"center": (int((pts[23][0] + pts[24][0]) / 2), int((pts[23][1] + pts[24][1]) / 2)) if len(pts) > 24 else (w//2, int(h*0.6)),
			"color": (0, 255, 0)  # Green
		},
		"feet": {
			"center": (int((pts[27][0] + pts[28][0]) / 2), int((pts[27][1] + pts[28][1]) / 2)) if len(pts) > 28 else (w//2, int(h*0.9)),
			"color": (255, 0, 0)  # Red
		}
	}
	
	# Draw violation indicators
	for violation in violations:
		item = violation.get("item", "").lower()
		severity = violation.get("severity", "medium")
		score = violation.get("score", 0.0)
		
		# Determine region and color based on violation type
		region_key = None
		if "top" in item or "coat" in item or "sports" in item:
			region_key = "torso"
		elif "bottom" in item or "trousers" in item or "skirt" in item or "pants" in item:
			region_key = "legs"
		elif "foot" in item or "shoe" in item:
			region_key = "feet"
		elif "id" in item.lower() or "card" in item.lower():
			region_key = "torso"  # ID cards are typically worn on torso/chest area
		
		if region_key and region_key in regions:
			center = regions[region_key]["center"]
			
			# Color based on severity
			if severity == "critical":
				color = (0, 0, 255)  # Red
				thickness = 4
			elif severity == "high":
				color = (0, 165, 255)  # Orange
				thickness = 3
			elif severity == "medium":
				color = (0, 255, 255)  # Yellow
				thickness = 2
			else:
				color = (128, 128, 128)  # Gray
				thickness = 2
			
			# Draw circle around the region
			radius = int(30 + (1.0 - score) * 20)  # Larger circle for worse scores
			cv2.circle(img, center, radius, color, thickness)
			
			# Draw X mark for violations
			if score < 0.5:  # Only show X for significant violations
				cv2.drawMarker(img, center, color, cv2.MARKER_TILTED_CROSS, 20, thickness)
			
			# Add text label
			label = f"{severity.upper()}: {int(score*100)}%"
			cv2.putText(img, label, (center[0] - 40, center[1] - radius - 10), 
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
	
	return img


def overlay_detailed_badge(bgr: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
	"""Create a detailed badge showing success/fail scores and violation summary"""
	img = bgr.copy()
	h, w = img.shape[:2]
	
	# Determine overall color based on status
	status = result.get("status", "UNKNOWN")
	if status == "PASS":
		color = (46, 125, 50)  # Green
	elif status == "WARNING":
		color = (0, 165, 255)  # Orange
	else:  # FAIL
		color = (30, 30, 200)  # Red
	
	# Create badge background
	badge_height = 120
	cv2.rectangle(img, (10, 10), (w - 10, badge_height), color, thickness=-1)
	
	# Add border
	cv2.rectangle(img, (10, 10), (w - 10, badge_height), (255, 255, 255), 2)
	
	# Main status text
	status_text = f"STATUS: {status}"
	cv2.putText(img, status_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
	
	# Success/Fail scores
	success_score = result.get("success_score", 0.0)
	fail_score = result.get("fail_score", 0.0)
	score_text = f"SUCCESS: {success_score:.1%} | FAIL: {fail_score:.1%}"
	cv2.putText(img, score_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
	
	# Violation summary
	violations = result.get("violations", {})
	total_violations = violations.get("total_violations", 0)
	if total_violations > 0:
		violation_text = f"VIOLATIONS: {total_violations} (C:{violations.get('critical', 0)} H:{violations.get('high', 0)} M:{violations.get('medium', 0)})"
		cv2.putText(img, violation_text, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	else:
		cv2.putText(img, "NO VIOLATIONS DETECTED", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
	
	return img
