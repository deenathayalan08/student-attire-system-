from typing import Any, Dict, Optional
import numpy as np
import cv2

try:
	import mediapipe as mp
	_mp_ok = True
except Exception:
	_mp_ok = False


_POSE = None


def _get_pose():
	global _POSE
	if _POSE is None and _mp_ok:
		_POSE = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False)
	return _POSE


def extract_pose(bgr_image: np.ndarray) -> Optional[Any]:
	if not _mp_ok:
		return None
	pose = _get_pose()
	rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
	res = pose.process(rgb)
	return res.pose_landmarks if res and res.pose_landmarks else None


def _region_mask_from_pose(image: np.ndarray, landmarks) -> Dict[str, np.ndarray]:
	h, w = image.shape[:2]
	mask = {
		"torso": np.zeros((h, w), dtype=np.uint8),
		"legs": np.zeros((h, w), dtype=np.uint8),
		"feet": np.zeros((h, w), dtype=np.uint8),
	}
	
	# Determine if this looks like a full-body image based on aspect ratio
	# Full-body images typically have height > width (portrait orientation)
	aspect_ratio = h / w if w > 0 else 1.0
	is_likely_full_body = aspect_ratio > 1.2  # Portrait orientation suggests full body
	
	if landmarks is None:
		# fallback to center bands - but only for torso if it looks like a headshot
		mask["torso"][int(0.2*h):int(0.5*h), int(0.25*w):int(0.75*w)] = 1
		
		# If aspect ratio suggests full-body, create fallback leg/feet masks
		# This handles cases where pose detection fails but image is full-body
		if is_likely_full_body:
			# Create leg mask in lower middle portion (typical leg region)
			mask["legs"][int(0.5*h):int(0.85*h), int(0.25*w):int(0.75*w)] = 1
			# Create feet mask at bottom (typical feet region)
			mask["feet"][int(0.85*h):h, int(0.25*w):int(0.75*w)] = 1
		return mask

	# Use relative y-bands guided by shoulders and hips when available
	# Landmarks indices: 11 (left shoulder), 12 (right shoulder), 23 (left hip), 24 (right hip), 27 (left foot index), 28 (right foot index)
	pts = [(lm.x * w, lm.y * h) for lm in landmarks.landmark]
	shoulder_y = np.mean([pts[11][1], pts[12][1]]) if len(pts) > 12 else 0.3*h
	hip_y = np.mean([pts[23][1], pts[24][1]]) if len(pts) > 24 else 0.55*h
	foot_y = np.mean([pts[27][1], pts[28][1]]) if len(pts) > 28 else 0.93*h

	x1, x2 = int(0.25*w), int(0.75*w)
	mask["torso"][int(shoulder_y):int(hip_y), x1:x2] = 1
	
	# Improved leg/feet mask creation - more lenient thresholds
	# Check if hips are detected - use more lenient threshold (0.85*h instead of 0.7*h)
	if len(pts) > 24:
		# If hips are detected anywhere in the image (not just upper portion), create leg mask
		# Use hip position if reasonable, otherwise use fallback
		if hip_y < 0.85*h:  # More lenient - hips can be lower in image
			leg_start = int(hip_y) if hip_y > 0.4*h else int(0.5*h)
			leg_end = int(0.9*h) if len(pts) <= 28 else min(h, int(foot_y))
			mask["legs"][leg_start:leg_end, x1:x2] = 1
		else:
			# Hips too low, but still try to create leg mask in middle-lower region
			mask["legs"][int(0.5*h):int(0.85*h), x1:x2] = 1
		
		# Feet mask - check if feet landmarks exist
		if len(pts) > 28:
			# Feet detected - use foot position
			if foot_y < 0.98*h:  # More lenient - feet can be very close to bottom
				foot_start = max(int(0.85*h), int(foot_y) - 20)
				mask["feet"][foot_start:min(h, int(foot_y)+10), x1:x2] = 1
			else:
				# Feet detected but at bottom edge - still create mask
				mask["feet"][int(0.9*h):h, x1:x2] = 1
		else:
			# No feet landmarks, but if we have legs and image looks full-body, create feet mask
			if is_likely_full_body and len(pts) > 24:
				mask["feet"][int(0.9*h):h, x1:x2] = 1
	elif is_likely_full_body:
		# No pose landmarks but image looks full-body - create fallback masks
		mask["legs"][int(0.5*h):int(0.85*h), int(0.25*w):int(0.75*w)] = 1
		mask["feet"][int(0.85*h):h, int(0.25*w):int(0.75*w)] = 1
	
	return mask


def _histogram_features(bgr: np.ndarray, mask: np.ndarray, bins: int) -> np.ndarray:
	hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], mask, [bins, max(2, bins//3), max(2, bins//3)], [0, 180, 0, 256, 0, 256])
	hist = cv2.normalize(hist, None).flatten()
	return hist


def extract_features_from_image(bgr_image: np.ndarray, pose_landmarks=None, bins: int = 24) -> Dict[str, Any]:
	masks = _region_mask_from_pose(bgr_image, pose_landmarks)
	features: Dict[str, Any] = {}
	
	# Histogram features for each region
	for region, m in masks.items():
		hist = _histogram_features(bgr_image, (m*255).astype(np.uint8), bins)
		for i, v in enumerate(hist):
			features[f"{region}_h{i}"] = float(v)

	# Enhanced color and texture analysis per region
	hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
	gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
	h, w = bgr_image.shape[:2]
	total_pixels = h * w
	
	for region, m in masks.items():
		masked_hsv = hsv[m.astype(bool)]
		masked_gray = gray[m.astype(bool)]
		
		# Check if mask has any pixels
		mask_pixel_count = np.sum(m.astype(bool))
		mask_area_ratio = mask_pixel_count / total_pixels if total_pixels > 0 else 0.0
		
		if masked_hsv.size == 0 or mask_pixel_count == 0:
			# Default values when no pixels in region
			features[f"{region}_mean_h"] = 0.0
			features[f"{region}_mean_s"] = 0.0
			features[f"{region}_mean_v"] = 0.0
			features[f"{region}_std_h"] = 0.0
			features[f"{region}_std_v"] = 0.0
			features[f"{region}_brightness"] = 0.0
			features[f"{region}_contrast"] = 0.0
			features[f"{region}_texture"] = 0.0
			features[f"{region}_mask_area"] = 0.0  # Track mask area for visibility check
		else:
			# Color statistics
			features[f"{region}_mean_h"] = float(np.mean(masked_hsv[:, 0]))
			features[f"{region}_mean_s"] = float(np.mean(masked_hsv[:, 1]))
			features[f"{region}_mean_v"] = float(np.mean(masked_hsv[:, 2]))
			features[f"{region}_std_h"] = float(np.std(masked_hsv[:, 0]))
			features[f"{region}_std_v"] = float(np.std(masked_hsv[:, 2]))
			
			# Brightness and contrast
			features[f"{region}_brightness"] = float(np.mean(masked_gray))
			features[f"{region}_contrast"] = float(np.std(masked_gray))
			
			# Texture analysis using Laplacian variance
			region_gray = gray.copy()
			region_gray[~m.astype(bool)] = 0
			laplacian = cv2.Laplacian(region_gray, cv2.CV_64F)
			features[f"{region}_texture"] = float(np.var(laplacian[m.astype(bool)]))
			
			# Track mask area ratio for visibility determination
			features[f"{region}_mask_area"] = float(mask_area_ratio)
	
	# Additional global features for better detection
	features["image_brightness"] = float(np.mean(gray))
	features["image_contrast"] = float(np.std(gray))
	features["image_height"] = float(h)
	features["image_width"] = float(w)
	
	# Detect if pose landmarks indicate full-body image
	# If we have hip and ankle/feet landmarks, it's definitely a full-body image
	has_full_body_pose = False
	if pose_landmarks is not None:
		pts = [(lm.x * w, lm.y * h) for lm in pose_landmarks.landmark]
		# Check if we have hips (23, 24) and ankles/feet (27, 28) - indicates full body
		if len(pts) > 28:
			# Has all landmarks including feet - definitely full body
			has_full_body_pose = True
		elif len(pts) > 24:
			# Has hips but maybe not feet - still likely full body
			has_full_body_pose = True
	features["has_full_body_pose"] = float(1.0 if has_full_body_pose else 0.0)
	
	# Pants length detection using pose landmarks
	# (h, w already defined above)
	if pose_landmarks is not None:
		pts = [(lm.x * w, lm.y * h) for lm in pose_landmarks.landmark]
		if len(pts) > 28:  # Ensure we have enough landmarks
			# Calculate pants length based on hip to ankle distance
			left_hip = pts[23]
			right_hip = pts[24]
			left_ankle = pts[27]
			right_ankle = pts[28]
			
			# Average hip and ankle positions
			hip_y = (left_hip[1] + right_hip[1]) / 2
			ankle_y = (left_ankle[1] + right_ankle[1]) / 2
			
			# Calculate pants length as percentage of body height
			# Use nose (0) and ankle (27/28) for total body height
			nose_y = pts[0][1]
			total_height = ankle_y - nose_y
			pants_length = ankle_y - hip_y
			
			if total_height > 0:
				pants_length_ratio = pants_length / total_height
				features["pants_length_ratio"] = float(pants_length_ratio)
				
				# Determine if pants are appropriate length (between 30-50% of body height)
				features["pants_length_appropriate"] = float(0.3 <= pants_length_ratio <= 0.5)
			else:
				features["pants_length_ratio"] = 0.0
				features["pants_length_appropriate"] = 0.0
		else:
			features["pants_length_ratio"] = 0.0
			features["pants_length_appropriate"] = 0.0
	else:
		features["pants_length_ratio"] = 0.0
		features["pants_length_appropriate"] = 0.0
	
	# Edge density (useful for detecting structured clothing vs skin)
	edges = cv2.Canny(gray, 50, 150)
	features["edge_density"] = float(np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]))
	
	# ID Card detection
	try:
		from .id_card_detector import detect_id_card
		id_card_result = detect_id_card(bgr_image)
		features["id_card_detected"] = float(id_card_result["detected"])
		features["id_card_confidence"] = float(id_card_result["confidence"])
		features["id_card_area"] = float(id_card_result["area"])
	except ImportError:
		# Fallback if id_card_detector is not available
		features["id_card_detected"] = 0.0
		features["id_card_confidence"] = 0.0
		features["id_card_area"] = 0.0
	
	return features
