"""
ID Card Detection Module

This module provides functionality to detect ID cards in images using computer vision techniques.
It looks for rectangular objects that could be ID cards, lanyards, or badges.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import math


def detect_rectangular_objects(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect rectangular objects in the image that could be ID cards or badges.
    
    Args:
        image: Input BGR image
        
    Returns:
        List of detected rectangular objects with properties
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_objects = []
    
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly rectangular (4 corners)
        if len(approx) >= 4:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            rect_area = w * h
            
            # Filter by size and aspect ratio
            if area > 500 and rect_area > 1000:  # Minimum size
                aspect_ratio = w / h
                
                # ID cards typically have aspect ratios between 1.4 and 1.8
                if 1.2 <= aspect_ratio <= 2.0:
                    # Calculate contour properties
                    solidity = area / rect_area
                    
                    # Check if it's roughly rectangular
                    if solidity > 0.7:
                        detected_objects.append({
                            'contour': contour,
                            'bbox': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'solidity': solidity,
                            'center': (x + w//2, y + h//2),
                            'detection_method': 'rectangular'
                        })
    
    return detected_objects


def detect_id_card_like_objects(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect objects that look like ID cards using multiple techniques.
    
    Args:
        image: Input BGR image
        
    Returns:
        List of detected ID card candidates
    """
    candidates = []
    
    # Method 1: Rectangular object detection
    rectangular_objects = detect_rectangular_objects(image)
    candidates.extend(rectangular_objects)
    
    # Method 2: Color-based detection (look for typical ID card colors)
    candidates.extend(detect_by_color_patterns(image))
    
    # Method 3: Edge-based detection
    candidates.extend(detect_by_edges(image))
    
    # Remove duplicates and rank by confidence
    ranked_candidates = rank_id_card_candidates(candidates, image)
    
    return ranked_candidates


def detect_by_color_patterns(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect ID cards by looking for typical color patterns.
    """
    candidates = []
    
    # Convert to HSV for better color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define color ranges for typical ID card backgrounds
    # White/light gray backgrounds
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Blue backgrounds (common in school IDs)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Red backgrounds
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    combined_mask = white_mask + blue_mask + red_mask
    
    # Find contours in the combined mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800:  # Minimum size
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            if 1.0 <= aspect_ratio <= 2.5:  # Reasonable aspect ratio
                candidates.append({
                    'contour': contour,
                    'bbox': (x, y, w, h),
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center': (x + w//2, y + h//2),
                    'detection_method': 'color_pattern'
                })
    
    return candidates


def detect_by_edges(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect ID cards using edge detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            # Approximate the contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) >= 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if 1.2 <= aspect_ratio <= 2.0:
                    candidates.append({
                        'contour': contour,
                        'bbox': (x, y, w, h),
                        'area': area,
                        'aspect_ratio': aspect_ratio,
                        'center': (x + w//2, y + h//2),
                        'detection_method': 'edge_detection'
                    })
    
    return candidates


def rank_id_card_candidates(candidates: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Rank ID card candidates by confidence score.
    """
    h, w = image.shape[:2]
    image_area = h * w
    
    for candidate in candidates:
        confidence = 0.0
        
        # Size score (prefer medium-sized objects)
        area_ratio = candidate['area'] / image_area
        if 0.005 <= area_ratio <= 0.1:  # 0.5% to 10% of image
            confidence += 0.3
        
        # Aspect ratio score
        aspect_ratio = candidate['aspect_ratio']
        if 1.4 <= aspect_ratio <= 1.8:  # Typical ID card ratio
            confidence += 0.3
        elif 1.2 <= aspect_ratio <= 2.0:
            confidence += 0.2
        
        # Position score (prefer objects in upper body area)
        center_y = candidate['center'][1]
        if center_y < h * 0.6:  # Upper 60% of image
            confidence += 0.2
        
        # Method bonus
        if candidate.get('detection_method') == 'rectangular':
            confidence += 0.1
        
        candidate['confidence'] = min(confidence, 1.0)
    
    # Sort by confidence and remove duplicates
    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Remove overlapping candidates
    filtered_candidates = []
    for candidate in candidates:
        is_duplicate = False
        for existing in filtered_candidates:
            # Check if bounding boxes overlap significantly
            if bbox_overlap(candidate['bbox'], existing['bbox']) > 0.5:
                is_duplicate = True
                break
        
        if not is_duplicate:
            filtered_candidates.append(candidate)
    
    return filtered_candidates


def bbox_overlap(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate overlap ratio between two bounding boxes.
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    union_area = w1 * h1 + w2 * h2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def detect_id_card(image: np.ndarray, confidence_threshold: float = 0.6) -> Dict[str, Any]:
    """
    Main function to detect ID cards in an image.
    
    Args:
        image: Input BGR image
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        Dictionary with detection results
    """
    candidates = detect_id_card_like_objects(image)
    
    # Find the best candidate above threshold
    best_candidate = None
    for candidate in candidates:
        if candidate['confidence'] >= confidence_threshold:
            best_candidate = candidate
            break
    
    if best_candidate:
        return {
            'detected': True,
            'confidence': best_candidate['confidence'],
            'bbox': best_candidate['bbox'],
            'center': best_candidate['center'],
            'area': best_candidate['area'],
            'aspect_ratio': best_candidate['aspect_ratio']
        }
    else:
        return {
            'detected': False,
            'confidence': 0.0,
            'bbox': None,
            'center': None,
            'area': 0,
            'aspect_ratio': 0
        }


def draw_id_card_detection(image: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
    """
    Draw ID card detection results on the image.
    
    Args:
        image: Input BGR image
        detection_result: Result from detect_id_card()
        
    Returns:
        Annotated image
    """
    img = image.copy()
    
    if detection_result['detected']:
        bbox = detection_result['bbox']
        confidence = detection_result['confidence']
        
        # Draw bounding box
        x, y, w, h = bbox
        color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)  # Green or yellow
        thickness = 2
        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        
        # Draw confidence text
        text = f"ID Card: {confidence:.1%}"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw center point
        center = detection_result['center']
        cv2.circle(img, center, 5, color, -1)
    else:
        # Draw "No ID Card" text
        h, w = img.shape[:2]
        cv2.putText(img, "No ID Card Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return img
