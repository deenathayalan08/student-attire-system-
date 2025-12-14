"""
Enhanced Object Detection Module
Uses color-based and shape-based detection for:
- Shoes (black/dark footwear)
- ID Cards (rectangular objects with specific aspect ratio)
- Chains/Lanyards (thin vertical lines in neck region)
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List, Any, Optional


class ShoeDetector:
    """Detect shoes using color-based analysis (HSV)"""
    
    def __init__(self):
        # Black shoe detection: low saturation, low-medium value
        self.black_lower = np.array([0, 0, 0])
        self.black_upper = np.array([180, 50, 100])  # Stricter: very low saturation = black/gray
        
        # Dark shoe detection: any hue, low value
        self.dark_lower = np.array([0, 0, 0])
        self.dark_upper = np.array([180, 255, 80])
        
        # Minimum area for shoe detection (pixels)
        self.min_area = 500
    
    def detect_shoes(self, image: np.ndarray, feet_region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Detect shoes in the feet region using color analysis.
        
        Args:
            image: BGR image
            feet_region: (x, y, w, h) bounding box for feet region
            
        Returns:
            Dictionary with detection results
        """
        h, w = image.shape[:2]
        
        # Default feet region: bottom 15% of image
        if feet_region is None:
            feet_region = (0, int(h * 0.85), w, int(h * 0.15))
        
        x, y, fw, fh = feet_region
        feet_roi = image[y:y+fh, x:x+fw]
        
        if feet_roi.size == 0:
            return {
                'detected': False,
                'is_black': False,
                'confidence': 0.0,
                'reason': 'No feet region detected'
            }
        
        # Convert to HSV
        hsv = cv2.cvtColor(feet_roi, cv2.COLOR_BGR2HSV)
        
        # Detect black shoes (low saturation)
        black_mask = cv2.inRange(hsv, self.black_lower, self.black_upper)
        black_area = np.sum(black_mask > 0)
        
        # Detect dark shoes (low value)
        dark_mask = cv2.inRange(hsv, self.dark_lower, self.dark_upper)
        dark_area = np.sum(dark_mask > 0)
        
        # Calculate brightness and saturation
        # Only consider pixels with reasonable value (not pure black background)
        valid_pixels = hsv[hsv[:, :, 2] > 30]  # Filter out very dark pixels
        if len(valid_pixels) > 0:
            avg_saturation = np.mean(valid_pixels[:, 1])
            avg_value = np.mean(valid_pixels[:, 2])
        else:
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
        
        # Detect skin tone (high saturation in orange/yellow range = bare feet)
        skin_lower = np.array([0, 30, 80])
        skin_upper = np.array([30, 200, 255])
        skin_mask = cv2.inRange(hsv, skin_lower, skin_upper)
        skin_area = np.sum(skin_mask > 0)
        
        total_area = feet_roi.shape[0] * feet_roi.shape[1]
        
        # Decision logic
        shoes_detected = False
        is_black = False
        confidence = 0.0
        reason = ""
        
        # Check for bare feet (skin tone)
        if skin_area > total_area * 0.3:
            shoes_detected = False
            confidence = 0.0
            reason = "Bare feet detected (skin tone present)"
        
        # Check for shoes (dark/black areas OR high saturation colored shoes)
        elif black_area > self.min_area or dark_area > self.min_area:
            shoes_detected = True
            
            # Determine if black shoes - STRICTER THRESHOLD
            # Black shoes must have very low saturation (< 40)
            if avg_saturation < 40:  # Stricter: very low saturation = black/gray
                is_black = True
                confidence = min(1.0, (40 - avg_saturation) / 40)
                reason = f"Black shoes detected (low saturation: {avg_saturation:.1f})"
            else:
                is_black = False
                confidence = min(1.0, dark_area / total_area * 2)
                reason = f"Colored shoes detected (saturation: {avg_saturation:.1f})"
        
        # Also check for colored shoes (high saturation, even if not dark)
        elif avg_saturation > 80:  # High saturation = colored shoes
            shoes_detected = True
            is_black = False
            confidence = min(1.0, avg_saturation / 255.0)
            reason = f"Colored shoes detected (high saturation: {avg_saturation:.1f})"
        
        else:
            shoes_detected = False
            confidence = 0.0
            reason = "No shoes detected (insufficient dark area and low saturation)"
        
        return {
            'detected': shoes_detected,
            'is_black': is_black,
            'confidence': confidence,
            'avg_saturation': float(avg_saturation),
            'avg_value': float(avg_value),
            'black_area': int(black_area),
            'dark_area': int(dark_area),
            'skin_area': int(skin_area),
            'reason': reason
        }


class IDCardDetector:
    """Detect ID cards using shape-based analysis"""
    
    def __init__(self):
        # ID card typical aspect ratio: 1.4 to 1.8 (width/height)
        self.min_aspect_ratio = 1.2
        self.max_aspect_ratio = 2.0
        
        # Size constraints (percentage of image)
        self.min_area_ratio = 0.005  # 0.5% of image
        self.max_area_ratio = 0.15   # 15% of image
        
        # Minimum contour area
        self.min_contour_area = 800
    
    def detect_id_card(self, image: np.ndarray, torso_region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Detect ID card using shape and color analysis.
        
        Args:
            image: BGR image
            torso_region: (x, y, w, h) bounding box for torso region
            
        Returns:
            Dictionary with detection results
        """
        h, w = image.shape[:2]
        
        # Default torso region: upper-middle 40% of image
        if torso_region is None:
            torso_region = (int(w * 0.25), int(h * 0.2), int(w * 0.5), int(h * 0.4))
        
        x, y, tw, th = torso_region
        torso_roi = image[y:y+th, x:x+tw]
        
        if torso_roi.size == 0:
            return {
                'detected': False,
                'confidence': 0.0,
                'bbox': None,
                'reason': 'No torso region detected'
            }
        
        # Convert to grayscale
        gray = cv2.cvtColor(torso_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply multiple edge detection methods for better ID card detection
        # Method 1: Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Method 2: Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Combine both methods
        combined = cv2.bitwise_or(thresh, edges)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        image_area = h * w
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area < self.min_contour_area:
                continue
            
            # Get bounding rectangle
            cx, cy, cw, ch = cv2.boundingRect(contour)
            
            # Check aspect ratio
            if ch == 0:
                continue
            aspect_ratio = cw / ch
            
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            # Check size relative to image
            area_ratio = area / image_area
            if not (self.min_area_ratio <= area_ratio <= self.max_area_ratio):
                continue
            
            # Calculate rectangularity (how rectangular is the contour)
            rect_area = cw * ch
            solidity = area / rect_area if rect_area > 0 else 0
            
            if solidity < 0.7:  # Must be fairly rectangular
                continue
            
            # Calculate confidence based on multiple factors
            confidence = 0.0
            
            # Aspect ratio score (closer to 1.6 = typical ID card)
            aspect_score = 1.0 - abs(aspect_ratio - 1.6) / 0.4
            confidence += aspect_score * 0.3
            
            # Size score (prefer medium-sized objects)
            if 0.01 <= area_ratio <= 0.08:
                confidence += 0.3
            elif 0.005 <= area_ratio <= 0.15:
                confidence += 0.2
            
            # Rectangularity score
            confidence += solidity * 0.2
            
            # Position score (prefer upper torso)
            if cy < th * 0.6:
                confidence += 0.2
            
            confidence = min(1.0, confidence)
            
            # Adjust coordinates to full image
            candidates.append({
                'bbox': (x + cx, y + cy, cw, ch),
                'confidence': confidence,
                'aspect_ratio': aspect_ratio,
                'area': area,
                'solidity': solidity
            })
        
        # Sort by confidence
        candidates.sort(key=lambda c: c['confidence'], reverse=True)
        
        if candidates:
            best = candidates[0]
            return {
                'detected': True,
                'confidence': best['confidence'],
                'bbox': best['bbox'],
                'aspect_ratio': best['aspect_ratio'],
                'area': best['area'],
                'reason': f"ID card detected with {best['confidence']:.1%} confidence"
            }
        else:
            return {
                'detected': False,
                'confidence': 0.0,
                'bbox': None,
                'reason': 'No rectangular object matching ID card criteria found'
            }


class ChainLanyardDetector:
    """Detect chains/lanyards in neck region using edge detection"""
    
    def __init__(self):
        # Minimum line length for chain detection
        self.min_line_length = 30
        self.max_line_gap = 10
    
    def detect_chain(self, image: np.ndarray, neck_region: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """
        Detect chain/lanyard in neck region using line detection.
        
        Args:
            image: BGR image
            neck_region: (x, y, w, h) bounding box for neck region
            
        Returns:
            Dictionary with detection results
        """
        h, w = image.shape[:2]
        
        # Default neck region: upper-middle area
        if neck_region is None:
            neck_region = (int(w * 0.35), int(h * 0.15), int(w * 0.3), int(h * 0.25))
        
        x, y, nw, nh = neck_region
        neck_roi = image[y:y+nh, x:x+nw]
        
        if neck_roi.size == 0:
            return {
                'detected': False,
                'confidence': 0.0,
                'reason': 'No neck region detected'
            }
        
        # Convert to grayscale
        gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Detect lines using Hough transform - STRICTER PARAMETERS
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=40,  # Increased from 30 - need stronger lines
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None:
            return {
                'detected': False,
                'confidence': 0.0,
                'line_count': 0,
                'reason': 'No lines detected in neck region'
            }
        
        # Filter for vertical/near-vertical lines (chains hang vertically)
        # STRICTER: chains must be very vertical (70-90 degrees)
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            if x2 - x1 == 0:
                angle = 90
            else:
                angle = abs(np.degrees(np.arctan((y2 - y1) / (x2 - x1))))
            
            # Check if line is vertical (70-90 degrees) - STRICTER
            if 70 <= angle <= 90:
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                # Only accept lines that are reasonably long (at least 40% of min_line_length)
                if length >= self.min_line_length * 0.8:
                    vertical_lines.append({
                        'coords': (x1, y1, x2, y2),
                        'angle': angle,
                        'length': length
                    })
        
        # Calculate confidence based on vertical lines - STRICTER THRESHOLD
        # Need at least 2 strong vertical lines to detect chain
        if len(vertical_lines) >= 2:
            # More vertical lines = higher confidence
            line_count = len(vertical_lines)
            avg_length = np.mean([l['length'] for l in vertical_lines])
            
            # Require longer lines for higher confidence
            length_score = min(1.0, avg_length / (nh * 0.5))  # Lines should span at least 50% of neck region
            count_score = min(1.0, line_count / 5.0)
            
            confidence = length_score * 0.6 + count_score * 0.4
            
            return {
                'detected': True,
                'confidence': confidence,
                'line_count': line_count,
                'avg_length': float(avg_length),
                'reason': f"Chain/lanyard detected ({line_count} vertical lines)"
            }
        else:
            return {
                'detected': False,
                'confidence': 0.0,
                'line_count': len(vertical_lines),
                'reason': f'Insufficient vertical lines for chain detection (found {len(vertical_lines)}, need 2+)'
            }


def detect_all_objects(image: np.ndarray, pose_landmarks=None) -> Dict[str, Any]:
    """
    Detect all objects (shoes, ID card, chain) in one pass.
    
    Args:
        image: BGR image
        pose_landmarks: MediaPipe pose landmarks (optional)
        
    Returns:
        Dictionary with all detection results
    """
    h, w = image.shape[:2]
    
    # Extract regions from pose landmarks if available
    feet_region = None
    torso_region = None
    neck_region = None
    
    if pose_landmarks is not None:
        pts = [(lm.x * w, lm.y * h) for lm in pose_landmarks.landmark]
        
        # Feet region (ankles to bottom)
        if len(pts) > 28:
            ankle_y = int((pts[27][1] + pts[28][1]) / 2)
            feet_region = (0, ankle_y, w, h - ankle_y)
        
        # Torso region (shoulders to hips)
        if len(pts) > 24:
            shoulder_y = int((pts[11][1] + pts[12][1]) / 2)
            hip_y = int((pts[23][1] + pts[24][1]) / 2)
            torso_x = int(w * 0.25)
            torso_w = int(w * 0.5)
            torso_region = (torso_x, shoulder_y, torso_w, hip_y - shoulder_y)
        
        # Neck region (nose to shoulders)
        if len(pts) > 12:
            nose_y = int(pts[0][1])
            shoulder_y = int((pts[11][1] + pts[12][1]) / 2)
            neck_x = int(w * 0.35)
            neck_w = int(w * 0.3)
            neck_region = (neck_x, nose_y, neck_w, shoulder_y - nose_y)
    
    # Initialize detectors
    shoe_detector = ShoeDetector()
    id_card_detector = IDCardDetector()
    chain_detector = ChainLanyardDetector()
    
    # Perform detections
    shoe_result = shoe_detector.detect_shoes(image, feet_region)
    id_card_result = id_card_detector.detect_id_card(image, torso_region)
    chain_result = chain_detector.detect_chain(image, neck_region)
    
    return {
        'shoes': shoe_result,
        'id_card': id_card_result,
        'chain': chain_result
    }
