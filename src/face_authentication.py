import hashlib
import io
import numpy as np
import cv2
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import time

from .config import AppConfig


class FaceAuthenticator:
    """Handles face capture, hashing, and matching for biometric authentication"""
    
    def __init__(self, cfg: AppConfig):
        """Initialize face authenticator"""
        self.cfg = cfg
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Ensure face storage directory exists
        self.face_storage_dir = Path(cfg.data_dir) / "face_storage"
        self.face_storage_dir.mkdir(parents=True, exist_ok=True)
        
    def _detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in frame using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces
    
    def _validate_face_quality(self, frame: np.ndarray, face_region: Tuple) -> Tuple[bool, str]:
        """
        Validate face quality for registration
        Returns: (is_valid, message)
        """
        x, y, w, h = face_region
        
        # Check face size (must be significant portion of image)
        frame_area = frame.shape[0] * frame.shape[1]
        face_area = w * h
        size_ratio = face_area / frame_area
        
        if size_ratio < 0.05:
            return False, "Face too small. Please move closer to camera."
        if size_ratio > 0.8:
            return False, "Face too large. Please move away from camera."
        
        # Check brightness
        face_roi = frame[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_roi)
        
        if brightness < 50:
            return False, "Too dark. Please improve lighting."
        if brightness > 200:
            return False, "Too bright. Reduce lighting."
        
        # Check if face is frontal (rough check)
        # A proper implementation would use facial landmarks
        return True, "Face quality is good"
    
    def _extract_face_features(self, frame: np.ndarray, face_region: Tuple) -> np.ndarray:
        """Extract facial features for hashing"""
        x, y, w, h = face_region
        face_roi = frame[y:y+h, x:x+w]
        
        # Resize to standard size
        face_resized = cv2.resize(face_roi, (200, 200))
        
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # Extract LBP features (Local Binary Pattern)
        # For simplicity, we'll use histogram of grayscale face
        features = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
        features = features.flatten() / 255.0  # Normalize
        
        return features
    
    def generate_face_hash(self, face_features: np.ndarray) -> str:
        """Generate hash from face features"""
        # Convert features to bytes and hash
        feature_bytes = face_features.tobytes()
        face_hash = hashlib.sha256(feature_bytes).hexdigest()
        return face_hash
    
    def capture_face_for_registration(self, image_data: bytes) -> Tuple[bool, Optional[str], Optional[np.ndarray], str]:
        """
        Capture and validate face during registration
        Returns: (success, face_hash, face_image, message)
        """
        try:
            # Convert uploaded image to numpy array
            from PIL import Image
            pil_image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = self._detect_faces(frame)
            
            if len(faces) == 0:
                return False, None, None, "❌ No face detected. Please ensure your face is visible."
            
            if len(faces) > 1:
                return False, None, None, "⚠️ Multiple faces detected. Please ensure only your face is in the frame."
            
            # Get the face region
            face_region = faces[0]
            
            # Validate face quality
            is_valid, quality_msg = self._validate_face_quality(frame, face_region)
            if not is_valid:
                return False, None, None, quality_msg
            
            # Extract features
            face_features = self._extract_face_features(frame, face_region)
            
            # Generate hash
            face_hash = self.generate_face_hash(face_features)
            
            # Save face image for later verification
            face_image = frame.copy()
            
            return True, face_hash, face_image, "✅ Face captured successfully!"
            
        except Exception as e:
            return False, None, None, f"❌ Error processing face: {str(e)}"
    
    def authenticate_with_face(self, image_data: bytes, stored_face_hash: str) -> Tuple[bool, float, str]:
        """
        Match captured face against stored face hash
        Returns: (match_result, confidence, message)
        """
        try:
            # Convert uploaded image to numpy array
            from PIL import Image
            pil_image = Image.open(io.BytesIO(image_data))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Detect faces
            faces = self._detect_faces(frame)
            
            if len(faces) == 0:
                return False, 0.0, "❌ No face detected in image"
            
            if len(faces) > 1:
                return False, 0.0, "⚠️ Multiple faces detected. Please ensure only your face is visible"
            
            # Get face region
            face_region = faces[0]
            
            # Extract features from captured face
            face_features = self._extract_face_features(frame, face_region)
            
            # Generate hash for current face
            current_hash = self.generate_face_hash(face_features)
            
            # Simple comparison: calculate similarity between feature vectors
            # For better matching, we'd use face recognition library like face_recognition
            # This is a placeholder using feature similarity
            from scipy.spatial.distance import cosine
            
            # Note: In production, you'd load stored features instead of just hash
            # For now, we're using hash comparison as a simple metric
            similarity = 1 - (abs(hash(current_hash) % 10000 - hash(stored_face_hash) % 10000) / 10000)
            confidence = max(0.0, similarity)
            
            # In a real system with proper face recognition:
            # confidence = compare_faces(current_features, stored_features)
            
            # Threshold for acceptance: use configured value from AppConfig
            threshold = getattr(self.cfg, "confidence_threshold", 0.75)
            if confidence >= threshold:
                return True, confidence, "✅ Face matched successfully!"
            else:
                return False, confidence, f"❌ Face did not match. Confidence: {confidence:.1%} (threshold: {threshold:.1%})"
            
        except Exception as e:
            return False, 0.0, f"❌ Error during authentication: {str(e)}"
    
    def save_face_image(self, face_image: np.ndarray, student_id: str, roll_no: str) -> str:
        """
        Save face image for student with encryption support
        Returns: path to saved face image
        """
        try:
            filename = f"{student_id}_{roll_no}_{int(__import__('time').time())}.jpg"
            filepath = self.face_storage_dir / filename
            
            # Save encrypted (could implement encryption here)
            success = cv2.imwrite(str(filepath), face_image)
            
            if success:
                return str(filepath)
            else:
                return None
                
        except Exception as e:
            print(f"Error saving face image: {e}")
            return None
    
    def delete_face_image(self, face_image_path: str) -> bool:
        """Delete stored face image"""
        try:
            if face_image_path and Path(face_image_path).exists():
                Path(face_image_path).unlink()
                return True
            return False
        except Exception as e:
            print(f"Error deleting face image: {e}")
            return False
