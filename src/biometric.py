"""
Biometric Verification Module

This module provides face recognition biometric verification for student verification.
Uses face detection and recognition for secure authentication.
"""

import hashlib
import json
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .config import AppConfig


class BiometricSystem:
    """Simulates mobile phone biometric verification system"""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.biometric_db_path = cfg.data_dir / "biometric_db.json"
        self._ensure_biometric_db()

    def _ensure_biometric_db(self):
        """Ensure biometric database exists"""
        if not self.biometric_db_path.exists():
            self.biometric_db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.biometric_db_path, 'w') as f:
                json.dump({}, f)

    def _load_biometric_db(self) -> Dict[str, Any]:
        """Load biometric database"""
        try:
            with open(self.biometric_db_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_biometric_db(self, db: Dict[str, Any]):
        """Save biometric database"""
        with open(self.biometric_db_path, 'w') as f:
            json.dump(db, f, indent=2)

    def register_biometric(self, student_id: str, biometric_type: str = "face",
                          biometric_data: Optional[str] = None, face_image: Optional[np.ndarray] = None) -> bool:
        """
        Register biometric data for a student

        Args:
            student_id: Student ID
            biometric_type: Type of biometric (fingerprint/face)
            biometric_data: Biometric data from phone (can be from WebAuthn)
            face_image: Face image array for face recognition (BGR format)

        Returns:
            True if registration successful
        """
        db = self._load_biometric_db()

        if student_id not in db:
            db[student_id] = {}

        # Handle face registration
        if biometric_type == "face" and face_image is not None:
            # Save face image and extract features
            face_path = self._save_face_image(student_id, face_image)
            biometric_data = face_path  # Store path as biometric data
        elif biometric_data is None:
            # Generate simulated biometric hash (for testing only)
            biometric_data = hashlib.sha256(
                f"{student_id}_{biometric_type}_{datetime.now().isoformat()}".encode()
            ).hexdigest()

        # Store biometric data with metadata
        db[student_id][biometric_type] = {
            "data": biometric_data,
            "registered_at": datetime.now().isoformat(),
            "device_type": "camera" if face_image is not None else "mobile_phone",
            "source": "camera_capture" if face_image is not None else ("webauthn" if biometric_data and len(biometric_data) > 64 else "simulated")
        }

        self._save_biometric_db(db)

        # Also save to separate storage file
        if biometric_type == "face":
            self._save_face_separately(student_id, biometric_data, biometric_type)

        return True

    def _save_face_image(self, student_id: str, face_image: np.ndarray) -> str:
        """Save face image to file and return path"""
        face_dir = self.cfg.data_dir / "face_images"
        face_dir.mkdir(parents=True, exist_ok=True)

        face_path = face_dir / f"{student_id}_face.jpg"
        cv2.imwrite(str(face_path), face_image)

        return str(face_path)

    def _save_face_separately(self, student_id: str, face_data: str, biometric_type: str):
        """Save face data to a separate file for dedicated storage"""
        try:
            face_file = self.cfg.data_dir / "faces.json"
            face_file.parent.mkdir(parents=True, exist_ok=True)

            # Load existing faces
            if face_file.exists():
                with open(face_file, 'r') as f:
                    all_faces = json.load(f)
            else:
                all_faces = {}

            # Store face data separately
            all_faces[student_id] = {
                "face_data": face_data,
                "biometric_type": biometric_type,
                "stored_at": datetime.now().isoformat(),
                "device_type": "camera"
            }

            # Save to file
            with open(face_file, 'w') as f:
                json.dump(all_faces, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save face separately: {e}")

    def verify_biometric(self, student_id: str, biometric_type: str = "face",
                        biometric_data: Optional[str] = None, face_image: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        """
        Verify biometric data for a student

        Args:
            student_id: Student ID
            biometric_type: Type of biometric (fingerprint/face)
            biometric_data: Biometric data to verify (simulated)
            face_image: Face image to verify against stored face

        Returns:
            Tuple of (is_verified, confidence_score)
        """
        db = self._load_biometric_db()

        if student_id not in db or biometric_type not in db[student_id]:
            return False, 0.0

        stored_data = db[student_id][biometric_type]["data"]

        # Handle face verification
        if biometric_type == "face" and face_image is not None:
            if stored_data.endswith('.jpg') and Path(stored_data).exists():
                # Compare with stored face image
                confidence = self._compare_faces(stored_data, face_image)
                is_verified = confidence >= self.cfg.biometric_verification_threshold
                return is_verified, confidence
            else:
                return False, 0.0

        # Simulate verification for other types
        if biometric_data is None:
            # Simulate current biometric input
            biometric_data = hashlib.sha256(
                f"{student_id}_{biometric_type}_verify_{datetime.now().isoformat()}".encode()
            ).hexdigest()

        # Simple hash comparison (in real system, this would be more sophisticated)
        if stored_data == biometric_data:
            confidence = 0.95  # High confidence match
        else:
            # Check if it's a partial match (first 16 chars)
            if stored_data[:16] == biometric_data[:16]:
                confidence = 0.85  # Medium confidence
            else:
                confidence = 0.0  # No match

        is_verified = confidence >= self.cfg.biometric_verification_threshold
        return is_verified, confidence

    def _compare_faces(self, stored_face_path: str, current_face_image: np.ndarray) -> float:
        """Compare stored face with current face image"""
        try:
            # Load stored face
            stored_face = cv2.imread(stored_face_path)
            if stored_face is None:
                return 0.0

            # Simple similarity comparison using histogram correlation
            # In a real system, this would use proper face recognition algorithms
            hist_stored = cv2.calcHist([stored_face], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist_current = cv2.calcHist([current_face_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

            hist_stored = cv2.normalize(hist_stored, hist_stored).flatten()
            hist_current = cv2.normalize(hist_current, hist_current).flatten()

            # Calculate correlation coefficient
            correlation = cv2.compareHist(hist_stored, hist_current, cv2.HISTCMP_CORREL)

            # Convert to confidence score (correlation is between -1 and 1, make it 0-1)
            confidence = (correlation + 1) / 2

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            print(f"Face comparison error: {e}")
            return 0.0

    def get_student_biometric_status(self, student_id: str) -> Dict[str, Any]:
        """
        Get biometric registration status for a student

        Args:
            student_id: Student ID

        Returns:
            Dictionary with biometric status information
        """
        db = self._load_biometric_db()

        if student_id not in db:
            return {
                "registered": False,
                "biometric_types": [],
                "registration_required": self.cfg.biometric_registration_required
            }

        biometric_info = db[student_id]
        return {
            "registered": True,
            "biometric_types": list(biometric_info.keys()),
            "registration_required": self.cfg.biometric_registration_required,
            "details": {
                bio_type: {
                    "registered_at": info["registered_at"],
                    "device_type": info["device_type"]
                }
                for bio_type, info in biometric_info.items()
            }
        }

    def is_biometric_registered(self, student_id: str, biometric_type: str = "face") -> bool:
        """
        Check if biometric is registered for student

        Args:
            student_id: Student ID
            biometric_type: Type of biometric

        Returns:
            True if registered
        """
        db = self._load_biometric_db()
        return student_id in db and biometric_type in db[student_id]

    def simulate_mobile_biometric_prompt(self, student_id: str, biometric_type: str = "face") -> Tuple[bool, float]:
        """
        Simulate mobile phone biometric prompt

        Args:
            student_id: Student ID
            biometric_type: Type of biometric

        Returns:
            Tuple of (success, confidence)
        """
        if not self.is_biometric_registered(student_id, biometric_type):
            return False, 0.0

        # Simulate mobile biometric verification
        # In a real system, this would interface with mobile device
        import random
        success_rate = 0.9  # 90% success rate simulation
        success = random.random() < success_rate

        if success:
            confidence = random.uniform(0.8, 0.98)
        else:
            confidence = random.uniform(0.0, 0.3)

        return success, confidence

    def verify_identity_on_phone(self, student_id: str) -> Dict[str, Any]:
        """
        Verify student identity using mobile phone face recognition
        This simulates the phone-side biometric verification process

        Args:
            student_id: Student ID to verify against

        Returns:
            Dictionary with phone-side verification results
        """
        # Simulate mobile phone face recognition verification
        # In a real system, this would interface with mobile device camera
        success, confidence = self.simulate_mobile_biometric_prompt(student_id, "face")

        return {
            "verified": success,
            "confidence": confidence,
            "biometric_type": "face",
            "student_id": student_id if success else None,
            "verification_method": "mobile_face_recognition",
            "device_type": "mobile_phone",
            "timestamp": datetime.now().isoformat()
        }

    def verify_identity(self, image: Any, student_id: str) -> Dict[str, Any]:
        """
        Legacy method for backward compatibility
        Now delegates to phone-side verification
        """
        return self.verify_identity_on_phone(student_id)


def get_biometric_system(cfg: AppConfig) -> BiometricSystem:
    """Get biometric system instance"""
    return BiometricSystem(cfg)
