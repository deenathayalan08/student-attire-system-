"""
Biometric Verification Module

This module provides mobile phone fingerprint biometric simulation for student verification.
Uses fingerprint sensor authentication similar to mobile phone authenticators.
"""

import hashlib
import json
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

    def register_biometric(self, student_id: str, biometric_type: str = "fingerprint",
                          biometric_data: Optional[str] = None) -> bool:
        """
        Register biometric data for a student

        Args:
            student_id: Student ID
            biometric_type: Type of biometric (fingerprint/face)
            biometric_data: Biometric data (simulated)

        Returns:
            True if registration successful
        """
        db = self._load_biometric_db()

        if student_id not in db:
            db[student_id] = {}

        # Simulate biometric data storage
        if biometric_data is None:
            # Generate simulated biometric hash
            biometric_data = hashlib.sha256(
                f"{student_id}_{biometric_type}_{datetime.now().isoformat()}".encode()
            ).hexdigest()

        db[student_id][biometric_type] = {
            "data": biometric_data,
            "registered_at": datetime.now().isoformat(),
            "device_type": "mobile_phone"
        }

        self._save_biometric_db(db)
        return True

    def verify_biometric(self, student_id: str, biometric_type: str = "fingerprint",
                        biometric_data: Optional[str] = None) -> Tuple[bool, float]:
        """
        Verify biometric data for a student

        Args:
            student_id: Student ID
            biometric_type: Type of biometric (fingerprint/face)
            biometric_data: Biometric data to verify (simulated)

        Returns:
            Tuple of (is_verified, confidence_score)
        """
        db = self._load_biometric_db()

        if student_id not in db or biometric_type not in db[student_id]:
            return False, 0.0

        stored_data = db[student_id][biometric_type]["data"]

        # Simulate verification
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

    def is_biometric_registered(self, student_id: str, biometric_type: str = "fingerprint") -> bool:
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

    def simulate_mobile_biometric_prompt(self, student_id: str, biometric_type: str = "fingerprint") -> Tuple[bool, float]:
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
        Verify student identity using mobile phone fingerprint sensor
        This simulates the phone-side biometric verification process

        Args:
            student_id: Student ID to verify against

        Returns:
            Dictionary with phone-side verification results
        """
        # Simulate mobile phone fingerprint verification
        # In a real system, this would interface with mobile device fingerprint sensor
        success, confidence = self.simulate_mobile_biometric_prompt(student_id, "fingerprint")

        return {
            "verified": success,
            "confidence": confidence,
            "biometric_type": "fingerprint",
            "student_id": student_id if success else None,
            "verification_method": "mobile_fingerprint",
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
