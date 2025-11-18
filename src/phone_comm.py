"""
Phone-PC Communication Module

This module simulates secure communication between mobile phone and PC
for transmitting verified student ID after biometric authentication.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import qrcode
from PIL import Image

from .config import AppConfig


class PhoneCommunication:
    """Simulates secure phone-PC communication for biometric verification"""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.comm_db_path = cfg.data_dir / "phone_comm.json"
        self._ensure_comm_db()

    def _ensure_comm_db(self):
        """Ensure communication database exists"""
        if not self.comm_db_path.exists():
            self.comm_db_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.comm_db_path, 'w') as f:
                json.dump({}, f)

    def _load_comm_db(self) -> Dict[str, Any]:
        """Load communication database"""
        try:
            with open(self.comm_db_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_comm_db(self, db: Dict[str, Any]):
        """Save communication database"""
        with open(self.comm_db_path, 'w') as f:
            json.dump(db, f, indent=2)

    def generate_verification_token(self, student_id: str, validity_minutes: int = 5) -> str:
        """
        Generate a secure token for phone verification

        Args:
            student_id: Student ID to verify
            validity_minutes: Token validity period

        Returns:
            Verification token string
        """
        token = str(uuid.uuid4())
        expiry = datetime.now() + timedelta(minutes=validity_minutes)

        db = self._load_comm_db()
        db[token] = {
            "student_id": student_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": expiry.isoformat(),
            "used": False,
            "device_type": "mobile_phone"
        }
        self._save_comm_db(db)

        return token

    def verify_token_and_get_student_id(self, token: str) -> Optional[str]:
        """
        Verify token and return associated student ID

        Args:
            token: Verification token from phone

        Returns:
            Student ID if token is valid, None otherwise
        """
        db = self._load_comm_db()

        if token not in db:
            return None

        token_data = db[token]

        # Check if token is expired
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        if datetime.now() > expires_at:
            # Clean up expired token
            del db[token]
            self._save_comm_db(db)
            return None

        # Check if token already used
        if token_data["used"]:
            return None

        # Mark token as used
        token_data["used"] = True
        token_data["used_at"] = datetime.now().isoformat()
        self._save_comm_db(db)

        return token_data["student_id"]

    def simulate_phone_verification(self, student_id: str) -> Tuple[bool, str]:
        """
        Simulate the complete phone verification process

        Args:
            student_id: Student ID to verify

        Returns:
            Tuple of (success, token_or_error_message)
        """
        # Generate token for verification
        token = self.generate_verification_token(student_id)

        # Simulate phone-side biometric verification
        # In real implementation, this would be done on the phone
        from .biometric import get_biometric_system
        bio_system = get_biometric_system(self.cfg)

        # Check if biometric is registered
        if not bio_system.is_biometric_registered(student_id):
            return False, "Biometric not registered for this student"

        # Simulate biometric verification on phone
        success, confidence = bio_system.simulate_mobile_biometric_prompt(student_id)

        if not success:
            return False, f"Biometric verification failed (confidence: {confidence:.2f})"

        # Return token for PC to verify
        return True, token

    def get_verification_qr_data(self, student_id: str) -> Optional[str]:
        """
        Generate QR code data for phone scanning

        Args:
            student_id: Student ID

        Returns:
            JSON string for QR code containing verification data
        """
        success, result = self.simulate_phone_verification(student_id)

        if not success:
            return None

        qr_data = {
            "type": "student_verification",
            "student_id": student_id,
            "token": result,
            "timestamp": datetime.now().isoformat(),
            "college_id": self.cfg.college_name.replace(" ", "_").lower()
        }

        return json.dumps(qr_data)

    def generate_qr_code(self, student_id: str) -> Optional[Image.Image]:
        """
        Generate QR code for phone scanning

        Args:
            student_id: Student ID to generate QR for

        Returns:
            PIL Image containing QR code, or None if failed
        """
        qr_data = self.get_verification_qr_data(student_id)
        if not qr_data:
            return None

        # Create QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_data)
        qr.make(fit=True)

        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        return img

    def generate_biometric_qr(self, student_id: str) -> Tuple[str, Image.Image]:
        """
        Generate a unique biometric QR code for a student without auto-simulating mobile auth.

        Returns:
            (token, qr_image)
        """
        token = self.generate_verification_token(student_id)

        qr_payload = {
            "type": "student_verification",
            "student_id": student_id,
            "token": token,
            "timestamp": datetime.now().isoformat(),
            "college_id": self.cfg.college_name.replace(" ", "_").lower()
        }

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(json.dumps(qr_payload))
        qr.make(fit=True)

        qr_image = qr.make_image(fill_color="black", back_color="white")
        return token, qr_image

    def simulate_phone_qr_scan(self, qr_image: Image.Image) -> Optional[str]:
        """
        Simulate phone scanning QR code and performing biometric verification

        Args:
            qr_image: QR code image from PC

        Returns:
            Verification token if successful, None otherwise
        """
        # In a real implementation, this would be done by a phone app
        # For simulation, we'll decode the QR and simulate the phone process

        # This is a simplified simulation - in reality, the phone app would:
        # 1. Scan QR code with camera
        # 2. Decode the JSON data
        # 3. Perform biometric authentication
        # 4. Send token back to PC

        # For demo purposes, we'll simulate successful verification
        # In a real app, this would be handled by the phone application

        return "simulated_token_from_phone"

    def cleanup_expired_tokens(self):
        """Clean up expired verification tokens"""
        db = self._load_comm_db()
        current_time = datetime.now()
        expired_tokens = []

        for token, data in db.items():
            expires_at = datetime.fromisoformat(data["expires_at"])
            if current_time > expires_at:
                expired_tokens.append(token)

        for token in expired_tokens:
            del db[token]

        if expired_tokens:
            self._save_comm_db(db)
            print(f"Cleaned up {len(expired_tokens)} expired tokens")


def get_phone_comm_system(cfg: AppConfig) -> PhoneCommunication:
    """Get phone communication system instance"""
    return PhoneCommunication(cfg)
