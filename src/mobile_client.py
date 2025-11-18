
import socketio
import json
import uuid
from datetime import datetime
import threading
import time
from typing import Dict, Any, Optional, Callable
from pathlib import Path

class MobileBiometricClient:
    """Client for communicating with mobile biometric server"""

    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self.sio = socketio.Client()
        self.connected = False
        self.current_session_id = None
        self.current_student_id = None

        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_biometric_result: Optional[Callable] = None
        self.on_auth_cancelled: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup Socket.IO event handlers"""

        @self.sio.on('connect')
        def on_connect():
            self.connected = True
            print("📱 Connected to mobile biometric server")
            if self.on_connected:
                self.on_connected()

        @self.sio.on('disconnect')
        def on_disconnect():
            self.connected = False
            print("📱 Disconnected from mobile biometric server")
            if self.on_disconnected:
                self.on_disconnected()

        @self.sio.on('session_joined')
        def on_session_joined(data):
            print(f"📱 Joined session: {data.get('session_id')}")
            self.current_session_id = data.get('session_id')
            self.current_student_id = data.get('student_id')

        @self.sio.on('biometric_result')
        def on_biometric_result(data):
            print(f"📱 Biometric result: {data}")
            if self.on_biometric_result:
                self.on_biometric_result(data)

        @self.sio.on('auth_cancelled')
        def on_auth_cancelled(data):
            print("📱 Authentication cancelled")
            if self.on_auth_cancelled:
                self.on_auth_cancelled(data)

        @self.sio.on('error')
        def on_error(data):
            print(f"📱 Error: {data}")
            if self.on_error:
                self.on_error(data)

    def connect(self) -> bool:
        """Connect to mobile biometric server"""
        try:
            self.sio.connect(self.server_url)
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            return self.connected
        except Exception as e:
            print(f"Failed to connect to mobile server: {e}")
            return False

    def disconnect(self):
        """Disconnect from mobile biometric server"""
        if self.connected:
            self.sio.disconnect()
            self.connected = False

    def join_verification_session(self, session_id: str, student_id: str) -> bool:
        """Join a verification session"""
        if not self.connected:
            return False

        try:
            self.sio.emit('join_session', {
                'session_id': session_id,
                'student_id': student_id
            })
            self.current_session_id = session_id
            self.current_student_id = student_id
            return True
        except Exception as e:
            print(f"Failed to join session: {e}")
            return False

    def perform_biometric_auth(self, success: bool = True, confidence: float = 0.9) -> bool:
        """Perform biometric authentication"""
        if not self.connected or not self.current_session_id or not self.current_student_id:
            return False

        try:
            self.sio.emit('biometric_auth', {
                'session_id': self.current_session_id,
                'student_id': self.current_student_id,
                'success': success,
                'confidence': confidence,
                'biometric_type': 'fingerprint'
            })
            return True
        except Exception as e:
            print(f"Failed to perform biometric auth: {e}")
            return False

    def cancel_authentication(self) -> bool:
        """Cancel ongoing authentication"""
        if not self.connected or not self.current_session_id:
            return False

        try:
            self.sio.emit('cancel_auth', {
                'session_id': self.current_session_id
            })
            return True
        except Exception as e:
            print(f"Failed to cancel auth: {e}")
            return False

    def simulate_qr_scan(self, qr_data: Dict[str, Any]) -> bool:
        """Simulate QR code scanning"""
        session_id = qr_data.get('session_id', str(uuid.uuid4()))
        student_id = qr_data.get('student_id', 'UNKNOWN')

        print(f"📱 Simulating QR scan for student {student_id}")
        return self.join_verification_session(session_id, student_id)

    def simulate_mobile_biometric_flow(self, qr_data: Dict[str, Any], auth_delay: float = 2.0) -> Dict[str, Any]:
        """Simulate complete mobile biometric flow"""
        result = {
            'success': False,
            'message': 'Flow not completed',
            'confidence': 0.0
        }

        # Join session
        if not self.simulate_qr_scan(qr_data):
            result['message'] = 'Failed to join session'
            return result

        # Wait for auth delay
        time.sleep(auth_delay)

        # Perform authentication
        import random
        success = random.random() > 0.1  # 90% success rate
        confidence = random.uniform(0.8, 0.98) if success else random.uniform(0.0, 0.3)

        if self.perform_biometric_auth(success, confidence):
            result = {
                'success': success,
                'message': f"Biometric {'successful' if success else 'failed'}",
                'confidence': confidence,
                'student_id': self.current_student_id,
                'session_id': self.current_session_id
            }
        else:
            result['message'] = 'Failed to perform authentication'

        return result

class MobileBiometricManager:
    """Manager for mobile biometric operations"""

    def __init__(self, server_url: str = "http://localhost:5000"):
        self.server_url = server_url
        self.client = None
        self.connected = False

    def start_mobile_connection(self) -> bool:
        """Start connection to mobile biometric server"""
        if self.client:
            self.client.disconnect()

        self.client = MobileBiometricClient(self.server_url)
        self.connected = self.client.connect()

        if self.connected:
            print("📱 Mobile biometric client connected successfully")
        else:
            print("❌ Failed to connect mobile biometric client")

        return self.connected

    def stop_mobile_connection(self):
        """Stop mobile biometric connection"""
        if self.client:
            self.client.disconnect()
            self.client = None
            self.connected = False
            print("📱 Mobile biometric client disconnected")

    def perform_mobile_biometric_verification(self, qr_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform biometric verification via mobile phone"""
        if not self.connected or not self.client:
            return {
                'success': False,
                'message': 'Mobile client not connected',
                'confidence': 0.0
            }

        print("📱 Starting mobile biometric verification...")
        result = self.client.simulate_mobile_biometric_flow(qr_data)

        if result['success']:
            print(f"✅ Mobile biometric successful: {result['confidence']:.1%} confidence")
        else:
            print(f"❌ Mobile biometric failed: {result['message']}")

        return result

    def is_connected(self) -> bool:
        """Check if mobile client is connected"""
        return self.connected and self.client and self.client.connected

# Global mobile manager instance
_mobile_manager = None

def get_mobile_biometric_manager(server_url: str = "http://localhost:5000") -> MobileBiometricManager:
    """Get global mobile biometric manager instance"""
    global _mobile_manager
    if _mobile_manager is None:
        _mobile_manager = MobileBiometricManager(server_url)
    return _mobile_manager

def start_mobile_server():
    """Start the mobile biometric server in a separate thread"""
    def run_server():
        import subprocess
        import sys
        try:
            subprocess.run([sys.executable, "mobile_server.py"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to start mobile server: {e}")

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    print("🚀 Mobile biometric server starting in background...")
    time.sleep(2)  # Give server time to start
    return server_thread

if __name__ == "__main__":
    # Test the mobile client
    print("Testing Mobile Biometric Client...")

    manager = get_mobile_biometric_manager()

    if manager.start_mobile_connection():
        # Test biometric verification
        test_qr_data = {
            'session_id': str(uuid.uuid4()),
            'student_id': 'TEST001',
            'type': 'student_verification'
        }

        result = manager.perform_mobile_biometric_verification(test_qr_data)
        print(f"Test result: {result}")

        manager.stop_mobile_connection()
    else:
        print("Failed to connect to mobile server. Make sure mobile_server.py is running.")
