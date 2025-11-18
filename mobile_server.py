"""
Mobile Phone Biometric Server

This Flask-SocketIO server simulates a mobile phone biometric authentication service.
It provides real-time communication between PC and mobile phone for biometric verification.

Features:
- WebSocket communication for real-time updates
- QR code scanning simulation
- Biometric authentication (fingerprint/face)
- Secure token exchange
- REST API endpoints for mobile app integration
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import uuid
from datetime import datetime, timedelta
import hashlib
import random
from pathlib import Path
import threading
import time

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state for mobile connections
mobile_connections = {}  # session_id -> mobile_info
active_sessions = {}     # session_id -> pc_info
biometric_db = {}        # student_id -> biometric_data

# HTML template for mobile phone interface
MOBILE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Mobile Biometric Auth</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        button { width: 100%; padding: 12px; margin: 10px 0; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        .scan-btn { background: #007bff; color: white; }
        .scan-btn:hover { background: #0056b3; }
        .auth-btn { background: #28a745; color: white; }
        .auth-btn:hover { background: #1e7e34; }
        .cancel-btn { background: #dc3545; color: white; }
        .cancel-btn:hover { background: #bd2130; }
        #qr-reader { width: 100%; height: 300px; border: 1px solid #ddd; border-radius: 5px; display: none; }
        .hidden { display: none; }
        .student-info { background: #f8f9fa; padding: 10px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h2>📱 Mobile Biometric Auth</h2>
        <div id="status" class="status info">Ready to scan QR code</div>

        <div id="scan-section">
            <button class="scan-btn" onclick="startQRScan()">📷 Scan QR Code</button>
            <div id="qr-reader"></div>
        </div>

        <div id="auth-section" class="hidden">
            <div id="student-info" class="student-info"></div>
            <button class="auth-btn" onclick="performBiometricAuth()">🔐 Authenticate Biometric</button>
            <button class="cancel-btn" onclick="cancelAuth()">❌ Cancel</button>
        </div>

        <div id="result-section" class="hidden">
            <div id="result-status" class="status"></div>
            <button class="scan-btn" onclick="resetApp()">🔄 Scan Another Code</button>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <script>
        const socket = io();
        let currentSessionId = null;
        let currentStudentId = null;
        let qrScanner = null;

        socket.on('connect', () => {
            console.log('Connected to server');
            updateStatus('Connected to biometric server', 'success');
        });

        socket.on('disconnect', () => {
            updateStatus('Disconnected from server', 'error');
        });

        socket.on('biometric_result', (data) => {
            showResult(data.success, data.message, data.confidence);
        });

        function updateStatus(message, type = 'info') {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = `status ${type}`;
        }

        function startQRScan() {
            updateStatus('Initializing camera...', 'info');

            // Simulate QR scanning (in real app, use camera API)
            setTimeout(() => {
                // Simulate successful QR scan
                const mockQRData = {
                    type: 'student_verification',
                    student_id: 'STU' + Math.floor(Math.random() * 1000).toString().padStart(3, '0'),
                    token: generateUUID(),
                    timestamp: new Date().toISOString(),
                    college_id: 'student_college'
                };

                handleQRScan(mockQRData);
            }, 2000);
        }

        function handleQRScan(qrData) {
            console.log('QR Scanned:', qrData);
            currentSessionId = qrData.session_id || generateUUID();
            currentStudentId = qrData.student_id;

            // Join session room
            socket.emit('join_session', { session_id: currentSessionId, student_id: currentStudentId });

            // Show auth section
            document.getElementById('scan-section').classList.add('hidden');
            document.getElementById('auth-section').classList.remove('hidden');

            // Update student info
            document.getElementById('student-info').innerHTML = `
                <strong>Student ID:</strong> ${currentStudentId}<br>
                <strong>College:</strong> ${qrData.college_id.replace('_', ' ').toUpperCase()}<br>
                <strong>Time:</strong> ${new Date(qrData.timestamp).toLocaleString()}
            `;

            updateStatus('Ready for biometric authentication', 'success');
        }

        function performBiometricAuth() {
            updateStatus('Authenticating...', 'info');

            // Simulate biometric authentication delay
            setTimeout(() => {
                // Simulate biometric success/failure
                const success = Math.random() > 0.1; // 90% success rate
                const confidence = success ? (0.8 + Math.random() * 0.2) : (Math.random() * 0.3);

                socket.emit('biometric_auth', {
                    session_id: currentSessionId,
                    student_id: currentStudentId,
                    success: success,
                    confidence: confidence,
                    biometric_type: 'fingerprint'
                });
            }, 1500);
        }

        function showResult(success, message, confidence) {
            document.getElementById('auth-section').classList.add('hidden');
            document.getElementById('result-section').classList.remove('hidden');

            const resultDiv = document.getElementById('result-status');
            resultDiv.textContent = message;
            resultDiv.className = `status ${success ? 'success' : 'error'}`;

            if (confidence !== undefined) {
                resultDiv.innerHTML += `<br>Confidence: ${(confidence * 100).toFixed(1)}%`;
            }
        }

        function cancelAuth() {
            socket.emit('cancel_auth', { session_id: currentSessionId });
            resetApp();
        }

        function resetApp() {
            currentSessionId = null;
            currentStudentId = null;

            document.getElementById('scan-section').classList.remove('hidden');
            document.getElementById('auth-section').classList.add('hidden');
            document.getElementById('result-section').classList.add('hidden');

            updateStatus('Ready to scan QR code', 'info');
        }

        function generateUUID() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
                const r = Math.random() * 16 | 0;
                const v = c == 'x' ? r : (r & 0x3 | 0x8);
                return v.toString(16);
            });
        }

        // Simulate QR scan for testing (remove in production)
        setTimeout(() => {
            if (!currentSessionId) {
                updateStatus('Demo: Auto-scanning QR code...', 'info');
                startQRScan();
            }
        }, 3000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve mobile phone interface"""
    return render_template_string(MOBILE_HTML)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(active_sessions),
        'connected_mobiles': len(mobile_connections)
    })

@app.route('/api/register_biometric', methods=['POST'])
def register_biometric():
    """Register biometric data for a student"""
    data = request.json
    student_id = data.get('student_id')
    biometric_type = data.get('biometric_type', 'fingerprint')
    biometric_data = data.get('biometric_data')

    if not student_id:
        return jsonify({'error': 'Student ID required'}), 400

    if student_id not in biometric_db:
        biometric_db[student_id] = {}

    # Simulate biometric data storage
    biometric_db[student_id][biometric_type] = {
        'data': biometric_data or hashlib.sha256(f"{student_id}_{biometric_type}_{datetime.now().isoformat()}".encode()).hexdigest(),
        'registered_at': datetime.now().isoformat(),
        'device_type': 'mobile_phone'
    }

    return jsonify({
        'success': True,
        'message': f'Biometric registered for {student_id}',
        'biometric_type': biometric_type
    })

@app.route('/api/verify_biometric/<student_id>', methods=['POST'])
def verify_biometric(student_id):
    """Verify biometric for a student"""
    if student_id not in biometric_db:
        return jsonify({'error': 'Biometric not registered'}), 404

    # Simulate verification
    success = random.random() > 0.1  # 90% success rate
    confidence = random.uniform(0.8, 0.98) if success else random.uniform(0.0, 0.3)

    return jsonify({
        'student_id': student_id,
        'success': success,
        'confidence': confidence,
        'biometric_type': 'fingerprint'
    })

@socketio.on('join_session')
def handle_join_session(data):
    """Handle mobile joining a verification session"""
    session_id = data.get('session_id')
    student_id = data.get('student_id')

    if not session_id or not student_id:
        emit('error', {'message': 'Session ID and Student ID required'})
        return

    join_room(session_id)
    mobile_connections[session_id] = {
        'student_id': student_id,
        'joined_at': datetime.now().isoformat(),
        'status': 'connected'
    }

    print(f"Mobile joined session {session_id} for student {student_id}")
    emit('session_joined', {
        'session_id': session_id,
        'student_id': student_id,
        'message': 'Successfully joined verification session'
    })

@socketio.on('biometric_auth')
def handle_biometric_auth(data):
    """Handle biometric authentication from mobile"""
    session_id = data.get('session_id')
    student_id = data.get('student_id')
    success = data.get('success', False)
    confidence = data.get('confidence', 0.0)
    biometric_type = data.get('biometric_type', 'fingerprint')

    if not session_id or not student_id:
        emit('error', {'message': 'Session ID and Student ID required'})
        return

    # Update mobile connection status
    if session_id in mobile_connections:
        mobile_connections[session_id]['status'] = 'authenticated' if success else 'failed'
        mobile_connections[session_id]['auth_result'] = {
            'success': success,
            'confidence': confidence,
            'biometric_type': biometric_type,
            'timestamp': datetime.now().isoformat()
        }

    # Send result back to mobile
    result_message = f"Biometric {'successful' if success else 'failed'}"
    if confidence:
        result_message += f" (confidence: {confidence:.1%})"

    emit('biometric_result', {
        'success': success,
        'message': result_message,
        'confidence': confidence
    })

    # Notify PC of authentication result
    emit('pc_biometric_result', {
        'session_id': session_id,
        'student_id': student_id,
        'success': success,
        'confidence': confidence,
        'biometric_type': biometric_type,
        'timestamp': datetime.now().isoformat()
    }, room=session_id, skip_sid=True)

    print(f"Biometric auth for session {session_id}: {'SUCCESS' if success else 'FAILED'}")

@socketio.on('cancel_auth')
def handle_cancel_auth(data):
    """Handle authentication cancellation"""
    session_id = data.get('session_id')

    if session_id in mobile_connections:
        mobile_connections[session_id]['status'] = 'cancelled'
        emit('auth_cancelled', {'message': 'Authentication cancelled'})

        # Notify PC
        emit('pc_auth_cancelled', {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }, room=session_id, skip_sid=True)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle mobile disconnection"""
    # Find and remove disconnected mobile
    for session_id, mobile_info in list(mobile_connections.items()):
        if mobile_info.get('status') == 'connected':
            mobile_info['status'] = 'disconnected'
            print(f"Mobile disconnected from session {session_id}")

def cleanup_expired_sessions():
    """Clean up expired sessions and connections"""
    current_time = datetime.now()
    expired_sessions = []

    for session_id, mobile_info in list(mobile_connections.items()):
        joined_at = datetime.fromisoformat(mobile_info['joined_at'])
        if (current_time - joined_at).total_seconds() > 3600:  # 1 hour timeout
            expired_sessions.append(session_id)

    for session_id in expired_sessions:
        del mobile_connections[session_id]
        print(f"Cleaned up expired session {session_id}")

    # Schedule next cleanup
    threading.Timer(300, cleanup_expired_sessions).start()  # Clean every 5 minutes

if __name__ == '__main__':
    # Start cleanup thread
    cleanup_expired_sessions()

    print("🚀 Starting Mobile Biometric Server...")
    print("📱 Mobile interface: http://localhost:5000")
    print("🔗 WebSocket endpoint: ws://localhost:5000/socket.io")
    print("📊 Health check: http://localhost:5000/api/health")

    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
