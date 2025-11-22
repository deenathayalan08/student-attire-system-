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

@app.route('/biometric/<student_id>')
def biometric_verification_page(student_id):
    """Serve biometric verification page that triggers phone fingerprint sensor"""
    token = request.args.get('token')
    if not token:
        return jsonify({'error': 'Token required'}), 400
    
    # Create biometric verification page with WebAuthn
    biometric_page = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Biometric Verification - {student_id}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .container {{
            max-width: 400px;
            width: 100%;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            text-align: center;
        }}
        .fingerprint-icon {{
            font-size: 80px;
            margin: 20px 0;
            animation: pulse 2s infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
        .status {{
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            font-weight: bold;
        }}
        .info {{
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }}
        .success {{
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .error {{
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        button {{
            width: 100%;
            padding: 15px;
            margin: 10px 0;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            background: #28a745;
            color: white;
        }}
        button:hover {{
            background: #218838;
        }}
        button:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        .token-display {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-family: monospace;
            word-break: break-all;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 Biometric Verification</h1>
        <div class="fingerprint-icon">👆</div>
        <div id="status" class="status info">
            <p><strong>Student ID:</strong> {student_id}</p>
            <p>Place your finger on the fingerprint sensor</p>
        </div>
        
        <button id="authBtn" onclick="startFingerprintAuth()">🔒 Authenticate with Fingerprint</button>
        
        <div id="result" style="display: none;">
            <div id="resultStatus" class="status"></div>
            <div id="tokenDisplay" class="token-display" style="display: none;">
                <strong>Verification Token:</strong><br>
                <span id="tokenValue"></span>
            </div>
        </div>
    </div>

    <script>
        const studentId = '{student_id}';
        const token = '{token}';
        let fingerprintData = null;

        async function startFingerprintAuth() {{
            const btn = document.getElementById('authBtn');
            const status = document.getElementById('status');
            
            btn.disabled = true;
            status.innerHTML = '<p>🔐 Accessing fingerprint sensor...</p><p>Please place your finger on the sensor</p>';
            
            try {{
                // Check if WebAuthn is supported
                if (!window.PublicKeyCredential) {{
                    throw new Error('Web Authentication API is not supported on this device. Please use a modern browser with biometric support.');
                }}
                
                // Request fingerprint authentication using WebAuthn
                const credential = await navigator.credentials.get({{
                    publicKey: {{
                        challenge: new Uint8Array(32).fill(0), // In production, use a random challenge from server
                        allowCredentials: [],
                        timeout: 60000,
                        userVerification: 'required', // This triggers fingerprint/face authentication
                        rpId: window.location.hostname
                    }}
                }});
                
                // Extract fingerprint data
                if (credential) {{
                    // Convert credential to base64 for transmission
                    const rawId = Array.from(new Uint8Array(credential.rawId))
                        .map(b => String.fromCharCode(b))
                        .join('');
                    fingerprintData = btoa(rawId);
                    
                    // Store fingerprint data separately
                    await storeFingerprintData(fingerprintData);
                    
                    // Send to server
                    await sendFingerprintToServer(fingerprintData, credential);
                    
                }} else {{
                    throw new Error('Fingerprint authentication was cancelled or failed');
                }}
                
            }} catch (error) {{
                console.error('Fingerprint auth error:', error);
                status.className = 'status error';
                status.innerHTML = `<p>❌ Authentication Failed</p><p>${{error.message}}</p>`;
                btn.disabled = false;
            }}
        }}

        async function storeFingerprintData(fingerprintData) {{
            // Store fingerprint data in browser's local storage (separate storage)
            const storageKey = `fingerprint_${{studentId}}`;
            const fingerprintRecord = {{
                student_id: studentId,
                fingerprint_data: fingerprintData,
                captured_at: new Date().toISOString(),
                device_info: navigator.userAgent
            }};
            
            // Store in localStorage
            localStorage.setItem(storageKey, JSON.stringify(fingerprintRecord));
            
            // Also store in IndexedDB for more robust storage
            if ('indexedDB' in window) {{
                const request = indexedDB.open('BiometricDB', 1);
                request.onupgradeneeded = (event) => {{
                    const db = event.target.result;
                    if (!db.objectStoreNames.contains('fingerprints')) {{
                        const objectStore = db.createObjectStore('fingerprints', {{ keyPath: 'student_id' }});
                        objectStore.createIndex('captured_at', 'captured_at', {{ unique: false }});
                    }}
                }};
                request.onsuccess = (event) => {{
                    const db = event.target.result;
                    const transaction = db.transaction(['fingerprints'], 'readwrite');
                    const store = transaction.objectStore('fingerprints');
                    store.put(fingerprintRecord);
                }};
            }}
        }}

        async function sendFingerprintToServer(fingerprintData, credential) {{
            const status = document.getElementById('status');
            const result = document.getElementById('result');
            const resultStatus = document.getElementById('resultStatus');
            const tokenDisplay = document.getElementById('tokenDisplay');
            const tokenValue = document.getElementById('tokenValue');
            
            try {{
                // Send fingerprint data to server
                const response = await fetch('/api/capture_fingerprint', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{
                        student_id: studentId,
                        token: token,
                        fingerprint_data: fingerprintData,
                        credential_id: Array.from(new Uint8Array(credential.rawId))
                            .map(b => b.toString(16).padStart(2, '0'))
                            .join(''),
                        timestamp: new Date().toISOString()
                    }})
                }});
                
                const data = await response.json();
                
                if (data.success) {{
                    status.className = 'status success';
                    status.innerHTML = '<p>✅ Fingerprint captured successfully!</p>';
                    
                    result.style.display = 'block';
                    resultStatus.className = 'status success';
                    resultStatus.innerHTML = '<p>✅ <strong>Verification Complete</strong></p><p>Your fingerprint has been captured and stored.</p>';
                    
                    tokenDisplay.style.display = 'block';
                    tokenValue.textContent = token;
                    
                    // Show success message
                    alert('✅ Fingerprint authentication successful!\\n\\nVerification Token: ' + token);
                }} else {{
                    throw new Error(data.error || 'Failed to store fingerprint data');
                }}
                
            }} catch (error) {{
                console.error('Server error:', error);
                status.className = 'status error';
                status.innerHTML = `<p>❌ Error sending data to server</p><p>${{error.message}}</p>`;
            }}
        }}

        // Auto-start fingerprint authentication when page loads
        window.addEventListener('load', () => {{
            // Small delay to ensure page is fully loaded
            setTimeout(() => {{
                startFingerprintAuth();
            }}, 500);
        }});
    </script>
</body>
</html>
"""
    return render_template_string(biometric_page)

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

@app.route('/api/capture_fingerprint', methods=['POST'])
def capture_fingerprint():
    """Capture and store fingerprint data from phone"""
    data = request.json
    student_id = data.get('student_id')
    token = data.get('token')
    fingerprint_data = data.get('fingerprint_data')
    credential_id = data.get('credential_id')
    timestamp = data.get('timestamp')

    if not student_id or not fingerprint_data:
        return jsonify({'error': 'Student ID and fingerprint data required'}), 400

    # Store fingerprint data separately in biometric database
    if student_id not in biometric_db:
        biometric_db[student_id] = {}

    # Store fingerprint data with metadata
    biometric_db[student_id]['fingerprint'] = {
        'data': fingerprint_data,  # Base64 encoded fingerprint data
        'credential_id': credential_id,
        'captured_at': timestamp or datetime.now().isoformat(),
        'device_type': 'mobile_phone',
        'verification_token': token,
        'source': 'webauthn'
    }

    # Also save to file for persistence
    try:
        import json as json_lib
        from pathlib import Path
        biometric_file = Path('data') / 'fingerprints.json'
        biometric_file.parent.mkdir(exist_ok=True)
        
        # Load existing fingerprints
        if biometric_file.exists():
            with open(biometric_file, 'r') as f:
                all_fingerprints = json_lib.load(f)
        else:
            all_fingerprints = {}
        
        # Store fingerprint data separately
        all_fingerprints[student_id] = biometric_db[student_id]['fingerprint']
        
        # Save to file
        with open(biometric_file, 'w') as f:
            json_lib.dump(all_fingerprints, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save fingerprint to file: {e}")

    return jsonify({
        'success': True,
        'message': f'Fingerprint captured and stored for {student_id}',
        'student_id': student_id,
        'token': token,
        'timestamp': timestamp
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
