# 🆘 Emergency Login Feature

## Overview
The system now includes an **Emergency Login** feature that allows students to login using their **Student ID (username) and password** when face authentication is not available or fails.

## Registration Flow (Updated)

### Stage 1: Student ID Generation
- Batch Year selection
- Department selection
- Section selection (A, B, C, etc.)
- Student Number (001-999)
- **Auto-generates Student ID** in format: `YY + DEPT_CODE + SECTION + NUMBER`
  - Example: `24CS1001` (2024, Computer Science, Section A, Student #1)

### Stage 2: Student Details
- Full Name
- Email
- Phone
- Gender (Male/Female)
- Contact Information
- Terms & Conditions agreement

### Stage 3: Emergency Login Setup ⚠️ **NEW**
- **Username:** Automatically set to Student ID (from Stage 1)
- **Password:** Student creates their own emergency password
  - Minimum 6 characters
  - Must contain letters and numbers
  - Password confirmation required
- **Purpose:** Backup login method when face authentication fails

### Stage 4: Face Capture (Biometric)
- Camera capture with cropping tool
- Face quality validation
- Face hash generation
- Biometric data storage
- **Primary login method**

## Login Methods

### 1. Face Authentication (Primary) 🔐
- **Default login method**
- Uses camera to capture face
- Matches against registered face biometrics
- Confidence threshold: 75% (configurable)
- Real-time quality metrics:
  - Face size
  - Brightness
  - Clarity
  - Liveness detection

### 2. Emergency Login (Backup) 🆘
- **When to use:**
  - Face not visible (mask, poor lighting, etc.)
  - Camera not available
  - Face recognition confidence too low
  - Emergency situations

- **Credentials:**
  - **Username:** Student ID (e.g., `24CS1001`)
  - **Password:** Emergency password set during registration

- **How to access:**
  1. Go to "Face Authentication" page
  2. Select "🆘 Emergency Login (Username & Password)"
  3. Enter Student ID and password
  4. Click "🔓 Login"

## Security Features

### Password Requirements
- Minimum 6 characters
- Must contain both letters and numbers
- Password strength validation during registration
- Secure password hashing (SHA-256 with salt)

### Authentication Logging
- All login attempts are logged
- Emergency logins are tracked separately
- Event logging includes:
  - Student ID
  - Login method (face vs emergency)
  - Timestamp
  - Authentication status

### Password Storage
- Passwords are **never stored in plain text**
- Uses secure hashing with salt
- Salt is unique per user
- Hash format: `{salt}:{hashed_password}`

## User Experience

### Registration
1. Student completes Stages 1-2 (ID generation and details)
2. **Stage 3:** Student creates emergency password
   - Clear instructions about emergency use
   - Password strength indicator
   - Confirmation required
3. **Stage 4:** Student captures face for biometric login
4. Registration complete - both login methods available

### Login
1. **Primary:** Student uses face authentication
   - Fast and convenient
   - No password to remember
   - Biometric security

2. **Backup:** If face fails, use emergency login
   - Toggle to emergency login option
   - Enter Student ID and password
   - Access granted with same privileges

## Benefits

### For Students
- ✅ **Convenience:** Face login is fast and easy
- ✅ **Reliability:** Emergency password as backup
- ✅ **Accessibility:** Works even with face obstructions
- ✅ **Security:** Two-factor options (biometric + password)

### For Administrators
- ✅ **Reduced support:** Students can self-recover access
- ✅ **Audit trail:** All logins are logged
- ✅ **Flexibility:** Multiple authentication methods
- ✅ **Security:** Strong password requirements

## Technical Implementation

### Database Schema
```sql
-- users table stores emergency login credentials
CREATE TABLE users (
    username TEXT PRIMARY KEY,      -- Student ID
    password TEXT,                  -- Hashed password
    role TEXT,                      -- 'student', 'admin', etc.
    full_name TEXT,
    email TEXT,
    assigned_class TEXT
);

-- students table stores biometric data
CREATE TABLE students (
    id TEXT PRIMARY KEY,            -- Student ID
    name TEXT,
    face_hash TEXT,                 -- Face biometric hash
    face_image_path TEXT,           -- Stored face image
    verified INTEGER DEFAULT 0,     -- Verification status
    -- ... other fields
);
```

### Authentication Flow
```python
# Emergency Login
user = authenticate_user(username, password, cfg)
if user:
    # Create session with emergency auth flag
    user_data = {
        **user,
        'auth_method': 'emergency_password',
        'auth_time': datetime.now().isoformat()
    }
    st.session_state['user'] = user_data

# Face Authentication
student, confidence, message = face_auth.find_matching_student(image_bytes)
if student and confidence >= threshold:
    # Create session with face auth flag
    user_data = {
        **student,
        'auth_method': 'face',
        'auth_time': datetime.now().isoformat(),
        'confidence_score': confidence
    }
    st.session_state['user'] = user_data
```

## Configuration

### Password Policy (in `src/validation.py`)
```python
def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if not any(c.isalpha() for c in password):
        return False, "Password must contain letters"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain numbers"
    return True, None
```

### Face Recognition Threshold (in `src/config.py`)
```python
class AppConfig:
    confidence_threshold: float = 0.75  # 75% confidence required
```

## Usage Examples

### Example 1: Normal Registration
```
Student: John Doe
Batch: 2024
Department: Computer Science (CS)
Section: A
Number: 001

Generated:
- Student ID: 24CS1001
- Username: 24CS1001
- Password: (student creates, e.g., "john2024")
- Face: (captured via camera)

Login Options:
1. Face Authentication (primary)
2. Emergency: 24CS1001 / john2024
```

### Example 2: Emergency Login Scenario
```
Situation: Student wearing mask, face not visible

Action:
1. Go to Face Authentication page
2. Click "🆘 Emergency Login"
3. Enter Student ID: 24CS1001
4. Enter Password: john2024
5. Click Login
6. Access granted!
```

## Future Enhancements

### Potential Improvements
- [ ] Password reset via email
- [ ] Two-factor authentication (2FA)
- [ ] Biometric + password requirement for sensitive actions
- [ ] Password expiry and rotation policy
- [ ] Account lockout after failed attempts
- [ ] Security questions for password recovery
- [ ] SMS-based OTP for emergency login

## Support

### For Students
- **Forgot Password:** Contact administrator
- **Face Not Working:** Use emergency login
- **Registration Issues:** Contact IT support

### For Administrators
- **Password Reset:** Use admin dashboard
- **View Login History:** Check event logs
- **Security Settings:** Configure in `src/config.py`

## Conclusion

The Emergency Login feature provides a robust backup authentication method while maintaining the convenience and security of face-based biometric authentication. Students can now access the system even when face recognition is not available, ensuring continuous access to the platform.

---

**Last Updated:** December 3, 2025
**Version:** 1.0
**Status:** ✅ Implemented and Active
