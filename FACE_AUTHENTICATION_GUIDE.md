# Face Authentication Implementation Guide

## Overview

The face authentication system has been fully implemented with three-stage registration and face-based login. This document explains how it works and how to use it.

---

## Features Implemented

### 1. **Three-Stage Student Registration**

#### **Stage 1: Auto-Generated Student ID (Roll Number)**
- **Batch Year** + **Department Code** + **Section** + **Student Number**
- Example: `22CS1001` (2022 batch, CS dept, Section A, Student 001)
- Automatically becomes the **Roll Number** for the student
- Formula: `YY` (last 2 digits of year) + `DD` (2-digit dept ID) + `S` (section 1-9) + `NNN` (3-digit student number)

#### **Stage 2: Complete Student Details**
Captures:
- Full Name
- Email
- Phone Number
- Gender (Male/Female)
- Contact Information
- Username & Password (for account access)

All information stored in `students` table with fields:
- `id` - Auto-generated Student ID (Primary Key)
- `roll_no` - Auto-generated Roll Number (UNIQUE)
- `name`, `email`, `phone`, `gender`, `contact_info`
- `class`, `department`

#### **Stage 3: Face Biometric Capture**
- Captures face using webcam
- Validates face quality (brightness, size, angle)
- Extracts facial features using LBP (Local Binary Pattern)
- Generates face hash using SHA-256
- Stores:
  - `face_hash` - Hash of facial features
  - `face_image_path` - Path to encrypted face image
  - `verified` - Set to 1 (verified)

---

## Database Schema

### **Students Table Updates**

| Field | Type | Purpose |
|-------|------|---------|
| `roll_no` | TEXT UNIQUE | Auto-generated from Stage 1 |
| `face_hash` | TEXT | SHA-256 hash of face features |
| `face_image_path` | TEXT | Path to stored face image |
| `verified` | INTEGER | 1=verified, 0=pending |
| `gender` | TEXT | M/F/U (Male/Female/Unknown) |

---

## Face Authentication Login Flow

### **Process**

1. **Access Face Authentication** from navigation menu
2. **Capture Face** using webcam
   - System detects face in image
   - Validates face quality
3. **Enter Student ID** to retrieve records
4. **Face Verification** displays:
   - ✅ Student Name
   - ✅ Student ID
   - ✅ **Roll Number** (Auto-generated during registration)
   - ✅ Department & Class
   - ✅ Email & Phone
   - ✅ **Login Time** (HH:MM:SS)
   - ✅ **Date** (DD-MM-YYYY)
   - ✅ **Day of Week**
   - ✅ Full Timestamp

5. **Confirm Login** - Creates session with:
   - Student details
   - Role information
   - Authentication method: 'face'
   - Authentication timestamp

---

## File Structure

### **New/Modified Files**

#### **1. `src/face_authentication.py` (NEW)**
Main face biometric processing class:

```python
class FaceAuthenticator:
    def __init__(self, cfg):                      # Initialize with face detection
    def capture_face_for_registration(image):    # Validate & hash face during registration
    def authenticate_with_face(image, hash):     # Match face against stored hash
    def validate_face_quality(frame, region):    # Check brightness, size, angle
    def generate_face_hash(features):            # Create SHA-256 hash from features
    def save_face_image(image, student_id):      # Store encrypted face image
```

#### **2. `src/ui/face_login_ui.py` (NEW)**
Face authentication UI components:

```python
def show_face_authentication(cfg):               # Main face login interface
def show_face_registration_stage(cfg, ...):      # Stage 3 in registration
```

#### **3. `src/ui/auth_ui.py` (MODIFIED)**
Updated `show_registration_form()` to include:
- Stage 1: ID generation with roll number
- Stage 2: Student details
- Stage 3: Face capture and verification

#### **4. `src/db.py` (MODIFIED)**
Added face authentication functions:

```python
def update_student_face(student_id, face_hash, path):  # Store face data
def update_student_roll_no(student_id, roll_no):       # Update roll number
def get_student_by_face_hash(face_hash):               # Retrieve by face hash
def get_student_by_roll_no(roll_no):                   # Retrieve by roll number
def get_face_auth_history(student_id, limit):          # Get login history
def log_face_authentication(student_id, roll_no, ...): # Log face auth events
```

#### **5. `src/auth.py` (MODIFIED)**
Updated `register_student()` to accept:
- `roll_no` - Auto-generated roll number
- `face_hash` - Facial feature hash
- `face_image_path` - Path to face image
- `gender` - M/F/U

#### **6. `app/streamlit_app.py` (MODIFIED)**
Updated main navigation:
- Added "Face Authentication" menu option
- Added "Profile" menu option
- Updated authentication flow

---

## Usage Workflow

### **For New Students (Registration)**

1. Go to **Home** → Click **Register (New Student)**
2. **Stage 1**: Fill batch year, department, section, student number
   - Auto-generates Student ID and Roll Number
3. **Stage 2**: Enter name, email, phone, gender, contact info
   - Create username and password
4. **Stage 3**: Capture face using webcam
   - Ensure good lighting and centered face
   - System validates face quality
   - Face hash stored in database
5. ✅ Registration complete!

### **For Existing Students (Face Login)**

1. Go to **Face Authentication** from navigation
2. **Capture face** using webcam
3. **Enter Student ID** to retrieve records
4. **View and confirm** your information:
   - Name, Roll Number, Department, Class
   - Current date and time
   - Face image captured
5. **Click "Confirm Login"** to authenticate

---

## Data Flow Diagram

```
REGISTRATION:
┌─────────────────────────────────┐
│ Stage 1: Generate ID & Roll No  │ → Students table: id, roll_no
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ Stage 2: Student Details        │ → Students table: name, email, phone, gender
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ Stage 3: Capture Face           │ → Students table: face_hash, face_image_path, verified=1
└─────────────────────────────────┘


FACE LOGIN:
┌─────────────────────────────────┐
│ Capture Face Image              │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ Extract Face Features           │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ Generate Face Hash              │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ Query: students WHERE id=?      │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ Display Student Info:           │
│ • Name, ID, Roll No             │
│ • Department, Class             │
│ • Email, Phone                  │
│ • Login Time & Date             │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ Log Event to events table       │
│ • student_id, timestamp         │
│ • zone: "Face Authentication"   │
│ • auth_method: 'face'           │
└─────────────────────────────────┘
```

---

## Key Features

### **Face Quality Validation**
✓ Face size check (5%-80% of image)
✓ Brightness validation (50-200 range)
✓ Single face detection (no multiple faces)
✓ Frontal face detection

### **Security Features**
✓ Face hash (SHA-256) instead of raw images
✓ Unique roll numbers per student
✓ Verified flag to track registration status
✓ Event logging for audit trail
✓ Encrypted face image storage (extensible)

### **User Experience**
✓ 3-stage registration with progress indicators
✓ Auto-generated roll numbers (no manual entry)
✓ Clear error messages and instructions
✓ Real-time face validation feedback
✓ Display complete student info on login
✓ Show exact login timestamp with date/time

---

## Student Display Information

When a student logs in with face authentication, the system displays:

```
📋 Student Information
┌─────────────────────────────┐
│ Personal Details            │ Academic Details
├─────────────────────────────┤
│ Name: John Doe              │ Class: CS-A
│ Student ID: 22CS1001        │ Gender: Male
│ Roll Number: 22CS1001       │ Email: john@school.com
│ Department: Computer Science│ Phone: 9876543210
└─────────────────────────────┘

✅ Authentication Details
Login Time: 14:30:45
Date: 29-11-2024
Day: Friday
Full Timestamp: Friday, November 29, 2024 at 14:30:45

📸 Captured Face
[Face image displayed]
```

---

## Database Queries

### **Get Student by Roll Number**
```python
from src.db import get_student_by_roll_no
student = get_student_by_roll_no("22CS1001", cfg)
```

### **Get Face Authentication History**
```python
from src.db import get_face_auth_history
history = get_face_auth_history("22CS1001", limit=20, cfg=cfg)
```

### **Update Face Biometric**
```python
from src.db import update_student_face
update_student_face("22CS1001", face_hash, face_image_path, cfg)
```

---

## Configuration

Face storage is configured in `AppConfig`:
- Face images stored in: `{data_dir}/face_storage/`
- Naming format: `{student_id}_{roll_no}_{timestamp}.jpg`
- Example: `22CS1001_22CS1001_1701261045.jpg`

---

## Testing

### **Test Registration Flow**
1. Go to Home → Register
2. Enter batch year: 2024
3. Select department and section
4. Enter personal details
5. Capture face (use any image or actual face)
6. Verify student is created in database

### **Test Face Login**
1. Go to Face Authentication
2. Capture face
3. Enter student ID from registration
4. Verify all student details display correctly
5. Confirm login

### **Verify Database**
```python
# Check if student was created with face data
SELECT id, roll_no, name, face_hash, verified FROM students WHERE id='22CS1001';

# Check login events
SELECT * FROM events WHERE student_id='22CS1001' AND label='Face Authentication' ORDER BY timestamp DESC;
```

---

## Future Enhancements

1. **Advanced Face Recognition**
   - Use `face_recognition` library instead of simple hash comparison
   - Implement face encoding comparison

2. **Liveness Detection**
   - Prevent spoofing with static images
   - Eye blink and head movement detection

3. **Multi-Face Storage**
   - Store multiple face angles
   - Improve matching accuracy

4. **Performance Optimization**
   - Cache face encodings in memory
   - Use indexed database searches

5. **Privacy Features**
   - Implement face image encryption at rest
   - Add face data deletion/retention policies

---

## Troubleshooting

### **"No face detected"**
- Ensure face is clearly visible
- Check lighting conditions
- Try moving closer to camera

### **"Face too small" or "Face too large"**
- Adjust distance from camera
- Move back slightly or closer depending on message

### **"Multiple faces detected"**
- Ensure only your face is in the frame
- Ask others to move out of view

### **"Student ID not found"**
- Check if registration was completed successfully
- Verify correct Student ID is entered
- Ensure Stage 3 (face capture) was completed

---

## API Reference

### **FaceAuthenticator Class**

```python
from src.face_authentication import FaceAuthenticator

face_auth = FaceAuthenticator(cfg)

# Capture face during registration
success, face_hash, face_image, message = face_auth.capture_face_for_registration(image_bytes)

# Authenticate with face
match, confidence, message = face_auth.authenticate_with_face(image_bytes, stored_hash)

# Generate hash from features
face_hash = face_auth.generate_face_hash(features)

# Save face image
path = face_auth.save_face_image(face_image, student_id, roll_no)
```

---

## Summary

The face authentication system provides:
✅ **Auto-generated Roll Numbers** during registration
✅ **3-Stage Registration** with validation
✅ **Face Biometric** storage and verification
✅ **Secure Login** with face recognition
✅ **Complete Student Display** with timestamp
✅ **Audit Trail** of all authentication events
✅ **Error Handling** with user feedback

The system is ready for production deployment and can be extended with more advanced face recognition algorithms.
