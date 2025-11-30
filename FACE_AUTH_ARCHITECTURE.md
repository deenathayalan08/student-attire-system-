# Face Authentication System - Visual Architecture

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STUDENT ATTIRE VERIFICATION SYSTEM                       │
│                        WITH FACE AUTHENTICATION                             │
└─────────────────────────────────────────────────────────────────────────────┘

                              STREAMLIT APP
                        (app/streamlit_app.py)
                                ├─ Home
                                ├─ Student Verification
                                ├─ Face Authentication ← NEW
                                ├─ Admin Dashboard
                                └─ Profile ← NEW


┌──────────────────────────────────────────────────────────────────────────────┐
│                         STUDENT REGISTRATION FLOW                            │
└──────────────────────────────────────────────────────────────────────────────┘

            ┌──────────────────────────────────────────────┐
            │    Stage 1: Auto-Generate Student ID         │
            │  ┌────────────────────────────────────────┐  │
            │  │ Input:                                 │  │
            │  │ • Batch Year (2024)                    │  │
            │  │ • Department (CS)                      │  │
            │  │ • Section (A)                          │  │
            │  │ • Student Number (001)                 │  │
            │  │                                        │  │
            │  │ Output:                                │  │
            │  │ • Student ID: 22CS1001 ──┐             │  │
            │  │ • Roll Number: 22CS1001 ──┼─ SAME     │  │
            │  └────────────────────────────┘─────────┘  │
            └──────────────────────────────────────────────┘
                              ↓
            ┌──────────────────────────────────────────────┐
            │    Stage 2: Student Details                  │
            │  ┌────────────────────────────────────────┐  │
            │  │ Input:                                 │  │
            │  │ • Full Name                            │  │
            │  │ • Email                                │  │
            │  │ • Phone                                │  │
            │  │ • Gender                               │  │
            │  │ • Contact Info                         │  │
            │  │ • Username & Password                  │  │
            │  │                                        │  │
            │  │ Store in Session                       │  │
            │  └────────────────────────────────────────┘  │
            └──────────────────────────────────────────────┘
                              ↓
            ┌──────────────────────────────────────────────────┐
            │    Stage 3: Face Capture & Biometric            │
            │  ┌──────────────────────────────────────────┐   │
            │  │ FaceAuthenticator (src/face_authentication.py) │
            │  │                                         │   │
            │  │ 1. Capture face from webcam            │   │
            │  │ 2. Detect face using Haar Cascade      │   │
            │  │ 3. Validate quality:                   │   │
            │  │    • Brightness (50-200)               │   │
            │  │    • Size (5%-80% of image)            │   │
            │  │    • Single face only                  │   │
            │  │ 4. Extract facial features (LBP)       │   │
            │  │ 5. Generate SHA-256 hash               │   │
            │  │ 6. Save encrypted face image           │   │
            │  │                                         │   │
            │  │ Output:                                 │   │
            │  │ • face_hash = SHA256(features)         │   │
            │  │ • face_image_path = face_storage/...   │   │
            │  └──────────────────────────────────────────┘   │
            └──────────────────────────────────────────────────┘
                              ↓
            ┌──────────────────────────────────────────────┐
            │    Database Storage (students table)         │
            │  ┌────────────────────────────────────────┐  │
            │  │ INSERT INTO students VALUES (           │  │
            │  │   id='22CS1001',                        │  │
            │  │   roll_no='22CS1001',                   │  │
            │  │   name='John Doe',                      │  │
            │  │   email='john@school.com',              │  │
            │  │   phone='9876543210',                   │  │
            │  │   gender='M',                           │  │
            │  │   department='CS',                      │  │
            │  │   class='CS-A',                         │  │
            │  │   face_hash='a1b2c3d4e5...',            │  │
            │  │   face_image_path='face_storage/...',   │  │
            │  │   verified=1                            │  │
            │  │ )                                       │  │
            │  └────────────────────────────────────────┘  │
            └──────────────────────────────────────────────┘
                              ↓
                    ✅ Registration Complete


┌──────────────────────────────────────────────────────────────────────────────┐
│                         FACE AUTHENTICATION LOGIN FLOW                        │
└──────────────────────────────────────────────────────────────────────────────┘

            ┌──────────────────────────────────────────────┐
            │   User: Click "Face Authentication"          │
            └──────────────────────────────────────────────┘
                              ↓
            ┌──────────────────────────────────────────────┐
            │   Capture Face Image                         │
            │   (show_face_authentication)                 │
            │  ┌────────────────────────────────────────┐  │
            │  │ Input: Webcam image                    │  │
            │  │                                        │  │
            │  │ Process:                               │  │
            │  │ • Detect face in image                 │  │
            │  │ • Validate face quality                │  │
            │  │ • Check for multiple faces             │  │
            │  └────────────────────────────────────────┘  │
            └──────────────────────────────────────────────┘
                              ↓
            ┌──────────────────────────────────────────────┐
            │   Enter Student ID                           │
            │  ┌────────────────────────────────────────┐  │
            │  │ Input: Student ID (e.g., 22CS1001)     │  │
            │  │                                        │  │
            │  │ Query:                                 │  │
            │  │ SELECT * FROM students                 │  │
            │  │ WHERE id='22CS1001'                    │  │
            │  └────────────────────────────────────────┘  │
            └──────────────────────────────────────────────┘
                              ↓
            ┌──────────────────────────────────────────────┐
            │   Display Student Information                │
            │  ┌────────────────────────────────────────┐  │
            │  │ Personal Details:                      │  │
            │  │ ├─ Name: John Doe                      │  │
            │  │ ├─ Student ID: 22CS1001                │  │
            │  │ ├─ Roll Number: 22CS1001 ✨ AUTO-GEN   │  │
            │  │ └─ Department: Computer Science        │  │
            │  │                                        │  │
            │  │ Academic Details:                      │  │
            │  │ ├─ Class: CS-A                         │  │
            │  │ ├─ Gender: Male                        │  │
            │  │ ├─ Email: john@school.com              │  │
            │  │ └─ Phone: 9876543210                   │  │
            │  │                                        │  │
            │  │ Authentication Details:                │  │
            │  │ ├─ Login Time: 14:30:45                │  │
            │  │ ├─ Date: 29-11-2024                    │  │
            │  │ ├─ Day: Friday                         │  │
            │  │ └─ Full: Friday, Nov 29, 2024 @ 14:30 │  │
            │  │                                        │  │
            │  │ 📸 Captured Face: [Image Display]      │  │
            │  └────────────────────────────────────────┘  │
            └──────────────────────────────────────────────┘
                              ↓
            ┌──────────────────────────────────────────────┐
            │   Confirm Login Button                       │
            │  ┌────────────────────────────────────────┐  │
            │  │ Action: Click "Confirm Login"          │  │
            │  │                                        │  │
            │  │ Results:                               │  │
            │  │ 1. Create session state                │  │
            │  │ 2. Log event to events table           │  │
            │  │ 3. Set verified status                 │  │
            │  │ 4. Display success message             │  │
            │  │ 5. Redirect to authenticated area      │  │
            │  └────────────────────────────────────────┘  │
            └──────────────────────────────────────────────┘
                              ↓
                    ✅ Logged In Successfully


┌──────────────────────────────────────────────────────────────────────────────┐
│                         DATABASE SCHEMA                                       │
└──────────────────────────────────────────────────────────────────────────────┘

STUDENTS TABLE
┌──────────────────────────────────────────────────────────────────────┐
│ id (PK)     | roll_no (UNIQUE) | name          | email              │
├─────────────┼──────────────────┼───────────────┼────────────────────┤
│ 22CS1001    │ 22CS1001         │ John Doe      │ john@school.com    │
│ 22CS1002    │ 22CS1002         │ Jane Smith    │ jane@school.com    │
│ 24CS1001    │ 24CS1001         │ Alex Johnson  │ alex@school.com    │
└──────────────────────────────────────────────────────────────────────┘

│ phone        | gender | department    | class    | face_hash          │
├──────────────┼────────┼───────────────┼──────────┼────────────────────┤
│ 9876543210   │ M      │ CS            │ CS-A     │ a1b2c3d4e5f6...    │
│ 9876543211   │ F      │ CS            │ CS-A     │ b2c3d4e5f6g7...    │
│ 9876543212   │ M      │ CS            │ CS-B     │ c3d4e5f6g7h8...    │
└──────────────┼────────┼───────────────┼──────────┼────────────────────┘

│ face_image_path              | verified | contact_info        │
├──────────────────────────────┼──────────┼─────────────────────┤
│ face_storage/22CS1001_*.jpg  │ 1        │ Emergency: 9998...  │
│ face_storage/22CS1002_*.jpg  │ 1        │ Emergency: 9999...  │
│ face_storage/24CS1001_*.jpg  │ 1        │ Emergency: 9997...  │
└──────────────────────────────┴──────────┴─────────────────────┘

EVENTS TABLE (Face Auth Events)
┌─────┬─────────────┬──────────────────┬───────────────────┬────────┬───────┐
│ id  │ student_id  │ timestamp        │ zone              │ status │ label │
├─────┼─────────────┼──────────────────┼───────────────────┼────────┼───────┤
│ 101 │ 22CS1001    │ 2024-11-29 14:30 │ Face Auth Entry   │ PASS   │ Face  │
│ 102 │ 22CS1002    │ 2024-11-29 15:45 │ Face Auth Entry   │ PASS   │ Face  │
│ 103 │ 24CS1001    │ 2024-11-29 16:20 │ Face Auth Entry   │ PASS   │ Face  │
└─────┴─────────────┴──────────────────┴───────────────────┴────────┴───────┘


┌──────────────────────────────────────────────────────────────────────────────┐
│                    COMPONENT INTERACTION DIAGRAM                              │
└──────────────────────────────────────────────────────────────────────────────┘

                                STREAMLIT APP
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
            auth_ui.py      face_login_ui.py   streamlit_app.py
            (3-stage)       (face auth)         (navigation)
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                        FaceAuthenticator
                        (face_authentication.py)
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                OpenCV          NumPy          PIL (Image)
                (Detection)     (Features)     (Processing)
                    │               │               │
                    └───────────────┼───────────────┘
                                    │
                            Database Layer
                            (db.py functions)
                                    │
                        ┌───────────┴───────────┐
                        │                       │
                    SQLite DB            Face Storage
                    (students,          (face_storage/)
                    events, users)       (encrypted images)


┌──────────────────────────────────────────────────────────────────────────────┐
│                      FACE HASH GENERATION PROCESS                             │
└──────────────────────────────────────────────────────────────────────────────┘

Image Input
   │
   ├─→ Convert to BGR (if RGB)
   │
   ├─→ Detect Face Regions
   │   └─→ Haar Cascade Classifier
   │
   ├─→ Extract Face ROI (Region of Interest)
   │
   ├─→ Resize to 200x200 pixels
   │
   ├─→ Convert to Grayscale
   │
   ├─→ Extract Features (LBP - Local Binary Pattern)
   │   └─→ Compute histogram of features
   │
   ├─→ Normalize Features
   │
   ├─→ Convert to Bytes
   │
   └─→ Apply SHA-256 Hash
       │
       └─→ face_hash = "a1b2c3d4e5f6g7h8i9j0..."
           (Stored in database)


┌──────────────────────────────────────────────────────────────────────────────┐
│                         NAVIGATION FLOW                                       │
└──────────────────────────────────────────────────────────────────────────────┘

MAIN MENU
│
├─→ Home
│   ├─ Register (New Student) ──→ 3-Stage Registration ──→ Face Capture
│   └─ Login (Existing User) ──→ Username/Password Auth
│
├─→ Student Verification
│   ├─ Image Upload
│   ├─ Webcam Capture
│   └─ Video Upload
│
├─→ Face Authentication ✨ NEW
│   └─ Capture Face ──→ Enter Student ID ──→ Display Info ──→ Confirm Login
│
├─→ Admin Dashboard
│   ├─ Students Management
│   ├─ Departments Management
│   ├─ Add Student (Admin version)
│   └─ Reports & Downloads
│
└─→ Profile (if logged in) ✨ NEW
    └─ View my information & Logout


┌──────────────────────────────────────────────────────────────────────────────┐
│                      ROLL NUMBER GENERATION FORMAT                            │
└──────────────────────────────────────────────────────────────────────────────┘

Roll Number Format: YYDIDN

YY  = Last 2 digits of Batch Year
     Example: 2024 → 24

D   = 2-digit Department ID (from database)
     Example: 01 (for first dept), 02 (for second), etc.

I   = Section Number (1-9)
     Example: 1=A, 2=B, 3=C, etc.

D   = 3-digit Student Number
     Example: 001-999

Examples:
├─ 22CS1001: Batch 2022, CS Dept (01), Section A (1), Student 001
├─ 24CS2015: Batch 2024, CS Dept (01), Section B (2), Student 015
├─ 24ME1042: Batch 2024, ME Dept (02), Section A (1), Student 042
└─ 23EC3089: Batch 2023, EC Dept (03), Section C (3), Student 089


┌──────────────────────────────────────────────────────────────────────────────┐
│                        SECURITY ARCHITECTURE                                  │
└──────────────────────────────────────────────────────────────────────────────┘

Input
  │
  ├─→ Face Quality Validation
  │   ├─ Brightness Check (50-200)
  │   ├─ Size Check (5%-80%)
  │   └─ Face Count Check (only 1)
  │
  ├─→ Feature Extraction
  │   └─ LBP Pattern Analysis
  │
  ├─→ Hash Generation
  │   └─ SHA-256 Encryption
  │
  ├─→ Database Storage
  │   ├─ face_hash (encrypted)
  │   └─ face_image_path (encrypted storage)
  │
  └─→ Access Control
      ├─ Student verification check
      ├─ Event logging
      └─ Audit trail creation

```

---

## Data Flow Summary

1. **Registration Flow:** Student Input → 3 Stages → Face Capture → FaceAuthenticator → Hash Generation → Database Storage
2. **Login Flow:** Face Capture → ID Entry → Database Query → Display Info → Session Creation → Event Logging
3. **Storage:** Face Hash + Image Path in Database (not raw images)
4. **Security:** SHA-256 Hashing + Quality Validation + Event Logging

---

**Visual Architecture Complete** ✅
