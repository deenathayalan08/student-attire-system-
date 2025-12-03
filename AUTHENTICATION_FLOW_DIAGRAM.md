# 🔐 Authentication Flow Diagram

## Complete System Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        HOME PAGE                                 │
│                                                                  │
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  Register            │      │  Login               │        │
│  │  (New Student)       │      │  (Existing User)     │        │
│  └──────────────────────┘      └──────────────────────┘        │
│           │                              │                       │
└───────────┼──────────────────────────────┼──────────────────────┘
            │                              │
            ▼                              ▼
┌───────────────────────┐      ┌───────────────────────────────┐
│  REGISTRATION FLOW    │      │     LOGIN FLOW                │
│  (4 Stages)           │      │     (2 Methods)               │
└───────────────────────┘      └───────────────────────────────┘
            │                              │
            │                              │
            ▼                              ▼
```

## Registration Flow (4 Stages)

```
┌─────────────────────────────────────────────────────────────────┐
│                     STAGE 1: ID GENERATION                       │
├─────────────────────────────────────────────────────────────────┤
│  Input:                                                          │
│  • Batch Year (2024)                                            │
│  • Department (Computer Science)                                │
│  • Section (A)                                                  │
│  • Student Number (001)                                         │
│                                                                  │
│  Output:                                                         │
│  • Student ID: 24CS1001                                         │
│  • Class: CS-A                                                  │
│  • Roll No: 24CS1001                                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STAGE 2: STUDENT DETAILS                      │
├─────────────────────────────────────────────────────────────────┤
│  Input:                                                          │
│  • Full Name (John Doe)                                         │
│  • Email (john@example.com)                                     │
│  • Phone (optional)                                             │
│  • Gender (Male/Female)                                         │
│  • Contact Info (optional)                                      │
│  • Terms Agreement ✓                                            │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              STAGE 3: EMERGENCY LOGIN SETUP ⚠️ NEW              │
├─────────────────────────────────────────────────────────────────┤
│  Purpose: Backup login when face authentication fails           │
│                                                                  │
│  Username: 24CS1001 (Auto-set from Stage 1)                    │
│                                                                  │
│  Password: [Student Creates]                                    │
│  • Minimum 6 characters                                         │
│  • Must contain letters AND numbers                            │
│  • Password confirmation required                               │
│  • Example: john2024, mary123abc                               │
│                                                                  │
│  Security:                                                       │
│  • SHA-256 hashing with salt                                   │
│  • Never stored in plain text                                  │
│  • Unique salt per user                                        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                STAGE 4: FACE CAPTURE (BIOMETRIC)                │
├─────────────────────────────────────────────────────────────────┤
│  Process:                                                        │
│  1. Camera permission request                                   │
│  2. Face capture via camera                                     │
│  3. Face cropping tool                                          │
│  4. Face quality validation:                                    │
│     • Face size check                                           │
│     • Brightness check                                          │
│     • Clarity check                                             │
│     • Liveness detection                                        │
│  5. Face hash generation (SHA-256)                             │
│  6. Face image storage (encrypted)                             │
│                                                                  │
│  Output:                                                         │
│  • Face Hash: abc123...                                         │
│  • Face Image Path: data/face_storage/24CS1001_xxx.jpg        │
│  • Verification Status: Verified ✓                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   REGISTRATION COMPLETE ✅                       │
├─────────────────────────────────────────────────────────────────┤
│  Database Records Created:                                       │
│  • users table: username, password (hashed)                     │
│  • students table: face_hash, face_image_path                  │
│                                                                  │
│  Login Methods Available:                                        │
│  1. Face Authentication (Primary)                               │
│  2. Emergency Login (Backup)                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Login Flow (2 Methods)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FACE AUTHENTICATION PAGE                      │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │  Select Login Method:   │
              │  ○ Face Authentication  │
              │  ○ Emergency Login      │
              └─────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            │                               │
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│  METHOD 1: FACE AUTH  │       │  METHOD 2: EMERGENCY  │
│  (Primary) 🔐         │       │  (Backup) 🆘          │
└───────────────────────┘       └───────────────────────┘
            │                               │
            ▼                               ▼
```

### Method 1: Face Authentication (Primary)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FACE AUTHENTICATION FLOW                      │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: Camera Capture                                         │
│  • Capture face image                                           │
│  • Crop face region                                             │
│  • Preview captured image                                       │
│                                                                  │
│  Step 2: Face Detection (20% progress)                          │
│  • Detect face in image                                         │
│  • Validate single face                                         │
│  • Check face position                                          │
│                                                                  │
│  Step 3: Quality Assessment (40% progress)                      │
│  • Face size ratio: 5-80%                                       │
│  • Brightness: 50-200                                           │
│  • Clarity/Blur score: >100                                     │
│  • Frontal angle check                                          │
│                                                                  │
│  Step 4: Liveness Check (60% progress)                          │
│  • Edge detection                                               │
│  • Texture analysis                                             │
│  • Spoofing detection                                           │
│  • Liveness score: >60%                                         │
│                                                                  │
│  Step 5: Database Matching (80% progress)                       │
│  • Extract face features                                        │
│  • Generate face hash                                           │
│  • Compare with all registered students                         │
│  • Find best match                                              │
│                                                                  │
│  Step 6: Confidence Check (100% progress)                       │
│  • Match confidence: X%                                         │
│  • Threshold: 75%                                               │
│  • Decision: PASS/FAIL                                          │
│                                                                  │
│  Result:                                                         │
│  ✅ Match Found (confidence ≥ 75%)                              │
│     → Display student information                               │
│     → Show authentication score                                 │
│     → Login button enabled                                      │
│                                                                  │
│  ❌ No Match / Low Confidence                                   │
│     → Show error message                                        │
│     → Suggest improvements                                      │
│     → Offer emergency login option                              │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOGIN SUCCESSFUL ✅                           │
├─────────────────────────────────────────────────────────────────┤
│  Session Created:                                                │
│  • User data stored in session                                  │
│  • Auth method: 'face'                                          │
│  • Auth time: timestamp                                         │
│  • Confidence score: X%                                         │
│                                                                  │
│  Event Logged:                                                   │
│  • Student ID                                                   │
│  • Zone: "Face Authentication"                                  │
│  • Status: "PASS"                                               │
│  • Details: confidence score                                    │
│                                                                  │
│  Redirect: Student Dashboard                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Method 2: Emergency Login (Backup)

```
┌─────────────────────────────────────────────────────────────────┐
│                    EMERGENCY LOGIN FLOW                          │
├─────────────────────────────────────────────────────────────────┤
│  When to Use:                                                    │
│  • Face not visible (mask, poor lighting)                       │
│  • Camera not available                                         │
│  • Face recognition fails                                       │
│  • Emergency situations                                         │
│                                                                  │
│  Input Form:                                                     │
│  ┌────────────────────────────────────────┐                    │
│  │ Student ID (Username): [24CS1001    ]  │                    │
│  │ Password:              [••••••••    ]  │                    │
│  │                                        │                    │
│  │  [🔓 Login]  [← Back to Face Login]   │                    │
│  └────────────────────────────────────────┘                    │
│                                                                  │
│  Validation:                                                     │
│  1. Check username exists                                       │
│  2. Verify password hash                                        │
│  3. Load student data                                           │
│                                                                  │
│  Authentication:                                                 │
│  • Hash input password with stored salt                         │
│  • Compare with stored hash                                     │
│  • Match: ✅ Login successful                                   │
│  • No match: ❌ Invalid credentials                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOGIN SUCCESSFUL ✅                           │
├─────────────────────────────────────────────────────────────────┤
│  Session Created:                                                │
│  • User data stored in session                                  │
│  • Auth method: 'emergency_password'                            │
│  • Auth time: timestamp                                         │
│                                                                  │
│  Event Logged:                                                   │
│  • Student ID                                                   │
│  • Zone: "Emergency Login"                                      │
│  • Status: "PASS"                                               │
│  • Details: "Emergency password login"                          │
│                                                                  │
│  Redirect: Student Dashboard                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

```
┌─────────────────────────────────────────────────────────────────┐
│                         USERS TABLE                              │
├─────────────────────────────────────────────────────────────────┤
│  username (PK)     │ Student ID (e.g., 24CS1001)               │
│  password          │ Hashed password (salt:hash)               │
│  role              │ 'student', 'admin', 'teacher'             │
│  full_name         │ Student full name                         │
│  email             │ Student email                             │
│  assigned_class    │ Class code (e.g., CS-A)                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                       STUDENTS TABLE                             │
├─────────────────────────────────────────────────────────────────┤
│  id (PK)           │ Student ID (e.g., 24CS1001)               │
│  name              │ Student full name                         │
│  roll_no           │ Roll number (same as ID)                  │
│  class             │ Class code (e.g., CS-A)                   │
│  department        │ Department name                           │
│  gender            │ M/F/U                                     │
│  email             │ Student email                             │
│  phone             │ Phone number                              │
│  face_hash         │ Face biometric hash                       │
│  face_image_path   │ Path to stored face image                 │
│  verified          │ 0=unverified, 1=verified                  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        EVENTS TABLE                              │
├─────────────────────────────────────────────────────────────────┤
│  id (PK)           │ Auto-increment                            │
│  student_id (FK)   │ Student ID                                │
│  timestamp         │ Event timestamp                           │
│  zone              │ "Face Authentication", "Emergency Login"  │
│  status            │ "PASS", "FAIL"                            │
│  score             │ Confidence score (0.0-1.0)                │
│  label             │ Event type label                          │
│  details           │ Additional details                        │
└─────────────────────────────────────────────────────────────────┘
```

## Security Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                      PASSWORD SECURITY                           │
├─────────────────────────────────────────────────────────────────┤
│  Registration:                                                   │
│  1. Student enters password: "john2024"                         │
│  2. Generate random salt: "a1b2c3d4..."                         │
│  3. Hash password + salt: SHA-256("john2024a1b2c3d4...")       │
│  4. Store: "a1b2c3d4...:hashed_value"                          │
│                                                                  │
│  Login:                                                          │
│  1. Student enters password: "john2024"                         │
│  2. Retrieve stored: "a1b2c3d4...:hashed_value"                │
│  3. Extract salt: "a1b2c3d4..."                                │
│  4. Hash input + salt: SHA-256("john2024a1b2c3d4...")          │
│  5. Compare hashes: match = login success                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FACE BIOMETRIC SECURITY                       │
├─────────────────────────────────────────────────────────────────┤
│  Registration:                                                   │
│  1. Capture face image                                          │
│  2. Extract face features (LBP, histogram)                      │
│  3. Generate face hash: SHA-256(features)                       │
│  4. Store face hash in database                                 │
│  5. Encrypt and store face image                                │
│                                                                  │
│  Login:                                                          │
│  1. Capture face image                                          │
│  2. Extract face features                                       │
│  3. Generate face hash                                          │
│  4. Compare with all stored face hashes                         │
│  5. Calculate similarity score (0-100%)                         │
│  6. If score ≥ 75%: login success                              │
└─────────────────────────────────────────────────────────────────┘
```

## Error Handling

```
┌─────────────────────────────────────────────────────────────────┐
│                      ERROR SCENARIOS                             │
├─────────────────────────────────────────────────────────────────┤
│  Face Authentication Errors:                                     │
│  • No face detected → Suggest better lighting/positioning       │
│  • Multiple faces → Ask to be alone in frame                    │
│  • Low confidence → Offer emergency login                       │
│  • Camera error → Redirect to emergency login                   │
│                                                                  │
│  Emergency Login Errors:                                         │
│  • Invalid username → Check Student ID format                   │
│  • Wrong password → Suggest password reset                      │
│  • Account locked → Contact administrator                       │
│                                                                  │
│  Registration Errors:                                            │
│  • Duplicate Student ID → Contact admin                         │
│  • Weak password → Show requirements                            │
│  • Face quality low → Retake photo                              │
└─────────────────────────────────────────────────────────────────┘
```

---

**Legend:**
- 🔐 = Primary/Secure method
- 🆘 = Emergency/Backup method
- ✅ = Success state
- ❌ = Error state
- ⚠️ = Warning/Important
- → = Flow direction
- ▼ = Next step

**Last Updated:** December 3, 2025
