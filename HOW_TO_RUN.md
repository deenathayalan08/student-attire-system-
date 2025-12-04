# 🚀 How to Run the Student Attire Verification System

## Quick Start (3 Steps)

### Step 1: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 2: Apply Navigation Fix (Optional but Recommended)
```powershell
python apply_navigation_fix.py
```

### Step 3: Run the Application
```powershell
streamlit run app\streamlit_app.py
```

That's it! The app will open in your browser at `http://localhost:8501`

---

## Detailed Setup Guide

### Prerequisites

✅ **Python 3.8+** installed
✅ **pip** package manager
✅ **Webcam** (for face authentication)
✅ **Windows** (you're already on Windows)

### Step-by-Step Installation

#### 1. Check Python Installation

```powershell
python --version
```

Should show Python 3.8 or higher. If not installed, download from [python.org](https://www.python.org/downloads/)

#### 2. Navigate to Project Directory

```powershell
cd C:\Users\DEENA\Documents\studentattire
```

#### 3. Create Virtual Environment (Recommended)

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1
```

If you get an error about execution policy:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

This will install:
- streamlit (web framework)
- opencv-python (computer vision)
- mediapipe (pose detection)
- scikit-learn (ML models)
- pandas, numpy (data processing)
- And more...

#### 5. Verify Installation

```powershell
# Check if streamlit is installed
streamlit --version

# Should show: Streamlit, version 1.38.0
```

#### 6. Apply Navigation Fix (Recommended)

```powershell
python apply_navigation_fix.py
```

This fixes all navigation issues and makes the app professional.

#### 7. Run the Application

```powershell
streamlit run app\streamlit_app.py
```

The app will automatically open in your default browser at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.x.x:8501 (for other devices on your network)

---

## First Time Usage

### 1. Home Page
When you first run the app, you'll see the home page with options to:
- **Register** (for new students)
- **Face Login** (for existing students)
- **Admin Login** (for administrators)

### 2. Register a New Student

Click **"Register Now"** and complete 4 stages:

**Stage 1: Generate Student ID**
- Select batch year (e.g., 2024)
- Select department
- Select section (A, B, C, etc.)
- Enter student number
- System auto-generates Student ID

**Stage 2: Student Details**
- Enter full name
- Enter email
- Enter phone (optional)
- Select gender
- Agree to terms

**Stage 3: Emergency Login Setup**
- Create a password for emergency login
- This is backup when face auth fails

**Stage 4: Face Registration**
- Allow camera access
- Capture your face photo
- Crop your face
- Confirm and register

### 3. Login with Face Authentication

Click **"Face Login"**:
- Allow camera access
- Capture your face
- System matches against database
- Automatic login if match found

### 4. Admin Access

**Default Admin Credentials:**
- Username: `admin`
- Password: `admin123`

Admin can:
- View all students
- Manage departments
- View compliance reports
- Verify student attire
- Access system settings

---

## Using the System

### For Students:

1. **Login** with face authentication
2. **My Dashboard** - View your profile and history
3. **Verify Attire** - Upload photo or use webcam to check uniform compliance
4. **Profile** - View your details and verification history

### For Admins:

1. **Login** with admin credentials
2. **Admin Dashboard** - View all students and statistics
3. **Verification** - Verify any student's attire
4. **Departments** - Manage departments and classes
5. **Settings** - Configure system policies

### Attire Verification:

**Three Methods:**
1. **Upload Image** - Upload a full-body photo
2. **Webcam** - Capture photo using webcam
3. **Video** - Upload a video for frame-by-frame analysis

**What It Checks:**
- ✅ Proper uniform (shirt, pants)
- ✅ Black shoes (for males)
- ✅ ID card visibility
- ✅ Dress code compliance
- ✅ Footwear requirements

---

## Troubleshooting

### Issue: "streamlit: command not found"

**Solution:**
```powershell
# Make sure virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Or install streamlit globally
pip install streamlit
```

### Issue: "No module named 'cv2'"

**Solution:**
```powershell
pip install opencv-python
```

### Issue: "Camera not working"

**Solutions:**
1. Allow camera permissions in Windows Settings
2. Close other apps using the camera
3. Try a different browser
4. Check if camera is properly connected

### Issue: "Port 8501 already in use"

**Solution:**
```powershell
# Stop existing streamlit processes
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force

# Or run on different port
streamlit run app\streamlit_app.py --server.port 8502
```

### Issue: "Database locked"

**Solution:**
```powershell
# Close all instances of the app
# Delete the lock file if exists
Remove-Item data\attire.db-journal -ErrorAction SilentlyContinue

# Restart the app
streamlit run app\streamlit_app.py
```

### Issue: "Navigation not working"

**Solution:**
```powershell
# Apply the navigation fix
python apply_navigation_fix.py

# Restart the app
streamlit run app\streamlit_app.py
```

---

## Advanced Configuration

### Run on Custom Port

```powershell
streamlit run app\streamlit_app.py --server.port 8080
```

### Run on Network (Access from other devices)

```powershell
streamlit run app\streamlit_app.py --server.address 0.0.0.0
```

Then access from other devices using: `http://YOUR_IP:8501`

### Enable Debug Mode

```powershell
streamlit run app\streamlit_app.py --logger.level=debug
```

### Disable Auto-Reload

```powershell
streamlit run app\streamlit_app.py --server.runOnSave false
```

---

## Project Structure

```
studentattire/
├── app/
│   └── streamlit_app.py          # Main application
├── src/
│   ├── auth.py                    # Authentication
│   ├── db.py                      # Database
│   ├── verify.py                  # Verification logic
│   ├── features.py                # Feature extraction
│   ├── model.py                   # ML model
│   ├── face_authentication.py     # Face recognition
│   └── ui/                        # UI components
├── data/
│   ├── attire.db                  # SQLite database
│   ├── face_storage/              # Face images
│   └── images/                    # Verification images
├── models/                        # Trained ML models
├── requirements.txt               # Dependencies
└── apply_navigation_fix.py        # Navigation fix script
```

---

## Stopping the Application

### Method 1: In Terminal
Press `Ctrl + C` in the terminal where streamlit is running

### Method 2: Close Browser
Just close the browser tab (app keeps running in background)

### Method 3: Kill Process
```powershell
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force
```

---

## Development Mode

### Watch for Changes (Auto-reload)

Streamlit automatically reloads when you save files. Just edit and save!

### Clear Cache

In the browser, click the menu (☰) → "Clear cache"

Or add to your code:
```python
st.cache_data.clear()
```

### View Logs

Logs are shown in the terminal where you ran `streamlit run`

---

## Production Deployment

### Option 1: Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy!

### Option 2: Local Server

```powershell
# Install as Windows service or use PM2
# Or run in background with nohup equivalent
```

### Option 3: Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/streamlit_app.py"]
```

---

## Quick Commands Reference

```powershell
# Install dependencies
pip install -r requirements.txt

# Apply navigation fix
python apply_navigation_fix.py

# Run application
streamlit run app\streamlit_app.py

# Run on custom port
streamlit run app\streamlit_app.py --server.port 8080

# Stop application
Ctrl + C

# Kill all streamlit processes
Get-Process | Where-Object {$_.ProcessName -like "*streamlit*"} | Stop-Process -Force

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Deactivate virtual environment
deactivate

# Update dependencies
pip install -r requirements.txt --upgrade

# Check streamlit version
streamlit --version

# View streamlit help
streamlit --help
```

---

## Testing the System

### Test Checklist:

1. **Home Page**
   - [ ] Loads without errors
   - [ ] Shows welcome message
   - [ ] Navigation buttons work

2. **Registration**
   - [ ] Can generate student ID
   - [ ] Can enter details
   - [ ] Can capture face
   - [ ] Registration completes

3. **Face Login**
   - [ ] Camera works
   - [ ] Face detection works
   - [ ] Login successful
   - [ ] Redirects to verification

4. **Attire Verification**
   - [ ] Image upload works
   - [ ] Webcam capture works
   - [ ] Analysis shows results
   - [ ] Violations detected correctly

5. **Admin Dashboard**
   - [ ] Admin login works
   - [ ] Can view students
   - [ ] Can manage departments
   - [ ] Reports generate

---

## Getting Help

### Resources:
- **Streamlit Docs**: https://docs.streamlit.io
- **OpenCV Docs**: https://docs.opencv.org
- **Project README**: `README.md`
- **Navigation Fix**: `README_NAVIGATION_FIX.md`

### Common Issues:
- Check `requirements.txt` for correct versions
- Ensure Python 3.8+ is installed
- Verify camera permissions
- Check firewall settings

---

## Summary

**To run your project:**

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Apply navigation fix (optional)
python apply_navigation_fix.py

# 3. Run the app
streamlit run app\streamlit_app.py
```

**That's it!** Your app will open at `http://localhost:8501` 🎉

---

**Need help?** Just ask! I'm here to assist you. 😊
