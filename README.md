# Student Attire Verification System

A professional computer vision system for verifying student dress code compliance using face authentication and AI-powered attire detection.

## Features

- **Face Authentication** - Biometric login using face recognition
- **Student Registration** - 4-stage registration with face capture
- **Attire Verification** - AI-powered dress code compliance checking
- **Admin Dashboard** - Comprehensive student and compliance management
- **Department Management** - Organize students by departments and classes
- **Real-time Analysis** - Image, webcam, and video verification support

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for face authentication)
- Windows OS

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   streamlit run app\streamlit_app.py
   ```

   Or simply double-click `RUN_APP.bat`

3. **Access the app:**
   Open your browser at `http://localhost:8501`

## Usage

### For Students

1. **Register** - Complete 4-stage registration:
   - Generate Student ID
   - Enter personal details
   - Set emergency password
   - Capture face biometric

2. **Login** - Use face authentication to access your dashboard

3. **Verify Attire** - Upload photos or use webcam to check dress code compliance

### For Administrators

1. **Login** with admin credentials:
   - Username: `admin`
   - Password: `admin123`

2. **Manage Students** - View, add, or remove students

3. **View Reports** - Access compliance statistics and reports

4. **Manage Departments** - Organize classes and departments

## Dress Code Verification

The system checks for:
- ✅ Formal shirt (any color)
- ✅ Full-length pants (any color)
- ✅ Closed shoes (any color)

The system focuses on formal attire detection, not specific colors or poses.

## Project Structure

```
studentattire/
├── app/
│   └── streamlit_app.py          # Main application
├── src/
│   ├── auth.py                    # Authentication logic
│   ├── db.py                      # Database operations
│   ├── verify.py                  # Verification logic
│   ├── features.py                # Feature extraction
│   ├── face_authentication.py     # Face recognition
│   └── ui/                        # UI components
├── data/
│   ├── attire.db                  # SQLite database
│   └── face_storage/              # Face images
├── models/                        # ML models
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

## Troubleshooting

### Camera not working
- Allow camera permissions in Windows Settings
- Close other apps using the camera
- Try a different browser

### "streamlit: command not found"
```bash
pip install -r requirements.txt
```

### Port already in use
```bash
streamlit run app\streamlit_app.py --server.port 8502
```

## Documentation

- `HOW_TO_RUN.md` - Detailed setup and usage guide
- `ALL_FIXES_SUMMARY.md` - Technical fixes and solutions

## License

This project is for educational purposes.

## Support

For issues or questions, refer to the documentation files or check the troubleshooting section above.
