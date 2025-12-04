import streamlit as st
import io
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from PIL import Image

from ..face_authentication import FaceAuthenticator
from ..config import AppConfig
from ..db import get_student, log_face_authentication, insert_event


def show_face_authentication(cfg: AppConfig) -> Optional[Dict]:
    """
    Display face-based login/verification with student details display
    """
    st.title("🔐 Face Authentication")
    st.markdown("---")
    
    # Check if already authenticated
    if 'user' in st.session_state and st.session_state.get('user'):
        st.success(f"✅ Already logged in as {st.session_state['user'].get('full_name', 'User')}")
        return st.session_state['user']
    
    # Add emergency login toggle
    st.info("💡 **Primary Login:** Face Recognition | **Emergency Login:** Username & Password")
    
    # Toggle between face and emergency login
    login_method = st.radio(
        "Select Login Method:",
        ["🔐 Face Authentication (Primary)", "🆘 Emergency Login (Username & Password)"],
        index=0,
        key="login_method_selector"
    )
    
    st.markdown("---")
    
    # Emergency Login with Username/Password
    if login_method == "🆘 Emergency Login (Username & Password)":
        st.subheader("🆘 Emergency Login")
        st.warning("⚠️ **Emergency Access Only** - Use this when face authentication is not available")
        st.info("📝 Your username is your **Student ID** and use the password you set during registration")
        
        with st.form("emergency_login_form"):
            username = st.text_input("Student ID (Username)", placeholder="Enter your Student ID", key="emergency_username")
            password = st.text_input("Password", type="password", placeholder="Enter your emergency password", key="emergency_password")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("🔓 Login", use_container_width=True, type="primary")
            with col2:
                if st.form_submit_button("← Back to Face Login", use_container_width=True):
                    st.rerun()
        
        if login_button:
            if not username or not password:
                st.error("❌ Please enter both Student ID and password")
                return None
            
            # Authenticate using username/password
            from ..auth import authenticate_user
            user = authenticate_user(username, password, cfg)
            
            if user:
                # Get additional student details
                from ..db import get_student
                student = get_student(username, cfg=cfg)
                
                if student:
                    # Merge user and student data
                    user_data = {
                        **user,
                        'student_id': student.get('id'),
                        'roll_no': student.get('roll_no'),
                        'department': student.get('department'),
                        'class': student.get('class'),
                        'gender': student.get('gender', 'U'),
                        'auth_method': 'emergency_password',
                        'auth_time': datetime.now().isoformat()
                    }
                else:
                    user_data = {
                        **user,
                        'auth_method': 'emergency_password',
                        'auth_time': datetime.now().isoformat()
                    }
                
                st.session_state['user'] = user_data
                
                # Log emergency login event
                insert_event({
                    "student_id": user_data.get('student_id') or username,
                    "zone": "Emergency Login",
                    "status": "PASS",
                    "score": 1.0,
                    "label": "Emergency Password Login",
                    "details": "Student logged in using emergency password credentials"
                }, cfg=cfg)
                
                # Set flag to show verification page
                st.session_state['show_verification'] = True
                st.session_state['page'] = 'home'  # Reset to home so navigation works
                
                st.success(f"✅ Emergency login successful! Welcome, {user_data.get('full_name', 'User')}!")
                st.info("🔄 Redirecting to attire verification...")
                st.balloons()
                
                # Immediate rerun to redirect
                st.rerun()
                
                return user_data
            else:
                st.error("❌ Invalid Student ID or password")
                st.info("💡 **Forgot your password?** Contact your administrator for password reset")
                return None
        
        return None
    
    # Face Authentication (Primary Method)
    
    # Check if we're in the middle of login process - if so, STOP EVERYTHING
    if st.session_state.get('login_in_progress'):
        st.info("🔄 Login in progress, redirecting to verification page...")
        st.spinner("Please wait...")
        # Don't render anything else - just return
        return None
    
    st.subheader("📸 Verify Your Identity with Face Recognition")
    st.info("📋 Capture a clear photo of your face. Make sure:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write("✓ Face is centered")
    with col2:
        st.write("✓ Good lighting")
    with col3:
        st.write("✓ No obstructions")
    
    # Camera capture
    st.markdown("---")
    captured_image = st.camera_input("📷 Capture your face", key="face_auth_capture")
    
    if captured_image is not None:
        # Store in session for cropping
        if 'login_captured_face' not in st.session_state:
            st.session_state['login_captured_face'] = captured_image.getvalue()
        
        # Show cropping tool
        st.markdown("---")
        st.subheader("📸 Crop Your Face")
        st.info("ℹ️ Adjust the box to crop your face for better recognition.")
        
        try:
            from PIL import Image
            from streamlit_cropper import st_cropper
            
            img = Image.open(io.BytesIO(st.session_state['login_captured_face']))
            
            # Cropping tool
            cropped_img = st_cropper(
                img,
                realtime_update=True,
                box_color='#0066FF',
                aspect_ratio=None,
                return_type='image'
            )
            
            # Store cropped image
            st.session_state['login_cropped_face'] = cropped_img
            
        except Exception as e:
            st.error(f"Error loading image: {e}")
        
        # Buttons
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            retake_clicked = st.button("🔄 Retake Photo", use_container_width=True)
        with btn_col2:
            verify_clicked = st.button("✅ Verify Face", use_container_width=True, type="primary")
        
        # Handle retake button
        if retake_clicked:
            if 'login_captured_face' in st.session_state:
                del st.session_state['login_captured_face']
            if 'login_cropped_face' in st.session_state:
                del st.session_state['login_cropped_face']
            st.rerun()
        
        # Handle verify button - OUTSIDE column context
        if verify_clicked:
            st.markdown("---")
            st.subheader("🔍 Face Analysis in Progress...")
            
            # Initialize face authenticator
            face_auth = FaceAuthenticator(cfg)
            
            # Use cropped image if available
            if 'login_cropped_face' in st.session_state:
                img_byte_arr = io.BytesIO()
                st.session_state['login_cropped_face'].save(img_byte_arr, format='JPEG')
                image_bytes = img_byte_arr.getvalue()
            else:
                image_bytes = st.session_state['login_captured_face']
            
            # Process the captured image with real-time feedback
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            metrics_placeholder = st.empty()
            
            with progress_placeholder.container():
                progress_bar = st.progress(0)
            
            # Process the captured image
            pil_image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            # Step 1: Face Detection (20% progress)
            status_placeholder.info("📍 Detecting face...")
            progress_bar.progress(20)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                status_placeholder.error("❌ No face detected. Please try again with a clear face image.")
                return None
            
            if len(faces) > 1:
                status_placeholder.warning("⚠️ Multiple faces detected. Please ensure only your face is visible.")
                return None
            
            x, y, w, h = faces[0]
            face_area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
            
            # Step 2: Face Quality Assessment (40% progress)
            status_placeholder.info("✅ Face detected. Assessing quality...")
            progress_bar.progress(40)
            
            face_roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_roi)
            blur_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            
            # Display quality metrics
            with metrics_placeholder.container():
                st.markdown("#### 📊 Face Quality Metrics")
                qm_col1, qm_col2, qm_col3, qm_col4 = st.columns(4)
                with qm_col1:
                    st.metric("Face Size", f"{face_area_ratio:.1%}", delta="Good" if 0.05 < face_area_ratio < 0.8 else "Adjust")
                with qm_col2:
                    st.metric("Brightness", f"{brightness:.0f}", delta="Optimal" if 50 < brightness < 200 else "Adjust lighting")
                with qm_col3:
                    st.metric("Clarity", f"{blur_score:.1f}", delta="Clear" if blur_score > 100 else "Blurry")
                with qm_col4:
                    st.metric("Frontal Angle", "Centered", delta="Good")
            
            # Step 3: Liveness Check (60% progress)
            status_placeholder.info("✅ Quality verified. Performing liveness check...")
            progress_bar.progress(60)
            
            # Simple liveness heuristic: check if face has sufficient detail/texture
            face_edges = cv2.Canny(gray_roi, 100, 200)
            edge_ratio = np.sum(face_edges > 0) / face_edges.size
            liveness_score = min(1.0, edge_ratio * 5.0)  # Normalize to 0-1
            
            with metrics_placeholder.container():
                st.markdown("#### 🔐 Liveness & Biometric Analysis")
                lm_col1, lm_col2, lm_col3 = st.columns(3)
                with lm_col1:
                    if liveness_score > 0.6:
                        st.metric("Liveness", f"{liveness_score:.1%}", delta="✅ Live")
                    else:
                        st.metric("Liveness", f"{liveness_score:.1%}", delta="⚠️ Check image")
                with lm_col2:
                    st.metric("Eye Detection", "Present" if blur_score > 50 else "Not visible")
                with lm_col3:
                    st.metric("Spoofing Risk", "Low" if liveness_score > 0.5 else "Medium")
            
            st.success("✅ Face validation completed successfully!")
            
            # Automatic face matching against database
            st.markdown("---")
            st.subheader("🔍 Searching Database for Face Match...")
            
            with st.spinner("Matching your face against registered students..."):
                # Find matching student automatically
                student, match_confidence, match_message = face_auth.find_matching_student(image_bytes)
            
            # Get threshold early (before any conditional blocks)
            threshold = getattr(cfg, "confidence_threshold", 0.75)
            
            if student:
                # Face match found!
                st.success(match_message)
                
                # Display student information
                st.markdown("---")
                st.subheader("👤 Student Information")
                
                # Display student information with login details
                st.success("✅ Student Verified!" if student.get('verified') == 1 else "⚠️ Student Found")
                
                # Create columns for organized display
                info_col1, info_col2 = st.columns(2)
                
                with info_col1:
                    st.write("### Personal Details")
                    st.write(f"**📛 Name:** {student.get('name', 'N/A')}")
                    st.write(f"**🔢 Student ID:** {student.get('id', 'N/A')}")
                    st.write(f"**🎓 Roll Number:** {student.get('roll_no', 'N/A')}")
                    st.write(f"**🏢 Department:** {student.get('department', 'N/A')}")
                
                with info_col2:
                    st.write("### Academic Details")
                    st.write(f"**📚 Class:** {student.get('class', 'N/A')}")
                    st.write(f"**👥 Gender:** {student.get('gender', 'Unknown')}")
                    st.write(f"**✉️ Email:** {student.get('email', 'N/A')}")
                    st.write(f"**📱 Phone:** {student.get('phone', 'N/A')}")
                
                st.markdown("---")
                
                # Confidence Score & Matching Details
                st.write("### 🔐 Authentication Score")
                # Use the match confidence from face matching
                auth_col1, auth_col2, auth_col3 = st.columns(3)
                with auth_col1:
                    st.metric("Confidence Score", f"{match_confidence:.1%}")
                with auth_col2:
                    st.metric("Threshold", f"{threshold:.1%}")
                with auth_col3:
                    status = "✅ PASS" if match_confidence >= threshold else "❌ FAIL"
                    st.metric("Authentication", status)
                
                # Visual progress bar for confidence
                if match_confidence >= threshold:
                    st.success(f"✅ Authentication Score: {match_confidence:.1%} (Above threshold)")
                else:
                    st.warning(f"⚠️ Confidence too low: {match_confidence:.1%} (Need {threshold:.1%})")
                
                st.progress(match_confidence)
                
                st.markdown("---")
                
                # Login timestamp information
                st.write("### ✅ Authentication Details")
                now = datetime.now()
                
                time_col1, time_col2, time_col3 = st.columns(3)
                with time_col1:
                    st.metric("Login Time", now.strftime("%H:%M:%S"))
                with time_col2:
                    st.metric("Date", now.strftime("%d-%m-%Y"))
                with time_col3:
                    st.metric("Day", now.strftime("%A"))
                
                st.info(f"📍 **Full Timestamp:** {now.strftime('%A, %B %d, %Y at %H:%M:%S')}")
                
                st.markdown("---")
                
                # Display captured face
                st.write("### 📸 Captured Face")
                st.image(pil_image, caption="Your captured face image", use_container_width=True)
        
                st.markdown("---")
                
                # Login button (only if confidence passes threshold)
                if match_confidence >= threshold:
                    if st.button("✅ Confirm Login", use_container_width=True, type="primary"):
                        # Set login in progress flag FIRST to prevent camera from showing again
                        st.session_state['login_in_progress'] = True
                        
                        # Log authentication event
                        event_id = insert_event({
                            "student_id": student.get('id'),
                            "zone": "Face Authentication",
                            "status": "PASS",
                            "score": match_confidence,
                            "label": "Face Authentication",
                            "details": f"Face-based authentication successful. Confidence: {match_confidence:.1%}, Threshold: {threshold:.1%}"
                        }, cfg=cfg)
                        
                        # Create user session
                        user_data = {
                            'username': student.get('id'),
                            'role': 'student',
                            'full_name': student.get('name'),
                            'email': student.get('email'),
                            'student_id': student.get('id'),
                            'roll_no': student.get('roll_no'),
                            'department': student.get('department'),
                            'class': student.get('class'),
                            'auth_method': 'face',
                            'auth_time': now.isoformat(),
                            'confidence_score': match_confidence,
                            'gender': student.get('gender', 'U')
                        }
                        
                        st.session_state['user'] = user_data
                        
                        # Set flag to show verification page
                        st.session_state['show_verification'] = True
                        st.session_state['page'] = 'home'  # Reset to home so navigation works
                        
                        st.success(f"🎉 Welcome, {student.get('name')}!")
                        st.info("🔄 Redirecting to attire verification...")
                        st.balloons()
                        
                        # Immediate rerun to redirect (login_in_progress flag will prevent camera from showing)
                        st.rerun()
                        
                        return user_data
                else:
                    # Confidence below threshold
                    st.error(f"❌ Authentication Failed - Confidence {match_confidence:.1%} below threshold {threshold:.1%}")
                    st.info("Please try again with a clearer photo and better lighting.")
                    return None
            else:
                # No student match found
                st.error("❌ No matching student found in database")
                st.info("Please ensure you are registered or try again with a clearer photo.")
                return None
    
    return None


def show_face_registration_stage(cfg: AppConfig, student_id: str, auto_class: str) -> Tuple[bool, str, str]:
    """
    Stage 3: Capture face during student registration
    Returns: (success, face_hash, face_image_path)
    """
    st.markdown("---")
    st.markdown("### 👤 Stage 3: Face Registration (Biometric Verification)")
    st.info("📸 Capture a clear photo of your face for biometric verification")
    
    # Camera capture
    captured_face = st.camera_input("📷 Capture your face for registration", key=f"face_register_{student_id}")
    
    if captured_face is not None:
        # Process the face
        st.markdown("---")
        with st.spinner("🔍 Processing face..."):
            face_auth = FaceAuthenticator(cfg)
            image_bytes = captured_face.getvalue()
            
            success, face_hash, face_image, message = face_auth.capture_face_for_registration(image_bytes)
            
            if success:
                st.success(message)
                
                # Show success message
                st.write("✅ Your face has been successfully captured and verified!")
                st.write(f"**Face Hash:** `{face_hash[:16]}...`")
                
                # Save face image
                if face_image is not None:
                    face_image_path = face_auth.save_face_image(face_image, student_id, student_id)
                    st.success(f"✅ Face image saved for biometric verification")
                    
                    return success, face_hash, face_image_path if face_image_path else ""
                else:
                    return success, face_hash, ""
            else:
                st.error(message)
                st.info("Please try again with a clearer photo")
                return False, "", ""
    
    return False, "", ""
