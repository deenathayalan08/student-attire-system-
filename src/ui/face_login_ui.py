"""
Complete Face Login UI with ALL Features:
- Real-time face detection
- Image cropping
- Working retake button
- Proper redirect after login
"""

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


def analyze_face_quality(image_bytes: bytes) -> Tuple[bool, str, dict]:
    """
    Analyze face quality with multiple detection attempts
    Returns: (has_face, message, metrics)
    """
    try:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        pil_image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try multiple detection strategies
        detection_attempts = [
            {'scaleFactor': 1.1, 'minNeighbors': 4, 'minSize': (60, 60)},
            {'scaleFactor': 1.05, 'minNeighbors': 3, 'minSize': (50, 50)},
            {'scaleFactor': 1.05, 'minNeighbors': 2, 'minSize': (40, 40)},
            {'scaleFactor': 1.03, 'minNeighbors': 2, 'minSize': (30, 30)},
        ]
        
        faces = []
        for attempt in detection_attempts:
            faces = face_cascade.detectMultiScale(gray, **attempt)
            if len(faces) > 0:
                break
        
        h, w = frame.shape[:2]
        
        metrics = {
            'face_count': len(faces),
            'face_size': 0.0,
            'brightness': 0.0,
            'clarity': 0.0,
            'centered': False
        }
        
        # Try MediaPipe if Haar Cascade fails
        if len(faces) == 0:
            try:
                import mediapipe as mp
                mp_face_detection = mp.solutions.face_detection
                
                with mp_face_detection.FaceDetection(min_detection_confidence=0.3) as face_detection:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(rgb_frame)
                    
                    if results.detections:
                        detection = results.detections[0]
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        fw = int(bbox.width * w)
                        fh = int(bbox.height * h)
                        faces = np.array([[x, y, fw, fh]])
            except ImportError:
                pass
        
        # No face detected
        if len(faces) == 0:
            avg_brightness = np.mean(gray)
            tips = []
            if avg_brightness < 60:
                tips.append("• Too dark - turn on lights")
            elif avg_brightness > 200:
                tips.append("• Too bright - reduce lighting")
            tips.extend([
                "• Ensure face is clearly visible",
                "• Remove sunglasses/hat",
                "• Move closer to camera",
                "• Click 'Try Anyway' to skip check"
            ])
            return False, "❌ No face detected. Try:\n" + "\n".join(tips), metrics
        
        # Multiple faces
        if len(faces) > 1:
            return False, "⚠️ Multiple faces detected. Ensure only your face is visible.", metrics
        
        # Analyze single face
        x, y, fw, fh = faces[0]
        face_area = fw * fh
        image_area = w * h
        face_size_ratio = face_area / image_area
        
        face_roi = gray[y:y+fh, x:x+fw]
        brightness = np.mean(face_roi)
        blur_score = cv2.Laplacian(face_roi, cv2.CV_64F).var()
        
        center_x_face = x + fw // 2
        center_y_face = y + fh // 2
        x_offset = abs(center_x_face - w // 2) / w
        y_offset = abs(center_y_face - h // 2) / h
        centered = x_offset < 0.2 and y_offset < 0.2
        
        metrics = {
            'face_count': 1,
            'face_size': face_size_ratio,
            'brightness': brightness,
            'clarity': blur_score,
            'centered': centered
        }
        
        # Quality checks
        issues = []
        if face_size_ratio < 0.05:
            issues.append("Face too small - move closer")
        elif face_size_ratio > 0.7:
            issues.append("Face too large - move back")
        if brightness < 50:
            issues.append("Too dark - improve lighting")
        elif brightness > 210:
            issues.append("Too bright - reduce lighting")
        if blur_score < 80:
            issues.append("Image blurry - hold steady")
        if not centered:
            issues.append("Face not centered")
        
        if issues:
            return False, "⚠️ Face detected but needs adjustment:\n" + "\n".join(f"• {issue}" for issue in issues), metrics
        
        return True, "✅ Face detected! Quality is good.", metrics
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}", {}


def show_face_authentication(cfg: AppConfig) -> Optional[Dict]:
    """
    Face authentication with cropping, retake, and proper redirect
    """
    st.title("🔐 Face Authentication")
    st.markdown("---")
    
    # Check if already authenticated
    if 'user' in st.session_state and st.session_state.get('user'):
        st.success(f"✅ Already logged in as {st.session_state['user'].get('full_name', 'User')}")
        return st.session_state['user']
    
    # Instructions
    st.info("💡 **Tip:** Ensure good lighting and position your face clearly")
    
    with st.expander("📋 How to capture a good face photo", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("✅ **Position**")
            st.write("• Face centered")
            st.write("• Look at camera")
        with col2:
            st.write("✅ **Lighting**")
            st.write("• Good lighting")
            st.write("• No shadows")
        with col3:
            st.write("✅ **Quality**")
            st.write("• Hold steady")
            st.write("• Clear image")
    
    st.markdown("---")
    
    # Camera capture
    st.subheader("📸 Step 1: Capture Your Face")
    captured_image = st.camera_input(
        "Click to capture your face",
        key="face_auth_camera",
        help="Position your face in the center"
    )
    
    if captured_image is not None:
        # Store captured image
        if 'login_captured_face' not in st.session_state:
            st.session_state['login_captured_face'] = captured_image.getvalue()
        
        image_bytes = st.session_state['login_captured_face']
        
        # Show cropping tool
        st.markdown("---")
        st.subheader("✂️ Step 2: Crop Your Face (Optional)")
        st.info("ℹ️ Adjust the box to crop your face for better recognition")
        
        try:
            from streamlit_cropper import st_cropper
            
            img = Image.open(io.BytesIO(image_bytes))
            
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
            
            # Convert cropped image to bytes
            img_byte_arr = io.BytesIO()
            cropped_img.save(img_byte_arr, format='JPEG')
            final_image_bytes = img_byte_arr.getvalue()
            
        except ImportError:
            st.warning("⚠️ Cropping tool not available. Using full image.")
            final_image_bytes = image_bytes
        except Exception as e:
            st.warning(f"⚠️ Cropping error: {e}. Using full image.")
            final_image_bytes = image_bytes
        
        # Analyze face quality
        st.markdown("---")
        st.subheader("🔍 Step 3: Face Quality Analysis")
        
        with st.spinner("Analyzing face quality..."):
            has_face, message, metrics = analyze_face_quality(final_image_bytes)
        
        # Show results
        if has_face:
            st.success(message)
        else:
            st.warning(message)
        
        # Show metrics
        if metrics.get('face_count', 0) > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                face_size = metrics.get('face_size', 0)
                size_ok = 0.05 < face_size < 0.7
                st.metric("Face Size", f"{face_size:.1%}", delta="✅" if size_ok else "⚠️")
            with col2:
                brightness = metrics.get('brightness', 0)
                bright_ok = 50 < brightness < 210
                st.metric("Brightness", f"{brightness:.0f}", delta="✅" if bright_ok else "⚠️")
            with col3:
                clarity = metrics.get('clarity', 0)
                clear_ok = clarity > 80
                st.metric("Clarity", f"{clarity:.0f}", delta="✅" if clear_ok else "⚠️")
            with col4:
                centered = metrics.get('centered', False)
                st.metric("Position", "✅ Centered" if centered else "⚠️ Off")
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retake Photo", use_container_width=True, key="retake_main"):
                # Clear stored images
                if 'login_captured_face' in st.session_state:
                    del st.session_state['login_captured_face']
                if 'login_cropped_face' in st.session_state:
                    del st.session_state['login_cropped_face']
                st.rerun()
        
        with col2:
            if not has_face:
                if st.button("⚠️ Try Anyway", use_container_width=True, key="bypass_quality"):
                    has_face = True
                    st.info("⚠️ Quality check bypassed")
        
        with col3:
            if has_face:
                if st.button("✅ Proceed", use_container_width=True, type="primary", key="proceed_login"):
                    # Authenticate
                    st.markdown("---")
                    st.subheader("🔐 Step 4: Authenticating...")
                    
                    with st.spinner("Searching for matching face..."):
                        face_auth = FaceAuthenticator(cfg)
                        student, confidence, match_message = face_auth.find_matching_student(final_image_bytes)
                    
                    threshold = getattr(cfg, "confidence_threshold", 0.75)
                    
                    # Show results even if confidence is low
                    if student:
                        # Face match found (even if confidence is low)
                        if confidence >= threshold:
                            st.success(f"✅ {match_message}")
                        else:
                            st.warning(f"⚠️ {match_message}")
                            st.info("💡 Confidence is lower than ideal, but you can still proceed")
                        
                        # Show student info
                        st.markdown("---")
                        st.subheader("👤 Student Information")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**📛 Name:** {student.get('name', 'N/A')}")
                            st.write(f"**🔢 Student ID:** {student.get('id', 'N/A')}")
                            st.write(f"**🎓 Roll Number:** {student.get('roll_no', 'N/A')}")
                        with col2:
                            st.write(f"**🏢 Department:** {student.get('department', 'N/A')}")
                            st.write(f"**📚 Class:** {student.get('class', 'N/A')}")
                            st.write(f"**Confidence:** {confidence:.1%}")
                        
                        # Show confidence bar
                        st.progress(confidence)
                        if confidence >= threshold:
                            st.success(f"✅ Confidence {confidence:.1%} meets threshold {threshold:.1%}")
                        else:
                            st.warning(f"⚠️ Confidence {confidence:.1%} below threshold {threshold:.1%}, but you can still login")
                        
                        st.markdown("---")
                        
                        # Allow login even with low confidence (user can decide)
                        if st.button("🚀 Complete Login", use_container_width=True, type="primary", key="complete_login"):
                            # Log event
                            insert_event({
                                "student_id": student.get('id'),
                                "zone": "Face Authentication",
                                "status": "PASS" if confidence >= threshold else "WARNING",
                                "score": confidence,
                                "label": "Face Authentication",
                                "details": f"Face authentication. Confidence: {confidence:.1%}, Threshold: {threshold:.1%}"
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
                                'auth_time': datetime.now().isoformat(),
                                'confidence_score': confidence,
                                'gender': student.get('gender', 'U')
                            }
                            
                            # Set user in session
                            st.session_state['user'] = user_data
                            
                            # Clear face auth data
                            for key in ['login_captured_face', 'login_cropped_face']:
                                if key in st.session_state:
                                    del st.session_state[key]
                            
                            # Set redirect flag
                            st.session_state['redirect_to_verification'] = True
                            
                            st.success(f"🎉 Welcome, {student.get('name')}!")
                            st.balloons()
                            st.info("🔄 Redirecting to verification page...")
                            
                            # Force rerun to trigger redirect
                            st.rerun()
                    else:
                        st.error("❌ No matching face found in database")
                        st.info("💡 **Possible reasons:**")
                        st.info("• You haven't registered yet - Please register first")
                        st.info("• Your face wasn't captured during registration")
                        st.info("• Try with better lighting and clearer photo")
                        st.info("• Use Emergency Login (username + password) as alternative")
            else:
                st.button("❌ Quality Failed", use_container_width=True, disabled=True)
    
    return None


def show_face_registration_stage(cfg: AppConfig, student_id: str, auto_class: str) -> Tuple[bool, str, str]:
    """
    Face registration with cropping and quality check
    Returns: (success, face_hash, face_image_path)
    """
    st.markdown("---")
    st.markdown("### 👤 Stage 3: Face Registration")
    st.info("📸 Capture a clear photo of your face for biometric verification")
    
    # Instructions
    with st.expander("📋 Tips for good face capture", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("✅ **Do:**")
            st.write("• Face centered and clear")
            st.write("• Good lighting")
            st.write("• Look directly at camera")
        with col2:
            st.write("❌ **Don't:**")
            st.write("• Wear sunglasses")
            st.write("• Cover your face")
            st.write("• Use in dark room")
    
    # Camera capture
    captured_face = st.camera_input("📷 Capture your face", key=f"face_register_{student_id}")
    
    if captured_face is not None:
        image_bytes = captured_face.getvalue()
        
        # Show cropping tool
        st.markdown("---")
        st.subheader("✂️ Crop Your Face (Optional)")
        
        try:
            from streamlit_cropper import st_cropper
            
            img = Image.open(io.BytesIO(image_bytes))
            cropped_img = st_cropper(
                img,
                realtime_update=True,
                box_color='#0066FF',
                aspect_ratio=None,
                return_type='image'
            )
            
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            cropped_img.save(img_byte_arr, format='JPEG')
            final_image_bytes = img_byte_arr.getvalue()
        except:
            final_image_bytes = image_bytes
        
        # Analyze quality
        st.markdown("---")
        with st.spinner("🔍 Analyzing face quality..."):
            has_face, message, metrics = analyze_face_quality(final_image_bytes)
        
        if has_face:
            st.success(message)
        else:
            st.warning(message)
        
        # Show metrics
        if metrics.get('face_count', 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Face Size", f"{metrics.get('face_size', 0):.1%}")
            with col2:
                st.metric("Brightness", f"{metrics.get('brightness', 0):.0f}")
            with col3:
                st.metric("Clarity", f"{metrics.get('clarity', 0):.0f}")
        
        st.markdown("---")
        
        # Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Retake", use_container_width=True, key=f"retake_reg_{student_id}"):
                st.rerun()
        
        with col2:
            if has_face or st.button("⚠️ Try Anyway", key=f"bypass_reg_{student_id}"):
                if st.button("✅ Confirm & Register", use_container_width=True, type="primary", key=f"confirm_reg_{student_id}"):
                    with st.spinner("🔍 Processing face..."):
                        face_auth = FaceAuthenticator(cfg)
                        success, face_hash, face_image, msg = face_auth.capture_face_for_registration(final_image_bytes)
                        
                        if success:
                            st.success(msg)
                            st.write(f"**Face Hash:** `{face_hash[:16]}...`")
                            
                            if face_image is not None:
                                face_image_path = face_auth.save_face_image(face_image, student_id, student_id)
                                st.success("✅ Face image saved")
                                return success, face_hash, face_image_path if face_image_path else ""
                            else:
                                return success, face_hash, ""
                        else:
                            st.error(msg)
                            return False, "", ""
    
    return False, "", ""
