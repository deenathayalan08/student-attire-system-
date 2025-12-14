"""
Real-time Face Detection and Capture UI
Provides live feedback during face capture for better user experience
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from typing import Optional, Tuple


class RealtimeFaceCapture:
    """Handle real-time face detection with visual feedback"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_face_with_feedback(self, image_bytes: bytes) -> Tuple[bool, str, Optional[np.ndarray], dict]:
        """
        Detect face and provide detailed feedback
        Returns: (success, message, annotated_image, metrics)
        """
        try:
            # Convert to numpy array
            pil_image = Image.open(io.BytesIO(image_bytes))
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,  # More sensitive
                minNeighbors=4,   # Less strict
                minSize=(80, 80)  # Smaller minimum size
            )
            
            h, w = frame.shape[:2]
            annotated = frame.copy()
            
            metrics = {
                'face_count': len(faces),
                'face_size': 0.0,
                'brightness': 0.0,
                'clarity': 0.0,
                'position': 'unknown'
            }
            
            # No face detected
            if len(faces) == 0:
                # Draw helpful guide
                center_x, center_y = w // 2, h // 2
                guide_size = min(w, h) // 3
                
                # Draw face guide oval
                cv2.ellipse(
                    annotated,
                    (center_x, center_y),
                    (guide_size, int(guide_size * 1.3)),
                    0, 0, 360,
                    (0, 255, 255),  # Yellow
                    3
                )
                
                # Add text
                cv2.putText(
                    annotated,
                    "Position your face here",
                    (center_x - 150, center_y - guide_size - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                
                return False, "❌ No face detected. Position your face in the yellow guide.", annotated, metrics
            
            # Multiple faces
            if len(faces) > 1:
                for (x, y, fw, fh) in faces:
                    cv2.rectangle(annotated, (x, y), (x+fw, y+fh), (0, 0, 255), 3)
                
                return False, "⚠️ Multiple faces detected. Ensure only your face is visible.", annotated, metrics
            
            # Single face detected - analyze quality
            x, y, fw, fh = faces[0]
            
            # Draw green rectangle around face
            cv2.rectangle(annotated, (x, y), (x+fw, y+fh), (0, 255, 0), 3)
            
            # Calculate metrics
            face_area = fw * fh
            image_area = w * h
            face_size_ratio = face_area / image_area
            
            face_roi = frame[y:y+fh, x:x+fw]
            gray_roi = gray[y:y+fh, x:x+fw]
            
            brightness = np.mean(gray_roi)
            blur_score = cv2.Laplacian(gray_roi, cv2.CV_64F).var()
            
            # Position check
            center_x_face = x + fw // 2
            center_y_face = y + fh // 2
            center_x_img = w // 2
            center_y_img = h // 2
            
            x_offset = abs(center_x_face - center_x_img) / w
            y_offset = abs(center_y_face - center_y_img) / h
            
            metrics = {
                'face_count': 1,
                'face_size': face_size_ratio,
                'brightness': brightness,
                'clarity': blur_score,
                'x_offset': x_offset,
                'y_offset': y_offset
            }
            
            # Quality checks
            issues = []
            
            # Size check
            if face_size_ratio < 0.08:
                issues.append("Face too small - move closer")
                cv2.putText(annotated, "Move closer", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            elif face_size_ratio > 0.6:
                issues.append("Face too large - move back")
                cv2.putText(annotated, "Move back", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # Brightness check
            if brightness < 60:
                issues.append("Too dark - improve lighting")
                cv2.putText(annotated, "Increase lighting", (x, y+fh+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif brightness > 200:
                issues.append("Too bright - reduce lighting")
                cv2.putText(annotated, "Reduce lighting", (x, y+fh+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Clarity check
            if blur_score < 100:
                issues.append("Image blurry - hold steady")
                cv2.putText(annotated, "Hold steady", (x, y+fh+50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Position check
            if x_offset > 0.15 or y_offset > 0.15:
                issues.append("Face not centered")
                cv2.putText(annotated, "Center your face", (x, y+fh+75), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Add checkmark if all good
            if not issues:
                cv2.putText(annotated, "Perfect! Ready to capture", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                return True, "✅ Face detected! Quality is good. Ready to capture.", annotated, metrics
            else:
                message = "⚠️ Face detected but needs adjustment:\n" + "\n".join(f"• {issue}" for issue in issues)
                return False, message, annotated, metrics
                
        except Exception as e:
            return False, f"❌ Error processing image: {str(e)}", None, {}
    
    def draw_quality_indicators(self, image: np.ndarray, metrics: dict) -> np.ndarray:
        """Draw quality indicators on image"""
        h, w = image.shape[:2]
        
        # Create overlay
        overlay = image.copy()
        
        # Quality bar background
        bar_height = 80
        cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
        
        # Blend overlay
        alpha = 0.6
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Draw quality metrics
        y_pos = 25
        x_start = 20
        
        # Face size indicator
        face_size = metrics.get('face_size', 0)
        size_color = (0, 255, 0) if 0.08 < face_size < 0.6 else (0, 165, 255)
        cv2.putText(image, f"Size: {face_size:.1%}", (x_start, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, size_color, 2)
        
        # Brightness indicator
        brightness = metrics.get('brightness', 0)
        bright_color = (0, 255, 0) if 60 < brightness < 200 else (0, 165, 255)
        cv2.putText(image, f"Light: {brightness:.0f}", (x_start + 150, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, bright_color, 2)
        
        # Clarity indicator
        clarity = metrics.get('clarity', 0)
        clarity_color = (0, 255, 0) if clarity > 100 else (0, 165, 255)
        cv2.putText(image, f"Clarity: {clarity:.0f}", (x_start + 300, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, clarity_color, 2)
        
        # Overall status
        y_pos = 55
        if metrics.get('face_count', 0) == 1 and 0.08 < face_size < 0.6 and 60 < brightness < 200 and clarity > 100:
            cv2.putText(image, "Status: READY TO CAPTURE", (x_start, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Status: Adjust position/lighting", (x_start, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
        return image


def show_realtime_face_capture_ui(title: str = "📸 Face Capture", key_prefix: str = "face_capture") -> Optional[bytes]:
    """
    Show real-time face capture UI with live feedback
    Returns: captured image bytes if successful, None otherwise
    """
    st.subheader(title)
    
    # Instructions
    with st.expander("📋 How to capture a good face photo", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("✅ **Position**")
            st.write("• Face centered")
            st.write("• Look at camera")
            st.write("• Remove glasses")
        with col2:
            st.write("✅ **Lighting**")
            st.write("• Good lighting")
            st.write("• No shadows")
            st.write("• Avoid backlight")
        with col3:
            st.write("✅ **Quality**")
            st.write("• Hold steady")
            st.write("• Clear image")
            st.write("• No obstructions")
    
    st.markdown("---")
    
    # Camera input with real-time feedback
    st.info("📷 **Step 1:** Click the camera button below to capture your face")
    
    captured_image = st.camera_input(
        "Capture your face",
        key=f"{key_prefix}_camera",
        help="Position your face in the center and ensure good lighting"
    )
    
    if captured_image is not None:
        st.success("✅ Image captured! Analyzing face quality...")
        
        # Analyze captured image
        face_detector = RealtimeFaceCapture()
        image_bytes = captured_image.getvalue()
        
        success, message, annotated_image, metrics = face_detector.detect_face_with_feedback(image_bytes)
        
        # Show analysis results
        st.markdown("---")
        st.subheader("🔍 Face Analysis Results")
        
        # Display annotated image
        if annotated_image is not None:
            # Add quality indicators
            annotated_with_indicators = face_detector.draw_quality_indicators(annotated_image, metrics)
            
            # Convert BGR to RGB for display
            annotated_rgb = cv2.cvtColor(annotated_with_indicators, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, caption="Face Detection Analysis", use_container_width=True)
        
        # Show message
        if success:
            st.success(message)
        else:
            st.warning(message)
        
        # Show detailed metrics
        if metrics.get('face_count', 0) > 0:
            st.markdown("#### 📊 Quality Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                face_size = metrics.get('face_size', 0)
                size_status = "✅ Good" if 0.08 < face_size < 0.6 else "⚠️ Adjust"
                st.metric("Face Size", f"{face_size:.1%}", delta=size_status)
            
            with col2:
                brightness = metrics.get('brightness', 0)
                bright_status = "✅ Good" if 60 < brightness < 200 else "⚠️ Adjust"
                st.metric("Brightness", f"{brightness:.0f}", delta=bright_status)
            
            with col3:
                clarity = metrics.get('clarity', 0)
                clarity_status = "✅ Clear" if clarity > 100 else "⚠️ Blurry"
                st.metric("Clarity", f"{clarity:.0f}", delta=clarity_status)
            
            with col4:
                x_offset = metrics.get('x_offset', 0)
                y_offset = metrics.get('y_offset', 0)
                position_status = "✅ Centered" if x_offset < 0.15 and y_offset < 0.15 else "⚠️ Off-center"
                st.metric("Position", position_status)
        
        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Retake Photo", use_container_width=True, key=f"{key_prefix}_retake"):
                st.rerun()
        
        with col2:
            if success:
                if st.button("✅ Use This Photo", use_container_width=True, type="primary", key=f"{key_prefix}_confirm"):
                    return image_bytes
            else:
                st.button("❌ Quality Check Failed", use_container_width=True, disabled=True, key=f"{key_prefix}_disabled")
                st.info("💡 Please retake the photo following the suggestions above")
    
    return None
