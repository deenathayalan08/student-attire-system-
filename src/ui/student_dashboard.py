"""Student Dashboard - Post-login attire verification interface"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, Any

from ..config import AppConfig
from ..features import extract_features_from_image, extract_pose
from ..verify import verify_attire_and_safety
from ..db import insert_event, get_student
from ..utils.vis import draw_pose_annotations, draw_violation_indicators, overlay_detailed_badge
from ..rbac import require_login, check_data_access_permission, has_permission, Permission


@require_login
def show_student_dashboard(cfg: AppConfig) -> None:
    """Display student dashboard with attire verification - RBAC Protected"""
    user = st.session_state.get('user')
    
    # Import RBAC functions
    from ..rbac import check_data_access_permission, has_permission, Permission
    
    if not user:
        st.error("❌ Please login first to access your dashboard")
        st.info("💡 You need to be logged in to view your student dashboard")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔐 Go to Login", use_container_width=True, type="primary"):
                st.session_state.current_page = 'face_auth'
                st.rerun()
        with col2:
            if st.button("🏠 Go to Home", use_container_width=True):
                st.session_state.current_page = 'home'
                st.rerun()
        return
    
    # Try multiple ways to get student ID
    student_id = (
        user.get('student_id') or 
        user.get('id') or 
        user.get('username') or 
        user.get('roll_no')
    )
    
    # Debug: Show what we're working with
    st.info(f"🔍 **Debug:** Looking for student with ID: `{student_id}`")
    st.info(f"📋 **User data keys:** {list(user.keys())}")
    
    if not student_id:
        st.error("❌ Unable to identify student. Please login again.")
        st.error("🔍 **Debug:** No student ID found in user data")
        
        # Show all user data for debugging
        with st.expander("🔧 User Data Debug", expanded=True):
            st.json(user)
        
        if st.button("🔐 Go to Login"):
            st.session_state.current_page = 'face_auth'
            st.rerun()
        return
    
    # RBAC: Check if user can access this student's data
    if not check_data_access_permission(student_id):
        st.error("🚫 **Access Denied**")
        st.warning("You don't have permission to access this student's data.")
        st.info("Students can only view their own dashboard.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🏠 Go to Home", use_container_width=True, type="primary"):
                st.session_state.current_page = 'home'
                st.rerun()
        with col2:
            if st.button("🔐 Login Again", use_container_width=True):
                st.session_state.current_page = 'face_auth'
                st.rerun()
        return
    
    student = get_student(student_id, cfg=cfg)
    if not student:
        st.error(f"❌ Student record not found for ID: {student_id}")
        
        # Try alternative lookups
        from ..db import get_student_by_roll_no
        if user.get('roll_no'):
            student = get_student_by_roll_no(user.get('roll_no'), cfg=cfg)
            if student:
                st.success("✅ Found student record using roll number!")
            else:
                st.warning("⚠️ Also tried roll number lookup - no record found")
        
        if not student:
            st.info("💡 **Possible solutions:**")
            st.info("• Make sure you completed registration properly")
            st.info("• Contact admin if your record is missing")
            st.info("• Try logging in again")
            
            # Show debug info to help troubleshoot
            with st.expander("🔧 Debug Information", expanded=True):
                st.write(f"**Searched for Student ID:** `{student_id}`")
                st.write("**Available user data fields:**")
                for key, value in user.items():
                    if key != 'password':
                        st.write(f"- {key}: `{value}`")
                
                # Test database connection
                st.write("**Database Test:**")
                try:
                    from ..db import get_all_students
                    all_students = get_all_students(cfg=cfg)
                    st.write(f"- Total students in database: {len(all_students)}")
                    if all_students:
                        st.write("- Sample student IDs in database:")
                        for i, s in enumerate(all_students[:5]):  # Show first 5
                            st.write(f"  • {s.get('id', 'No ID')} - {s.get('name', 'No Name')}")
                        if len(all_students) > 5:
                            st.write(f"  ... and {len(all_students) - 5} more")
                except Exception as e:
                    st.write(f"- Database error: {e}")
                
                # Check if student exists with different ID format
                st.write("**Alternative ID Search:**")
                possible_ids = [
                    user.get('student_id'),
                    user.get('id'), 
                    user.get('username'),
                    user.get('roll_no'),
                    str(user.get('student_id', '')).strip(),
                    str(user.get('username', '')).strip()
                ]
                for pid in possible_ids:
                    if pid and pid != student_id:
                        try:
                            test_student = get_student(str(pid), cfg=cfg)
                            if test_student:
                                st.success(f"✅ Found student with ID: `{pid}`")
                                break
                            else:
                                st.write(f"- Tried ID `{pid}`: Not found")
                        except Exception as e:
                            st.write(f"- Error testing ID `{pid}`: {e}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Try Login Again", use_container_width=True):
                    st.session_state.current_page = 'face_auth'
                    st.rerun()
            with col2:
                if st.button("📝 Register Again", use_container_width=True):
                    st.session_state.current_page = 'register'
                    st.rerun()
            return
    
    # Modern dashboard header
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px;">
        <div style="display: flex; align-items: center; color: white;">
            <div style="flex: 1;">
                <h1 style="margin: 0; font-size: 2rem; font-weight: 600;">📋 Student Dashboard</h1>
                <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.1rem;">Welcome back, {student.get('name', 'Student')[:30]}!</p>
            </div>
            <div style="text-align: right;">
                <div style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                    🔐 {user.get('auth_method', 'Unknown').title()} Auth
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug info (collapsible for development)
    with st.expander("🔧 System Debug (Development Mode)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**User Session Data:**")
            st.json({k: v for k, v in user.items() if k != 'password'})
        with col2:
            st.write("**Student Record Data:**")
            st.json({k: v for k, v in student.items() if k != 'password'})
        st.info(f"**Lookup ID:** `{student_id}`")
    
    # Student information cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #007bff;">
            <h4 style="color: #007bff; margin-bottom: 1rem; font-size: 1.1rem;">👤 Personal Info</h4>
            <div style="color: #495057; line-height: 1.8;">
                <div><strong>ID:</strong> {student.get('id', 'N/A')}</div>
                <div><strong>Roll No:</strong> {student.get('roll_no', 'N/A')}</div>
                <div><strong>Gender:</strong> {student.get('gender', 'N/A')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #28a745;">
            <h4 style="color: #28a745; margin-bottom: 1rem; font-size: 1.1rem;">🎓 Academic Info</h4>
            <div style="color: #495057; line-height: 1.8;">
                <div><strong>Class:</strong> {student.get('class', 'Not assigned')}</div>
                <div><strong>Department:</strong> {student.get('department', 'Not assigned')[:25]}{'...' if len(student.get('department', '')) > 25 else ''}</div>
                <div><strong>Status:</strong> {'✅ Verified' if student.get('verified') else '⚠️ Pending'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 4px solid #ffc107;">
            <h4 style="color: #ffc107; margin-bottom: 1rem; font-size: 1.1rem;">📞 Contact Info</h4>
            <div style="color: #495057; line-height: 1.8;">
                <div><strong>Email:</strong> {student.get('email', 'Not provided')[:20]}{'...' if len(student.get('email', '')) > 20 else ''}</div>
                <div><strong>Phone:</strong> {student.get('phone', 'Not provided')}</div>
                <div><strong>Login:</strong> {user.get('auth_method', 'Unknown').title()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Action Dashboard
    st.markdown("""
    <div style="margin: 1.5rem 0 1rem 0;">
        <h3 style="color: #2c3e50; font-weight: 600;">⚡ Quick Actions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-weight: 600; font-size: 0.9rem;">Accuracy Report</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 View Report", use_container_width=True, key="view_accuracy_btn"):
            st.session_state['show_accuracy'] = True
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📋</div>
            <div style="font-weight: 600; font-size: 0.9rem;">History</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📋 View History", use_container_width=True, key="view_report_btn"):
            st.session_state['show_report'] = True
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎓</div>
            <div style="font-weight: 600; font-size: 0.9rem;">Verify Now</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎓 Quick Verify", use_container_width=True, key="quick_verify_btn"):
            # Scroll to verification section
            st.session_state['scroll_to_verify'] = True
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">👤</div>
            <div style="font-weight: 600; font-size: 0.9rem;">Profile</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👤 My Profile", use_container_width=True, key="profile_quick_btn"):
            st.session_state.current_page = 'profile'
            st.rerun()
    
    # Show Accuracy Report if requested
    if st.session_state.get('show_accuracy', False):
        show_accuracy_report(student, cfg)
        if st.button("❌ Close Accuracy Report", key="close_accuracy"):
            st.session_state['show_accuracy'] = False
            st.rerun()
        st.markdown("---")
    
    # Show Verification History if requested
    if st.session_state.get('show_report', False):
        show_verification_report(student, cfg)
        if st.button("❌ Close Report", key="close_report"):
            st.session_state['show_report'] = False
            st.rerun()
        st.markdown("---")
    
    # Verification section with modern design
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; text-align: center; color: white; margin: 2rem 0;">
        <h2 style="margin-bottom: 1rem; font-weight: 600;">🎓 AI-Powered Attire Verification</h2>
        <p style="margin: 0; opacity: 0.9; font-size: 1.1rem;">Upload an image, take a photo, or record a video for instant compliance analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced verification options
    st.markdown("""
    <div style="margin: 1rem 0;">
        <h4 style="color: #2c3e50; text-align: center;">Choose Your Verification Method</h4>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs([
        "📤 Upload Image", 
        "📷 Live Camera", 
        "🎥 Video Analysis"
    ])
    
    with tab1:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
            <h4 style="color: #007bff; margin-bottom: 1rem;">📤 Upload Image Verification</h4>
            <p style="color: #6c757d; margin-bottom: 1rem;">Upload a clear, full-body image for AI analysis</p>
            <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #007bff;">
                <strong>📋 Best Practices:</strong><br>
                • Full-body shot preferred<br>
                • Good lighting conditions<br>
                • Plain background<br>
                • Uniform clearly visible
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png"], 
            key="dashboard_upload",
            help="Upload JPG, JPEG, or PNG files up to 200MB"
        )
        if uploaded:
            process_attire_verification(uploaded, student, cfg)
    
    with tab2:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
            <h4 style="color: #28a745; margin-bottom: 1rem;">📷 Live Camera Verification</h4>
            <p style="color: #6c757d; margin-bottom: 1rem;">Use your device camera for real-time capture</p>
            <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;">
                <strong>📋 Camera Tips:</strong><br>
                • Stand 3-4 feet from camera<br>
                • Ensure good lighting<br>
                • Keep device steady<br>
                • Full body in frame
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        camera = st.camera_input(
            "📸 Capture your photo", 
            key="dashboard_camera",
            help="Position yourself with full body visible and good lighting"
        )
        if camera:
            process_attire_verification(camera, student, cfg)
    
    with tab3:
        st.markdown("""
        <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 15px; margin-bottom: 1rem;">
            <h4 style="color: #ffc107; margin-bottom: 1rem;">🎥 Video Analysis</h4>
            <p style="color: #6c757d; margin-bottom: 1rem;">Upload a short video for comprehensive frame-by-frame analysis</p>
            <div style="background: white; padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107;">
                <strong>📋 Video Guidelines:</strong><br>
                • 5-30 seconds duration<br>
                • Show full attire clearly<br>
                • Steady recording<br>
                • Multiple angles helpful
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        video = st.file_uploader(
            "Choose a video file", 
            type=["mp4", "mov", "avi", "mkv"], 
            key="dashboard_video",
            help="Upload MP4, MOV, AVI, or MKV files up to 200MB"
        )
        if video:
            process_video_verification(video, student, cfg)


def process_attire_verification(image_file, student: Dict[str, Any], cfg: AppConfig) -> None:
    """Process image for attire verification"""
    try:
        image = Image.open(image_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Your Image", use_column_width=True)
        
        if st.button("🔍 Verify Attire", type="primary", use_container_width=True):
            with col2:
                with st.spinner("🔄 Analyzing your attire..."):
                    bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    pose = extract_pose(bgr)
                    features = extract_features_from_image(bgr, pose_landmarks=pose, bins=cfg.hist_bins)
                    result = verify_attire_and_safety(features, cfg, st.session_state.get('classifier'))
                    
                    annotated = draw_pose_annotations(bgr.copy(), pose)
                    violations = result.get("violations", {}).get("violations", [])
                    annotated = draw_violation_indicators(annotated, pose, violations)
                    annotated = overlay_detailed_badge(annotated, result)
                    
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analysis Result", use_column_width=True)
            
            st.markdown("---")
            
            # Display results
            status = result.get("status", "UNKNOWN")
            if status == "PASS":
                st.success("✅ ATTIRE VERIFICATION PASSED")
            elif status == "WARNING":
                st.warning("⚠️ ATTIRE VERIFICATION WARNING")
            else:
                st.error("❌ ATTIRE VERIFICATION FAILED")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Status", status)
            with col2:
                st.metric("Success Score", f"{result.get('success_score', 0):.1%}")
            with col3:
                st.metric("Violations", result.get("violations", {}).get("total_violations", 0))
            
            # Show violations if any
            if violations:
                st.markdown("### 🚨 Violations Detected")
                for i, v in enumerate(violations, 1):
                    severity = v.get('severity', 'medium')
                    severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")
                    
                    with st.expander(f"{severity_emoji} Violation {i}: {v.get('item', 'Unknown')}", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Required:** {v.get('required', 'N/A')}")
                            st.write(f"**Detected:** {v.get('detected', 'N/A')}")
                        with col2:
                            st.write(f"**Severity:** {severity.upper()}")
                            st.write(f"**Reason:** {v.get('reason', 'N/A')}")
            else:
                st.success("✅ No violations detected! Your attire is compliant.")
            
            # Log event
            insert_event({
                "student_id": student.get('id'),
                "zone": "Student Dashboard",
                "status": status,
                "score": result.get('success_score', 0),
                "label": "Self Verification",
                "details": f"Violations: {len(violations)}"
            }, cfg=cfg)
            
            st.balloons() if status == "PASS" else None
            
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")


def process_video_verification(video_file, student: Dict[str, Any], cfg: AppConfig) -> None:
    """Process video for attire verification"""
    import tempfile
    import os
    
    try:
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            tmp_path = tmp_file.name
        
        st.info("📹 Processing video frames...")
        
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            st.error("Failed to open video")
            return
        
        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = max(1, min(cfg.max_video_fps, int(cap.get(cv2.CAP_PROP_FPS) or 15)))
        sample_every = max(1, int((cap.get(cv2.CAP_PROP_FPS) or 15) // fps))
        
        ok_frames = 0
        total_frames = 0
        idx = 0
        
        frame_placeholder = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_every != 0:
                idx += 1
                continue
            idx += 1
            
            pose = extract_pose(frame)
            features = extract_features_from_image(frame, pose_landmarks=pose, bins=cfg.hist_bins)
            result = verify_attire_and_safety(features, cfg, st.session_state.get('classifier'))
            
            annotated = draw_pose_annotations(frame.copy(), pose)
            violations = result.get("violations", {}).get("violations", [])
            annotated = draw_violation_indicators(annotated, pose, violations)
            annotated = overlay_detailed_badge(annotated, result)
            
            frame_placeholder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Frame {idx}", use_column_width=True)
            
            total_frames += 1
            ok_frames += 1 if result["status"] == "PASS" else 0
            
            if frame_count > 0:
                progress.progress(min(1.0, idx / frame_count))
        
        cap.release()
        os.unlink(tmp_path)
        
        # Show summary
        st.markdown("---")
        st.markdown("### 📊 Video Analysis Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", total_frames)
        with col2:
            st.metric("Passed Frames", ok_frames)
        with col3:
            pass_rate = (ok_frames / max(1, total_frames)) * 100
            st.metric("Pass Rate", f"{pass_rate:.1f}%")
        
        if pass_rate >= 80:
            st.success(f"✅ Video verification PASSED! ({pass_rate:.1f}% compliance)")
        elif pass_rate >= 50:
            st.warning(f"⚠️ Video verification WARNING ({pass_rate:.1f}% compliance)")
        else:
            st.error(f"❌ Video verification FAILED ({pass_rate:.1f}% compliance)")
        
        # Log event
        insert_event({
            "student_id": student.get('id'),
            "zone": "Student Dashboard",
            "status": "PASS" if pass_rate >= 80 else "FAIL",
            "score": pass_rate / 100,
            "label": "Video Verification",
            "details": f"Frames: {total_frames}, Pass Rate: {pass_rate:.1f}%"
        }, cfg=cfg)
        
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")



def show_accuracy_report(student: Dict[str, Any], cfg: AppConfig) -> None:
    """Display student's accuracy and performance metrics"""
    import pandas as pd
    from ..db import get_student_stats, get_events_for_student
    
    st.markdown("### 📊 Accuracy Report")
    
    student_id = student.get('id')
    stats = get_student_stats(student_id, cfg=cfg)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        avg_score = stats.get('avg_score', 0.0)
        st.metric("Average Score", f"{avg_score:.1%}" if avg_score else "N/A")
    with col2:
        pass_rate = stats.get('pass_rate', 0.0)
        st.metric("Pass Rate", f"{pass_rate:.1%}" if pass_rate else "N/A")
    with col3:
        total_events = stats.get('total_events', 0)
        st.metric("Total Verifications", total_events)
    with col4:
        last_event = stats.get('last_event', 'Never')
        st.metric("Last Verified", last_event if last_event != 'Never' else "N/A")
    
    st.markdown("---")
    
    # Get recent events for chart
    events = get_events_for_student(student_id, limit=50, cfg=cfg)
    if events:
        df_events = pd.DataFrame(events)
        
        # Show accuracy trend chart
        if 'score' in df_events.columns and not df_events['score'].isnull().all():
            st.markdown("#### 📈 Accuracy Trend")
            chart_df = df_events[['timestamp', 'score']].copy()
            chart_df['timestamp'] = pd.to_datetime(chart_df['timestamp'])
            chart_df = chart_df.set_index('timestamp').sort_index()
            st.line_chart(chart_df['score'])
        
        # Show status distribution
        if 'status' in df_events.columns:
            st.markdown("#### 📊 Status Distribution")
            status_counts = df_events['status'].value_counts()
            st.bar_chart(status_counts)
    else:
        st.info("No verification history available yet. Complete your first verification to see accuracy metrics!")


def show_verification_report(student: Dict[str, Any], cfg: AppConfig) -> None:
    """Display student's verification history report"""
    import pandas as pd
    from ..db import get_events_for_student
    
    st.markdown("### 📋 Verification History Report")
    
    student_id = student.get('id')
    
    # Limit selector
    limit = st.selectbox("Show last", [10, 25, 50, 100], index=2, key="report_limit")
    
    events = get_events_for_student(student_id, limit=limit, cfg=cfg)
    
    if events:
        df_events = pd.DataFrame(events)
        
        # Format the dataframe for better display
        display_df = df_events.copy()
        if 'score' in display_df.columns:
            display_df['score'] = display_df['score'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = df_events.to_csv(index=False)
        st.download_button(
            label="📥 Download Report (CSV)",
            data=csv,
            file_name=f"verification_report_{student_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Summary statistics
        st.markdown("---")
        st.markdown("#### 📊 Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pass_count = len(df_events[df_events['status'] == 'PASS'])
            st.metric("Total PASS", pass_count)
        with col2:
            fail_count = len(df_events[df_events['status'] == 'FAIL'])
            st.metric("Total FAIL", fail_count)
        with col3:
            warning_count = len(df_events[df_events['status'] == 'WARNING'])
            st.metric("Total WARNING", warning_count)
    else:
        st.info("No verification history available yet. Complete your first verification to see your report!")
