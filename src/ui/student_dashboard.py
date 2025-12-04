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


def show_student_dashboard(cfg: AppConfig) -> None:
    """Display student dashboard with attire verification"""
    user = st.session_state.get('user')
    if not user:
        st.error("Please login first")
        return
    
    student_id = user.get('student_id') or user.get('id')
    student = get_student(student_id, cfg=cfg)
    if not student:
        st.error("Student record not found")
        return
    
    st.title("Student Dashboard")
    st.markdown(f"### Welcome, {student.get('name', 'Student')}!")
    
    # Student Info
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**ID:** {student.get('id')}")
        st.write(f"**Class:** {student.get('class')}")
        st.write(f"**Department:** {student.get('department')}")
    with col2:
        st.write(f"**Email:** {student.get('email')}")
        st.write(f"**Phone:** {student.get('phone')}")
    
    st.markdown("---")
    
    # Quick Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 View Accuracy Report", use_container_width=True, key="view_accuracy_btn"):
            st.session_state['show_accuracy'] = True
    with col2:
        if st.button("📋 View Verification History", use_container_width=True, key="view_report_btn"):
            st.session_state['show_report'] = True
    
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
    
    st.markdown("## 🎓 Attire Verification")
    st.info("📸 Verify your uniform compliance by uploading an image, taking a photo, or uploading a video.")
    
    tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Take Photo", "🎥 Upload Video"])
    
    with tab1:
        st.subheader("Upload Image")
        uploaded = st.file_uploader("Upload full-body image", type=["jpg", "jpeg", "png"], key="dashboard_upload")
        if uploaded:
            process_attire_verification(uploaded, student, cfg)
    
    with tab2:
        st.subheader("Take Photo")
        st.info("💡 Stand in good lighting with your full body visible")
        camera = st.camera_input("Capture full-body photo", key="dashboard_camera")
        if camera:
            process_attire_verification(camera, student, cfg)
    
    with tab3:
        st.subheader("Upload Video")
        st.info("💡 Upload a short video showing your full attire")
        video = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv"], key="dashboard_video")
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
