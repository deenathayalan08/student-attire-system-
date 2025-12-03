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
    st.markdown("## Attire Verification")
    
    tab1, tab2 = st.tabs(["Upload Image", "Take Photo"])
    
    with tab1:
        uploaded = st.file_uploader("Upload full-body image", type=["jpg", "jpeg", "png"])
        if uploaded:
            process_attire_verification(uploaded, student, cfg)
    
    with tab2:
        camera = st.camera_input("Capture full-body photo")
        if camera:
            process_attire_verification(camera, student, cfg)


def process_attire_verification(image_file, student: Dict[str, Any], cfg: AppConfig) -> None:
    """Process image for attire verification"""
    try:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Verify Attire", type="primary"):
            with st.spinner("Analyzing..."):
                bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                pose = extract_pose(bgr)
                features = extract_features_from_image(bgr, pose_landmarks=pose, bins=cfg.hist_bins)
                result = verify_attire_and_safety(features, cfg, st.session_state.get('classifier'))
                
                annotated = draw_pose_annotations(bgr.copy(), pose)
                violations = result.get("violations", {}).get("violations", [])
                annotated = draw_violation_indicators(annotated, pose, violations)
                annotated = overlay_detailed_badge(annotated, result)
                
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analysis Result")
                
                status = result.get("status", "UNKNOWN")
                if status == "PASS":
                    st.success("ATTIRE VERIFICATION PASSED")
                elif status == "WARNING":
                    st.warning("ATTIRE VERIFICATION WARNING")
                else:
                    st.error("ATTIRE VERIFICATION FAILED")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Success Score", f"{result.get('success_score', 0):.1%}")
                with col2:
                    st.metric("Violations", result.get("violations", {}).get("total_violations", 0))
                
                if violations:
                    st.subheader("Violations")
                    for v in violations:
                        st.error(f"{v.get('item')}: {v.get('detected')}")
                
                insert_event({
                    "student_id": student.get('id'),
                    "zone": "Student Dashboard",
                    "status": status,
                    "score": result.get('success_score', 0),
                    "label": "Self Verification",
                    "details": f"Violations: {len(violations)}"
                }, cfg=cfg)
    except Exception as e:
        st.error(f"Error: {str(e)}")
