import streamlit as st
from typing import Dict, Any
import pandas as pd
from PIL import Image
import cv2
import io

from ..config import AppConfig
from ..db import get_student, get_events_for_student
from ..features import extract_features_from_image, extract_pose
from ..verify import verify_attire_and_safety
from ..model import AttireClassifier
from ..utils.vis import draw_pose_annotations, draw_violation_indicators, overlay_detailed_badge


def show_student_dashboard(cfg: AppConfig):
    """Main student dashboard"""
    user = st.session_state['user']
    student_id = user.get('username')  # Assuming username is student ID for students

    st.title("👨‍🎓 Student Dashboard")
    st.markdown(f"Welcome, **{user['full_name']}**")

    # Get student info
    student_info = get_student(student_id, cfg=cfg)

    if student_info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Student ID", student_info['id'])
        with col2:
            st.metric("Class", student_info.get('class', 'N/A'))
        with col3:
            verified_status = "✅ Verified" if student_info.get('verified', 0) else "⏳ Pending"
            st.metric("Status", verified_status)

    # Dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📸 Quick Verify", "📊 My History", "👤 Profile", "❓ Help"])

    with tab1:
        show_quick_verification(cfg, student_id)

    with tab2:
        show_student_history(cfg, student_id)

    with tab3:
        show_student_profile(cfg, student_info, user)

    with tab4:
        show_help_section()


def show_quick_verification(cfg: AppConfig, student_id: str):
    """Quick verification tab for students"""
    st.header("Quick Attire Verification")

    st.markdown("""
    Take a photo to verify your attire compliance. Make sure:
    - You are in a well-lit area
    - Your full body is visible in the frame
    - Your student ID card is clearly visible
    - You are wearing the correct uniform
    """)

    # Zone selection
    zone = st.selectbox(
        "Select your current location",
        cfg.zones,
        index=0,
        help="Choose the zone where you are getting verified"
    )

    # Camera input
    st.markdown("### Take Photo")
    camera_input = st.camera_input("Capture your photo for verification")

    if camera_input is not None:
        # Process the image
        image = Image.open(camera_input).convert("RGB")
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        with st.spinner("Analyzing your attire..."):
            # Extract features
            pose = extract_pose(bgr_image)
            features = extract_features_from_image(bgr_image, pose_landmarks=pose, bins=cfg.hist_bins)

            # Load classifier if available
            classifier = None
            try:
                classifier = AttireClassifier()
                classifier.load(cfg.model_path, cfg)
            except:
                pass

            # Verify attire
            result = verify_attire_and_safety(features, cfg, classifier)

            # Create annotated image
            annotated = draw_pose_annotations(bgr_image.copy(), pose)

            # Add violation indicators
            violations = result.get("violations", {}).get("violations", [])
            annotated = draw_violation_indicators(annotated, pose, violations)

            # Add detailed badge
            annotated = overlay_detailed_badge(annotated, result)

        # Display results
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Your Photo")
            st.image(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        with col2:
            st.subheader("🔍 Analysis Result")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Show verification status
        status = result.get("status", "UNKNOWN")
        success_score = result.get("success_score", 0.0)

        if status == "PASS":
            st.success(f"✅ **VERIFICATION PASSED** - Compliance Score: {success_score:.1%}")
        elif status == "WARNING":
            st.warning(f"⚠️ **VERIFICATION WARNING** - Compliance Score: {success_score:.1%}")
        else:
            st.error(f"❌ **VERIFICATION FAILED** - Compliance Score: {success_score:.1%}")

        # Show violations if any
        violations = result.get("violations", {})
        if violations.get("total_violations", 0) > 0:
            st.error(f"🚨 {violations['total_violations']} Dress Code Violation(s) Detected")

            # Display each violation
            for i, violation in enumerate(violations.get("violations", []), 1):
                severity = violation.get("severity", "medium")
                severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")

                with st.expander(f"{severity_emoji} Violation {i}: {violation.get('item', 'Unknown')}", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Required:** {violation.get('required', 'N/A')}")
                        st.write(f"**Detected:** {violation.get('detected', 'N/A')}")
                        st.write(f"**Compliance Score:** {violation.get('score', 0.0):.1%}")
                    with col2:
                        st.write(f"**Severity:** {severity.upper()}")
                        st.write(f"**Reason:** {violation.get('reason', 'N/A')}")
        else:
            st.success("✅ No dress code violations detected!")

        # Log the event
        from ..db import insert_event
        event_id = insert_event({
            "student_id": student_id,
            "zone": zone,
            "status": status,
            "score": success_score,
            "label": result.get("label"),
            "details": str(result.get("details")),
        })

        st.info(f"✅ Verification logged (Event ID: #{event_id})")

        # ID Card status
        id_card_detected = features.get("id_card_detected", 0.0) > 0.5
        id_card_confidence = features.get("id_card_confidence", 0.0)

        st.markdown("---")
        st.subheader("🆔 ID Card Status")
        if id_card_detected:
            st.success(f"✅ ID Card Detected (Confidence: {id_card_confidence:.1%})")
        else:
            st.error(f"❌ ID Card Not Detected (Confidence: {id_card_confidence:.1%})")
            st.warning("Please ensure your student ID card is clearly visible for verification.")


def show_student_history(cfg: AppConfig, student_id: str):
    """Show student's verification history"""
    st.header("My Verification History")

    events = get_events_for_student(student_id, limit=20, cfg=cfg)

    if events:
        # Convert to DataFrame for better display
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')

        # Status badges
        def status_badge(status):
            if status == 'PASS':
                return '✅ PASS'
            elif status == 'FAIL':
                return '❌ FAIL'
            elif status == 'WARNING':
                return '⚠️ WARNING'
            else:
                return status

        df['Status'] = df['status'].apply(status_badge)

        # Display as table
        st.dataframe(
            df[['timestamp', 'zone', 'Status', 'score']].rename(columns={
                'timestamp': 'Date/Time',
                'zone': 'Location',
                'score': 'Score'
            }),
            use_container_width=True
        )

        # Statistics
        total_events = len(df)
        pass_count = (df['status'] == 'PASS').sum()
        pass_rate = pass_count / total_events if total_events > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Verifications", total_events)
        with col2:
            st.metric("Successful", pass_count)
        with col3:
            st.metric("Success Rate", f"{pass_rate:.1%}")

        # Recent trend (last 5 events)
        if len(df) >= 5:
            recent_scores = df['score'].head(5).tolist()
            st.subheader("Recent Performance Trend")
