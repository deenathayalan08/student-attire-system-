import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is on sys.path for `src` imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from src.config import AppConfig
from src.dataset import append_sample_to_dataset, ensure_dirs, load_dataset
from src.features import extract_features_from_image, extract_pose
from src.model import AttireClassifier
from src.utils.vis import draw_pose_annotations, overlay_badge, draw_violation_indicators, overlay_detailed_badge
from src.verify import verify_attire_and_safety
from src.db import (
	init_db, insert_event, list_events, upsert_setting, get_setting, 
	get_all_students, get_compliance_stats, add_student, update_student_verification, add_user,
	add_department, get_all_departments, get_department_by_id, get_classes_by_department,
	get_students_by_department, get_department_statistics, update_department, delete_department,
	search_departments, export_department_report, update_class_advisor, update_class_room, delete_student
)
from src.alerts import notify_non_compliance, get_id_card_status_message, get_detailed_id_card_message, notify_id_card_status
from src.security import check_and_alert_unauthorized_student, check_and_log_entry_time, check_and_log_exit_time, check_and_alert_emergency_violations


def init_session_state() -> None:
	if "classifier" not in st.session_state:
		st.session_state.classifier = AttireClassifier()
	if "config" not in st.session_state:
		st.session_state.config = AppConfig()
	if "dataset_df" not in st.session_state:
		st.session_state.dataset_df = None
	if "last_train_info" not in st.session_state:
		st.session_state.last_train_info = None
	if "zone" not in st.session_state:
		st.session_state.zone = "Gate"
	# Navigation state - single source of truth
	if "current_page" not in st.session_state:
		st.session_state.current_page = "home"


def navigate_to(page: str) -> None:
	"""Centralized navigation function - single point of control"""
	st.session_state.current_page = page
	# Clean up temporary navigation flags
	for key in ['show_verification', 'force_show_verification', 'page', 
	            'login_in_progress', 'login_captured_face', 'login_cropped_face']:
		if key in st.session_state:
			del st.session_state[key]
	st.rerun()


def get_current_page() -> str:
	"""Get current page from session state"""
	# Support legacy 'page' key for backward compatibility
	if 'page' in st.session_state and 'current_page' not in st.session_state:
		return st.session_state['page']
	return st.session_state.get('current_page', 'home')


def is_logged_in() -> bool:
	"""Check if user is logged in"""
	return st.session_state.get('user') is not None


def is_admin() -> bool:
	"""Check if current user is admin"""
	user = st.session_state.get('user')
	return user and user.get('role') == 'admin'


def is_student() -> bool:
	"""Check if current user is student"""
	user = st.session_state.get('user')
	return user and user.get('role') == 'student'


def logout_and_redirect() -> None:
	"""Logout user and redirect to home"""
	from src.auth import logout_user
	logout_user(st.session_state)
	# Clear all navigation state
	for key in ['current_page', 'page', 'show_verification', 'force_show_verification',
	            'login_in_progress', 'login_captured_face', 'login_cropped_face']:
		if key in st.session_state:
			del st.session_state[key]
	st.session_state.current_page = 'home'
	st.rerun()


def ensure_config_defaults(cfg: AppConfig) -> AppConfig:
	# Backward compatibility if session has older AppConfig
	if not hasattr(cfg, "policy_profile"):
		cfg.policy_profile = "regular"
	if not hasattr(cfg, "zones") or cfg.zones is None:
		cfg.zones = ["Gate", "Classroom", "Lab", "Sports"]
	if not hasattr(cfg, "current_label"):
		cfg.current_label = "compliant"
	return cfg


def sidebar_settings() -> None:
	# Only show settings if user is admin
	user = st.session_state.get('user')
	if not user or user.get('role') != 'admin':
		return

	# Toggle button for settings
	if 'show_settings' not in st.session_state:
		st.session_state.show_settings = False

	if st.sidebar.button("⚙️ Settings", key="settings_toggle"):
		st.session_state.show_settings = not st.session_state.show_settings

	# Show settings content only when toggled
	if st.session_state.show_settings:
		st.sidebar.markdown("---")
		st.sidebar.header("Settings")
		cfg: AppConfig = ensure_config_defaults(st.session_state.config)

		cfg.policy_profile = st.sidebar.selectbox("Policy profile", ["regular", "sports", "lab"], index=["regular", "sports", "lab"].index(cfg.policy_profile))
		st.sidebar.caption("Profiles adjust expected attire and safety items.")

		cfg.expected_top = st.sidebar.text_input("Expected top color keyword", value=cfg.expected_top)
		cfg.expected_bottom = st.sidebar.text_input("Expected bottom color keyword", value=cfg.expected_bottom)
		cfg.hist_bins = st.sidebar.slider("Histogram bins", 8, 64, cfg.hist_bins, 4)
		cfg.confidence_threshold = st.sidebar.slider("Decision threshold", 0.5, 0.95, float(cfg.confidence_threshold), 0.01)
		cfg.max_video_fps = st.sidebar.slider("Max video FPS", 5, 30, cfg.max_video_fps, 1)
		cfg.enable_rules = st.sidebar.checkbox("Enable rule-based checks", value=cfg.enable_rules)
		cfg.enable_model = st.sidebar.checkbox("Enable ML model", value=cfg.enable_model)
		cfg.save_frames = st.sidebar.checkbox("Save collected frames to dataset", value=False)
		cfg.current_label = st.sidebar.text_input("Label for saved samples", value=cfg.current_label)

		# ID Card detection settings
		st.sidebar.markdown("---")
		st.sidebar.subheader("ID Card Detection")
		cfg.enable_id_card_detection = st.sidebar.checkbox("Enable ID card detection", value=cfg.enable_id_card_detection)
		cfg.id_card_required = st.sidebar.checkbox("ID card required", value=cfg.id_card_required)
		cfg.id_card_confidence_threshold = st.sidebar.slider("ID card confidence threshold", 0.1, 0.9, float(cfg.id_card_confidence_threshold), 0.05)

		# Uniform policy settings
		st.sidebar.markdown("---")
		st.sidebar.subheader("Uniform Policy")
		cfg.policy_gender = st.sidebar.selectbox("Gender", ["male", "female"], index=0 if cfg.policy_gender == "male" else 1)
		cfg.require_shirt_for_male = st.sidebar.checkbox("Require shirt for males", value=getattr(cfg, "require_shirt_for_male", True))
		cfg.require_black_shoes_male = st.sidebar.checkbox("Require black shoes for males", value=getattr(cfg, "require_black_shoes_male", True))
		cfg.allow_any_color_pants_male = st.sidebar.checkbox("Allow any color pants for males", value=getattr(cfg, "allow_any_color_pants_male", True))
		cfg.require_footwear_male = st.sidebar.checkbox("Require footwear for males", value=cfg.require_footwear_male)

		st.sidebar.markdown("---")
		if st.sidebar.button("Load saved model"):
			try:
				st.session_state.classifier.load(cfg.model_path)
				st.success("Model loaded")
			except (IOError, ValueError) as e:
				st.error(f"Failed to load model: {e}")

		if st.sidebar.button("Clear session state"):
			for k in list(st.session_state.keys()):
				if k not in ["config"]:
					del st.session_state[k]
			st.rerun()


def render_home():
	st.title("🏫 Student Attire Verification System")
	st.markdown("---")

	# Welcome message
	st.markdown("""
	### Welcome to the Student Attire Verification System
	
	This system helps ensure compliance with dress code and safety regulations across campus.
	""")
	
	st.markdown("---")
	
	# Quick access cards
	st.markdown("### 🚀 Quick Access")
	
	col1, col2, col3 = st.columns(3)
	
	with col1:
		st.markdown("#### 🎓 Students")
		st.write("Login, register, and verify your attire compliance")
		if st.button("Go to Student Portal", use_container_width=True, type="primary", key="home_student"):
			navigate_to("student_portal")
	
	with col2:
		st.markdown("#### 👨‍💼 Admin")
		st.write("Manage students, departments, and view reports")
		if st.button("Go to Admin Portal", use_container_width=True, type="primary", key="home_admin"):
			navigate_to("admin_login")
	
	with col3:
		st.markdown("#### ℹ️ About")
		st.write("Learn more about the system and features")
		if st.button("View Information", use_container_width=True, key="home_info"):
			st.info("📚 System Information coming soon!")
	
	st.markdown("---")
	
	# System features
	st.markdown("### ✨ Key Features")
	
	col1, col2 = st.columns(2)
	
	with col1:
		st.markdown("""
		**For Students:**
		- 🔐 Face-based biometric authentication
		- 📸 Real-time attire verification
		- 📊 Personal compliance dashboard
		- 📱 Easy registration process
		""")
	
	with col2:
		st.markdown("""
		**For Administrators:**
		- 👥 Student management
		- 🏢 Department organization
		- 📈 Compliance reports
		- 🔔 Alert system
		""")


def render_student_portal():
	"""Student portal - landing page for students"""
	st.title("🎓 Student Portal")
	st.markdown("---")

	# Check if user is already logged in
	if is_logged_in():
		user = st.session_state.get('user')
		st.success(f"✅ Welcome back, {user.get('full_name', 'User')}!")
		
		st.markdown("### Quick Actions")
		col1, col2, col3 = st.columns(3)
		
		with col1:
			if is_student():
				if st.button("📋 My Dashboard", use_container_width=True, type="primary"):
					navigate_to("student_dashboard")
			else:
				if st.button("🎓 Verify Attire", use_container_width=True, type="primary"):
					navigate_to("verification")
		
		with col2:
			if st.button("👤 My Profile", use_container_width=True, key="profile_btn_portal"):
				navigate_to("profile")
		
		with col3:
			if st.button("🚪 Logout", use_container_width=True, key="logout_btn_portal"):
				logout_and_redirect()
	else:
		st.markdown("""
		### Welcome to Student Portal!

		Are you a new student or an existing user?
		""")

		col1, col2 = st.columns(2)

		with col1:
			st.markdown("#### 📝 New Student")
			st.write("Create your account and register your face for biometric authentication")
			if st.button("Register Now", use_container_width=True, type="primary", key="portal_register"):
				navigate_to('register')

		with col2:
			st.markdown("#### 🔐 Existing Student")
			st.write("Login using face authentication or emergency credentials")
			if st.button("Login Now", use_container_width=True, type="primary", key="portal_login"):
				navigate_to('face_auth')

		st.markdown("---")
		
		# Additional info
		with st.expander("ℹ️ How to get started", expanded=False):
			st.markdown("""
			**For New Students:**
			1. Click "Register Now"
			2. Fill in your details
			3. Capture your face photo
			4. Complete registration
			
			**For Existing Students:**
			1. Click "Login Now"
			2. Use face authentication
			3. Access your dashboard
			4. Verify your attire
			""")
		
		st.caption("💡 Tip: Make sure you have good lighting for face authentication")


def show_id_card_popup(violations: List[Dict[str, Any]]) -> bool:
	"""Show ID card popup message and return True if ID card violation found"""
	id_card_violations = []
	
	for violation in violations:
		if "id" in violation.get('item', '').lower() or "card" in violation.get('item', '').lower():
			id_card_violations.append(violation)
	
	if id_card_violations:
		st.error("🆔 **ID CARD REQUIRED!**")
		st.warning("⚠️ **Please wear your student ID card visibly!**")
		
		# Show specific ID card violation details
		for violation in id_card_violations:
			st.error(f"❌ **{violation.get('item', 'ID Card')}**: {violation.get('detected', 'Not detected')}")
			st.info(f"📋 **Required**: {violation.get('required', 'Valid student ID card visible')}")
		
		# Show instructions
		with st.expander("📝 **How to fix ID card issue:**", expanded=True):
			st.write("1. **Wear your student ID card** around your neck or on your shirt")
			st.write("2. **Make sure it's visible** and not covered by clothing")
			st.write("3. **Ensure good lighting** so the card can be clearly seen")
			st.write("4. **Position yourself** so the ID card is in the camera view")
			st.write("5. **Try again** by taking a new photo")
		
		return True
	return False


def handle_image(image: Image.Image, zone: str, student_id: Optional[str]) -> Dict[str, any]:
	cfg: AppConfig = st.session_state.config
	
	# SECURITY CHECK 1: Unauthorized Entry Detection
	if student_id:
		is_authorized = check_and_alert_unauthorized_student(student_id, zone, cfg)
		if not is_authorized:
			# Return early if unauthorized student detected
			return {
				"annotated": None,
				"result": {"status": "UNAUTHORIZED", "error": "Unauthorized student detected"},
				"features": {},
				"event_id": None,
			}
	
	bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
	pose = extract_pose(bgr)
	features = extract_features_from_image(bgr, pose_landmarks=pose, bins=cfg.hist_bins)
	result = verify_attire_and_safety(features, cfg, st.session_state.classifier)

	annotated = draw_pose_annotations(bgr.copy(), pose)
	
	# Add violation indicators
	violations = result.get("violations", {}).get("violations", [])
	annotated = draw_violation_indicators(annotated, pose, violations)
	
	# Add detailed badge
	annotated = overlay_detailed_badge(annotated, result)

	# Save to dataset optionally
	image_path = None
	if st.session_state.config.save_frames:
		image_path = str(append_sample_to_dataset(bgr, cfg.current_label, features))
	
	# SECURITY CHECK 2: Late Entry / Early Exit Tracking
	access_info = {}
	if student_id:
		# Check for entry (entering campus)
		if "Gate" in zone or "Entry" in zone:
			access_info = check_and_log_entry_time(student_id, zone, cfg)
		# Check for exit (leaving campus)
		elif "Exit" in zone or "Gate" in zone:
			access_info = check_and_log_exit_time(student_id, zone, cfg)

	# Log event
	event_id = insert_event({
		"student_id": student_id,
		"zone": zone,
		"status": result["status"],
		"score": result["score"],
		"label": result.get("label"),
		"details": str(result.get("details")),
		"image_path": image_path,
	})

	# Alerts on non-compliance
	if result["status"] != "PASS":
		notify_non_compliance(student_id, zone, str(result.get("details")))
	
	# SECURITY CHECK 3: Emergency Alerts for Incomplete Attire
	if student_id and violations:
		check_and_alert_emergency_violations(student_id, zone, violations, cfg)

	# Send ID card status notification
	if cfg.enable_id_card_detection:
		id_card_detected = features.get("id_card_detected", 0.0) > 0.5
		id_card_confidence = features.get("id_card_confidence", 0.0)
		notify_id_card_status(student_id, zone, id_card_detected, id_card_confidence, cfg)

	return {
		"annotated": annotated,
		"result": result,
		"features": features,  # Include features for ID card display
		"event_id": event_id,
		"access_info": access_info,  # Include access timing info
	}


def render_image_tab():
	st.subheader("Single Image")
	# Pre-fill student id when user is logged in (roll_no or id)
	_user = st.session_state.get('user') or {}
	prefill_id = _user.get('roll_no') or _user.get('student_id') or _user.get('id') or ""
	student_id = st.text_input("Student ID / RFID (optional)", value=prefill_id, key="student_id_image")
	zone = st.selectbox("Zone", st.session_state.config.zones, index=0, key="zone_image")
	upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="upload_image")
	if upload is not None:
		img = Image.open(upload).convert("RGB")
		resp = handle_image(img, zone, student_id or None)
		col1, col2 = st.columns(2)
		with col1:
			st.image(img, caption="Input")
		with col2:
			st.image(cv2.cvtColor(resp["annotated"], cv2.COLOR_BGR2RGB), caption="Analysis")
		
		# Display detailed results
		result = resp["result"]
		features = resp.get("features", {})
		
		# Show ID Card Detection Status
		st.markdown("---")
		st.subheader("🆔 ID Card Detection Status")
		
		if st.session_state.config.enable_id_card_detection:
			# Get ID card detection status
			id_card_detected = features.get("id_card_detected", 0.0) > 0.5
			id_card_confidence = features.get("id_card_confidence", 0.0)
			id_card_area = features.get("id_card_area", 0)
			
			# Display main status message
			if id_card_detected:
				if id_card_confidence >= 0.8:
					st.success(f"✅ **ID CARD DETECTED** - High Confidence ({id_card_confidence:.1%})")
				elif id_card_confidence >= 0.6:
					st.success(f"✅ **ID CARD DETECTED** - Good Confidence ({id_card_confidence:.1%})")
				else:
					st.warning(f"⚠️ **ID CARD DETECTED** - Low Confidence ({id_card_confidence:.1%})")
				
				# Show detection details
				with st.expander("📊 ID Card Detection Details", expanded=True):
					st.write(f"**Detection Status:** ✅ Detected")
					st.write(f"**Confidence Level:** {id_card_confidence:.1%}")
					st.write(f"**Detected Area:** {id_card_area:.0f} pixels")
					st.info("✅ Your student ID card is visible and properly detected!")
			else:
				st.error(f"❌ **NO ID CARD DETECTED** (confidence: {id_card_confidence:.1%})")
				
				# Show violation message and help instructions
				with st.expander("⚠️ ID Card Violation - How to Fix", expanded=True):
					st.error("**VIOLATION:** Student ID card is required but not detected!")
					st.write("📋 **Please follow these steps:**")
					st.write("1. **Wear your student ID card** clearly visible around your neck or on your shirt")
					st.write("2. **Ensure the ID card is not covered** by clothing or other objects")
					st.write("3. **Position yourself** so the ID card is in the center of the camera view")
					st.write("4. **Ensure good lighting** so the ID card is clearly visible")
					st.write("5. **Take a new photo** with your ID card properly visible")
					st.warning("**Note:** Your ID card must meet the minimum confidence threshold of 60% to pass verification.")
		else:
			st.info("ℹ️ ID card detection is disabled in settings")
		
		st.markdown("---")
		
		# Overall status and scores
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Status", result["status"], delta=None)
		with col2:
			st.metric("Success Score", f"{result['success_score']:.1%}")
		with col3:
			st.metric("Fail Score", f"{result['fail_score']:.1%}")
		
		# Check for ID card violations first and show popup
		violations = result.get("violations", {})
		if violations.get("total_violations", 0) > 0:
			# Show ID card popup if detected
			has_id_violation = show_id_card_popup(violations.get("violations", []))
			
			# Display all violations
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
		
		# Summary
		summary = result.get("summary", {})
		if summary:
			st.info(f"**Overall Compliance:** {summary.get('overall_compliance', 'N/A')}")
		
		# Raw data (collapsible)
		with st.expander("Raw Analysis Data"):
			st.json(result)
		
		st.caption(f"Logged event #{resp['event_id']}")

		# Action buttons: Accuracy & Report
		col_a, col_b = st.columns(2)
		with col_a:
			if st.button("Accuracy", key="accuracy_image"):
				from src.db import get_student
				st.markdown("**Accuracy Details**")
				st.write(f"Overall Score: {result.get('score', 0):.3f}")
				st.write(f"Success Score: {result.get('success_score', 0):.3%}")
				st.write(f"Fail Score: {result.get('fail_score', 0):.3%}")
				if student_id:
					student = get_student(student_id, cfg=st.session_state.config)
					if student:
						st.markdown("**Student Details**")
						st.json(student)
		with col_b:
			if st.button("Report", key="report_image"):
				from src.db import get_events_for_student
				if student_id:
					events = get_events_for_student(student_id, limit=50, cfg=st.session_state.config)
					if events:
						st.subheader("Report: Recent Events")
						st.dataframe(pd.DataFrame(events))
				else:
					st.info("No student ID provided for report")


def render_webcam_tab():
	st.subheader("Webcam")
	_user = st.session_state.get('user') or {}
	prefill_id = _user.get('roll_no') or _user.get('student_id') or _user.get('id') or ""
	student_id = st.text_input("Student ID / RFID (optional)", value=prefill_id, key="student_id_webcam")
	zone = st.selectbox("Zone", st.session_state.config.zones, index=0, key="zone_webcam")
	st.info("Use the camera to capture a frame.")
	cam = st.camera_input("Capture a frame", key="cam_webcam")
	if cam is not None:
		img = Image.open(io.BytesIO(cam.getvalue())).convert("RGB")
		resp = handle_image(img, zone, student_id or None)
		st.image(cv2.cvtColor(resp["annotated"], cv2.COLOR_BGR2RGB), caption="Analysis")
		
		# Display detailed results (same as image tab)
		result = resp["result"]
		
		# Overall status and scores
		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Status", result["status"], delta=None)
		with col2:
			st.metric("Success Score", f"{result['success_score']:.1%}")
		with col3:
			st.metric("Fail Score", f"{result['fail_score']:.1%}")
		
		# Check for ID card violations first and show popup
		violations = result.get("violations", {})
		if violations.get("total_violations", 0) > 0:
			# Show ID card popup if detected
			has_id_violation = show_id_card_popup(violations.get("violations", []))
			
			# Display all violations
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
		
		# Summary
		summary = result.get("summary", {})
		if summary:
			st.info(f"**Overall Compliance:** {summary.get('overall_compliance', 'N/A')}")
		
		# Raw data (collapsible)
		with st.expander("Raw Analysis Data"):
			st.json(result)
		
		st.caption(f"Logged event #{resp['event_id']}")

		# Action buttons: Accuracy & Report (webcam)
		col_a, col_b = st.columns(2)
		with col_a:
			if st.button("Accuracy", key="accuracy_webcam"):
				from src.db import get_student
				st.markdown("**Accuracy Details**")
				st.write(f"Overall Score: {result.get('score', 0):.3f}")
				st.write(f"Success Score: {result.get('success_score', 0):.3%}")
				st.write(f"Fail Score: {result.get('fail_score', 0):.3%}")
				if student_id:
					student = get_student(student_id, cfg=st.session_state.config)
					if student:
						st.markdown("**Student Details**")
						st.json(student)
		with col_b:
			if st.button("Report", key="report_webcam"):
				from src.db import get_events_for_student
				if student_id:
					events = get_events_for_student(student_id, limit=50, cfg=st.session_state.config)
					if events:
						st.subheader("Report: Recent Events")
						st.dataframe(pd.DataFrame(events))
				else:
					st.info("No student ID provided for report")


def render_video_tab():
	st.subheader("Video")
	_user = st.session_state.get('user') or {}
	prefill_id = _user.get('roll_no') or _user.get('student_id') or _user.get('id') or ""
	student_id = st.text_input("Student ID / RFID (optional)", value=prefill_id, key="student_id_video")
	zone = st.selectbox("Zone", st.session_state.config.zones, index=0, key="zone_video")
	video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"], key="upload_video")
	if video is None:
		return

	cfg: AppConfig = st.session_state.config
	bytes_data = video.read()
	tmp_path = "tmp_video.mp4"
	with open(tmp_path, "wb") as f:
		f.write(bytes_data)

	cap = cv2.VideoCapture(tmp_path)
	if not cap.isOpened():
		st.error("Failed to open video")
		return

	progress = st.progress(0)
	frame_area = st.empty()
	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	fps = max(1, min(cfg.max_video_fps, int(cap.get(cv2.CAP_PROP_FPS) or 15)))
	sample_every = max(1, int((cap.get(cv2.CAP_PROP_FPS) or 15) // fps))

	ok_frames = 0
	total_frames = 0
	idx = 0
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
		result = verify_attire_and_safety(features, cfg, st.session_state.classifier)
		annot = draw_pose_annotations(frame.copy(), pose)
		
		# Add violation indicators
		violations = result.get("violations", {}).get("violations", [])
		annot = draw_violation_indicators(annot, pose, violations)
		
		# Add detailed badge
		annot = overlay_detailed_badge(annot, result)
		frame_area.image(cv2.cvtColor(annot, cv2.COLOR_BGR2RGB))

		if st.session_state.config.save_frames and (idx % (sample_every * 5) == 0):
			append_sample_to_dataset(frame, cfg.current_label, features)

		total_frames += 1
		ok_frames += 1 if result["status"] == "PASS" else 0
		if frame_count > 0:
			progress.progress(min(1.0, idx / frame_count))

	cap.release()
	os.remove(tmp_path)

	st.success(f"Completed. PASS ratio: {ok_frames}/{total_frames} = {ok_frames/max(1,total_frames):.2f}")


def render_dataset_tab():
	st.subheader("Dataset & Training")
	ensure_dirs()

	st.markdown("#### Collected Samples")
	if st.button("Refresh dataset view") or st.session_state.dataset_df is None:
		st.session_state.dataset_df = load_dataset()
	st.dataframe(st.session_state.dataset_df)

	st.markdown("#### Train Classifier")
	label_col = st.selectbox("Label column", ["label"], index=0, key="label_col_dataset")
	if st.button("Train"):
		clf: AttireClassifier = st.session_state.classifier
		info = clf.train_from_dataframe(st.session_state.dataset_df, label_col=label_col, bins=st.session_state.config.hist_bins)
		clf.save(st.session_state.config.model_path)
		st.session_state.last_train_info = info
		st.success(f"Model trained. acc={info['cv_accuracy']:.2f}, n={info['num_samples']}")

	if st.session_state.last_train_info:
		st.json(st.session_state.last_train_info)




def render_student_verification():
	st.title("Student Verification")
	tabs = st.tabs(["Image", "Webcam", "Video"])
	with tabs[0]:
		render_image_tab()
	with tabs[1]:
		render_webcam_tab()
	with tabs[2]:
		render_video_tab()


def render_admin_tab():
	st.title("Admin Dashboard")
	
	# Get compliance stats
	stats = get_compliance_stats(cfg=st.session_state.config)
	
	# Display key metrics
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Total Students", stats["total_students"])
	with col2:
		st.metric("Verified Students", stats["verified_students"])
	with col3:
		st.metric("Compliance Rate", f"{stats['compliance_percentage']:.1f}%")
	with col4:
		st.metric("Total Events Today", stats["total_events"])
	
	st.markdown("---")
	
	# Tabs for different admin functions
	tab1, tab2, tab3, tab4, tab5 = st.tabs([
		"Students", 
		"Compliance Reports", 
		"Add Student", 
		"➕ Add Department",
		"📊 Departments"
	])
	
	with tab1:
		students = get_all_students(cfg=st.session_state.config)
		st.subheader("All Students")
		if students:
			# Enrich students with aggregated verification stats
			from src.db import get_student_stats
			rows = []
			total_avg_score = 0.0
			total_pass_rate = 0.0
			verified_count = 0
			for s in students:
				stats = get_student_stats(s.get('id'), cfg=st.session_state.config)
				row = dict(s)
				row['avg_score'] = stats.get('avg_score')
				row['pass_rate'] = stats.get('pass_rate')
				row['last_verified'] = stats.get('last_event')
				rows.append(row)
				if stats.get('avg_score') is not None:
					total_avg_score += stats.get('avg_score', 0.0)
				if stats.get('pass_rate') is not None:
					total_pass_rate += stats.get('pass_rate', 0.0)
				if s.get('verified', 0):
					verified_count += 1
			
			# Display aggregate metric cards
			st.markdown("#### 📊 Class Statistics")
			metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
			
			with metric_col1:
				st.metric("Total Students", len(students), delta=None)
			
			with metric_col2:
				st.metric("Verified", verified_count, delta=f"{verified_count}/{len(students)}")
			
			with metric_col3:
				avg_score = total_avg_score / len(rows) if rows else 0.0
				st.metric("Average Score", f"{avg_score:.1%}", delta=None)
			
			with metric_col4:
				avg_pass_rate = total_pass_rate / len(rows) if rows else 0.0
				st.metric("Pass Rate", f"{avg_pass_rate:.1%}", delta=None)
			
			st.markdown("---")
			
			# Display detailed table
			st.markdown("#### Student Details")
			df = pd.DataFrame(rows)
			st.dataframe(df)
			
			st.markdown("---")
			
			# Delete student section
			st.markdown("#### 🗑️ Delete Student")
			st.warning("⚠️ **Warning:** Deleting a student will permanently remove all their data including events, face images, and login credentials.")
			
			col1, col2 = st.columns([3, 1])
			with col1:
				student_ids = [s['id'] for s in students]
				student_options = [f"{s['id']} - {s['name']}" for s in students]
				selected_student = st.selectbox(
					"Select student to delete",
					options=range(len(student_options)),
					format_func=lambda x: student_options[x],
					key="delete_student_select"
				)
			
			with col2:
				st.write("")  # Spacing
				st.write("")  # Spacing
				if st.button("🗑️ Delete Student", type="secondary", use_container_width=True):
					student_id = student_ids[selected_student]
					student_name = students[selected_student]['name']
					
					# Confirmation dialog
					st.session_state['confirm_delete_student'] = student_id
					st.session_state['confirm_delete_name'] = student_name
			
			# Show confirmation dialog if delete was clicked
			if st.session_state.get('confirm_delete_student'):
				st.error(f"⚠️ **Confirm Deletion**")
				st.write(f"Are you sure you want to delete student **{st.session_state.get('confirm_delete_name')}** (ID: {st.session_state.get('confirm_delete_student')})?")
				st.write("This action cannot be undone!")
				
				col1, col2, col3 = st.columns([1, 1, 2])
				with col1:
					if st.button("✅ Yes, Delete", type="primary", use_container_width=True):
						from src.db import delete_student
						success, message = delete_student(st.session_state['confirm_delete_student'], cfg=st.session_state.config)
						
						if success:
							st.success(f"✅ {message}")
							# Delete face image file if exists
							import os
							face_storage = Path("data/face_storage")
							for file in face_storage.glob(f"*{st.session_state['confirm_delete_student']}*"):
								try:
									os.remove(file)
								except:
									pass
							
							# Clear confirmation
							del st.session_state['confirm_delete_student']
							del st.session_state['confirm_delete_name']
							st.rerun()
						else:
							st.error(f"❌ {message}")
				
				with col2:
					if st.button("❌ Cancel", use_container_width=True):
						del st.session_state['confirm_delete_student']
						del st.session_state['confirm_delete_name']
						st.rerun()
		else:
			st.info("No students in database")
	
	with tab2:
		st.subheader("Daily Compliance Report")
		compliance_df = pd.DataFrame([stats])
		st.dataframe(compliance_df)
		
		# Download button
		csv = compliance_df.to_csv(index=False).encode('utf-8')
		st.download_button("Download Compliance Report", csv, "compliance_report.csv", "text/csv")
	
	with tab3:
		st.subheader("Add/Update Student")

		# Get existing departments for selection
		from src.db import get_all_departments
		departments = get_all_departments(cfg=st.session_state.config)
		dept_options = [""] + [f"{d['name']} ({d['code']})" for d in departments]

		# Stage 1: Student ID Generation
		st.markdown("### 📝 Stage 1: Generate Student ID")
		with st.form("student_id_form"):
			col1, col2 = st.columns(2)

			with col1:
				batch_year = st.number_input("Batch Year *", min_value=2000, max_value=2100, value=2022, step=1, help="e.g., 2022 for 2022-2026 batch")
				selected_dept = st.selectbox("Department *", dept_options, index=0, help="Select from existing departments")

			with col2:
				section_options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
				section = st.selectbox("Section *", section_options, index=0, help="A=1, B=2, C=3, etc.")
				student_number = st.number_input("Student Number *", min_value=1, max_value=999, value=1, step=1, help="Joining order (001-999)")

			# Auto-generate Student ID
			auto_student_id = ""
			auto_class = ""
			if selected_dept and batch_year:
				# Extract department code and ID
				dept_name = selected_dept.split(" (")[0] if " (" in selected_dept else selected_dept
				dept_info = next((d for d in departments if d['name'] == dept_name), None)
				if dept_info:
					dept_code = dept_info['code'].upper()  # Department code (alphabets like CS, EE, ME)
					batch_yy = str(batch_year)[-2:]  # Last 2 digits of year
					section_letter = section  # Section letter (A, B, C, etc.)
					student_num = f"{student_number:03d}"  # 3-digit student number
					# New format: YY-CODE-SECTION-NNN (e.g., 22-CS-A-001)
					auto_student_id = f"{batch_yy}{dept_code}{section_letter}{student_num}"
					auto_class = f"{dept_info['code']}-{section}"

			col1, col2 = st.columns(2)
			with col1:
				st.text_input("Auto-Generated Student ID", value=auto_student_id, disabled=True, key="auto_student_id")
				if auto_student_id:
					st.caption(f"Format: YY(Year) + CODE(Dept) + SECTION + NNN(Number)")
			with col2:
				st.text_input("Auto-Generated Class", value=auto_class, disabled=True, key="auto_class")
				if auto_class:
					st.caption(f"Class: {auto_class}")

			generate_id = st.form_submit_button("Generate ID", use_container_width=True, type="secondary")

		# Stage 2: Student Details (only show if ID is generated)
		if auto_student_id:
			st.markdown("---")
			st.markdown("### 👤 Stage 2: Student Details")

			with st.form("student_details_form"):
				col1, col2 = st.columns(2)

				with col1:
					name = st.text_input("Full Name *", key="student_name")

				with col2:
					# Gender as radio buttons (Male/Female only)
					gender = st.radio("Gender *", ["Male", "Female"], index=0, key="student_gender_radio")
					gender_code = "M" if gender == "Male" else "F"

				email = st.text_input("Email", key="student_email")
				phone = st.text_input("Phone", key="student_phone")
				contact_info = st.text_area("Contact Info", height=80, key="student_contact")

				if st.form_submit_button("Add Student", use_container_width=True, type="primary"):
					if not name:
						st.error("Full Name is required!")
						return

					# Extract department name for storage
					dept_name = selected_dept.split(" (")[0] if " (" in selected_dept else selected_dept

					add_student({
						"id": auto_student_id,
						"name": name.strip(),
						"class": auto_class,
						"department": dept_name,
						"gender": gender_code,
						"email": email.strip() if email else None,
						"phone": phone.strip() if phone else None,
						"contact_info": contact_info.strip() if contact_info else None
					}, cfg=st.session_state.config)
					st.success(f"✅ Student added successfully! ID: {auto_student_id}")
					st.info(f"📚 Class: {auto_class} | Department: {dept_name}")
					
					# Show login credentials
					st.markdown("---")
					st.markdown("### 🔐 Login Credentials Created")
					col1, col2 = st.columns(2)
					with col1:
						st.write(f"**Username:** `{auto_student_id}`")
						st.write(f"**Password:** `{auto_student_id}`")
					with col2:
						st.write(f"**Login Method:** Face Authentication or Username/Password")
						st.caption("Student can change password after first login")
					
					st.warning("⚠️ **Important:** Share these credentials with the student securely.")
					st.balloons()
		else:
			st.info("👆 Please generate a Student ID first in Stage 1 above.")
	
	with tab4:
		render_add_department_tab()
	
	with tab5:
		render_departments_tab()


def render_add_department_tab():
	"""Render the Add Department form"""
	st.subheader("➕ Create New Department")

	with st.form("add_department_form"):
		col1, col2 = st.columns(2)

		with col1:
			dept_name = st.text_input("Department Name *", placeholder="e.g., Computer Science")
			dept_code = st.text_input("Department Code", placeholder="e.g., CS (auto-fill from name)")
			num_classes = st.number_input("Number of Classes *", min_value=1, max_value=26, value=1, step=1)

		with col2:
			head_name = st.text_input("Department Head (Optional)", placeholder="e.g., Prof. John Doe")
			head_email = st.text_input("Head Email (Optional)")
			location = st.text_input("Location (Optional)", placeholder="e.g., Block A, 2nd Floor")

		dept_email = st.text_input("Department Email (Optional)")
		dept_phone = st.text_input("Department Phone (Optional)")
		description = st.text_area("Description (Optional)", height=80)

		submitted = st.form_submit_button("Create Department", use_container_width=True, type="primary")

	if submitted:
		if not dept_name:
			st.error("Department Name is required!")
			return

		# Auto-generate code from name if not provided
		auto_code = dept_code.strip().upper() if dept_code.strip() else dept_name[:2].upper()

		from src.db import add_department

		dept_data = {
			"name": dept_name.strip(),
			"code": auto_code,
			"short_form": auto_code,
			"head_name": head_name.strip() if head_name else None,
			"head_email": head_email.strip() if head_email else None,
			"number_of_classes": int(num_classes),
			"location": location.strip() if location else None,
			"email": dept_email.strip() if dept_email else None,
			"phone": dept_phone.strip() if dept_phone else None,
			"description": description.strip() if description else None
		}

		success, dept_id, message = add_department(dept_data, cfg=st.session_state.config)

		if success:
			st.success(f"✅ {message}")
			st.info(f"Department created with {num_classes} classes: {auto_code}-A, {auto_code}-B, etc.")
			st.rerun()
		else:
			st.error(f"❌ Error: {message}")


def render_departments_tab():
	"""Render the Departments management view"""
	st.subheader("📊 Departments Management")
	
	from src.db import get_all_departments, get_department_statistics, update_department, delete_department, search_departments, export_department_report
	
	# Search functionality
	col1, col2 = st.columns([3, 1])
	with col1:
		search_term = st.text_input("🔍 Search departments by name or code", key="dept_search")
	with col2:
		st.write("")  # Spacing
		if st.button("🔄 Refresh", key="refresh_depts"):
			st.rerun()
	
	# Get departments
	if search_term.strip():
		departments = search_departments(search_term, cfg=st.session_state.config)
	else:
		departments = get_all_departments(cfg=st.session_state.config)
	
	if not departments:
		st.info("No departments found. Create one in the '➕ Add Department' tab.")
		return
	
	# Display departments table
	st.markdown("#### All Departments")

	# Create table data
	table_data = []
	for dept in departments:
		table_data.append({
			"Department": dept['name'],
			"Code": dept['code'],
			"Classes": dept['number_of_classes'],
			"Students": dept.get('total_students', 0),
			"Head": dept['head_name'] or "N/A",
			"Location": dept['location'] or "N/A",
			"Status": "✅ Active" if dept['status'] == 'active' else "⚠️ Inactive"
		})

	df_depts = pd.DataFrame(table_data)
	st.dataframe(df_depts, use_container_width=True, hide_index=True)
	
	# Department details section
	st.markdown("---")
	st.markdown("#### Department Details & Analytics")
	
	# Select department
	dept_options = [d['name'] for d in departments]
	selected_dept_name = st.selectbox("Select Department to View Details", dept_options, key="dept_select")
	
	if selected_dept_name:
		# Get full dept info
		selected_dept = next((d for d in departments if d['name'] == selected_dept_name), None)
		if selected_dept:
			dept_id = selected_dept['id']
			
			# Create tabs for department details
			detail_tabs = st.tabs(["Overview", "Statistics", "Classes", "Students", "Edit", "Actions"])
			
			# Overview tab
			with detail_tabs[0]:
				st.subheader(f"📋 {selected_dept['name']} Overview")

				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.metric("Department Code", selected_dept['code'])
				with col2:
					st.metric("Short Form", selected_dept['short_form'])
				with col3:
					st.metric("Total Classes", selected_dept['number_of_classes'])
				with col4:
					st.metric("Status", "✅ Active" if selected_dept['status'] == 'active' else "⚠️ Inactive")

				col1, col2 = st.columns(2)
				with col1:
					st.write(f"**Department Head:** {selected_dept['head_name'] or 'Not assigned'}")
					st.write(f"**Head Email:** {selected_dept['head_email'] or 'Not provided'}")
				with col2:
					st.write(f"**Location:** {selected_dept['location'] or 'Not provided'}")
					st.write(f"**Department Email:** {selected_dept['email'] or 'Not provided'}")
					st.write(f"**Phone:** {selected_dept['phone'] or 'Not provided'}")

				if selected_dept['description']:
					st.write(f"**Description:** {selected_dept['description']}")
			
			# Statistics tab
			with detail_tabs[1]:
				st.subheader("📊 Student Statistics")
				
				stats = get_department_statistics(dept_id, cfg=st.session_state.config)
				
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					st.metric("Total Students", stats.get('total_students', 0))
				with col2:
					st.metric("👨 Male", stats.get('male_count', 0))
				with col3:
					st.metric("👩 Female", stats.get('female_count', 0))
				with col4:
					st.metric("❓ Unknown", stats.get('unknown_count', 0))
				
				# Gender distribution chart
				if stats.get('total_students', 0) > 0:
					gender_data = {
						"Male": stats.get('male_count', 0),
						"Female": stats.get('female_count', 0),
						"Unknown": stats.get('unknown_count', 0)
					}
					st.bar_chart(gender_data)
			
			# Classes tab
			with detail_tabs[2]:
				st.subheader("📚 Classes in this Department")
				
				from src.db import get_classes_by_department
				classes = get_classes_by_department(dept_id, cfg=st.session_state.config)
				
				if classes:
					for idx, cls in enumerate(classes):
						with st.expander(f"Class {cls['class_letter']} ({cls['class_code']}) - {cls.get('student_count', 0)} students", expanded=False):
							col1, col2 = st.columns(2)
							with col1:
								st.write(f"**Class Code:** {cls['class_code']}")
								st.write(f"**Students:** {cls.get('student_count', 0)}/{cls.get('capacity', 50)}")
							with col2:
								st.write(f"**Class Advisor:** {cls['class_advisor'] or 'Not assigned'}")
								st.write(f"**Room Number:** {cls['room_number'] or 'Not assigned'}")
							
							# Edit class details
							with st.form(f"edit_class_form_{cls['id']}"):
								col1, col2 = st.columns(2)
								with col1:
									new_advisor = st.text_input("Class Advisor Name", value=cls['class_advisor'] or "", key=f"advisor_{cls['id']}")
								with col2:
									new_room = st.text_input("Room Number", value=cls['room_number'] or "", key=f"room_{cls['id']}")
								
								if st.form_submit_button("Update Class"):
									from src.db import update_class_advisor, update_class_room
									if new_advisor:
										update_class_advisor(cls['id'], new_advisor, cfg=st.session_state.config)
									if new_room:
										update_class_room(cls['id'], new_room, cfg=st.session_state.config)
									st.success("Class updated!")
									st.rerun()
				else:
					st.info("No classes in this department")
			
			# Students tab
			with detail_tabs[3]:
				st.subheader("👥 Students in this Department")
				
				from src.db import get_students_by_department
				students = get_students_by_department(selected_dept['name'], cfg=st.session_state.config)
				
				if students:
					st_df = pd.DataFrame(students)
					st.dataframe(st_df, use_container_width=True, hide_index=True)
					st.info(f"Total: {len(students)} students")
				else:
					st.info("No students in this department yet")
			
			# Edit tab
			with detail_tabs[4]:
				st.subheader("✏️ Edit Department Information")
				
				# Show success message if it exists in session state
				if st.session_state.get('dept_update_success'):
					st.success("✅ Changes have been saved successfully!")
					st.info(f"✓ Department Code: {st.session_state.get('dept_update_code', '')} (used for class names like {st.session_state.get('dept_update_code', '')}-A, {st.session_state.get('dept_update_code', '')}-B)")
					# Clear the message after displaying
					del st.session_state['dept_update_success']
					if 'dept_update_code' in st.session_state:
						del st.session_state['dept_update_code']
				
				st.info("ℹ️ **Note:** Department Code is used for class names (e.g., CS-A, EE-B). Use 2-4 letter abbreviations.")
				
				with st.form("edit_department_form"):
					col1, col2 = st.columns(2)
					
					with col1:
						new_name = st.text_input("Department Name", value=selected_dept['name'])
						new_code = st.text_input("Department Code (Alphabetic) *", value=selected_dept['code'], 
							help="Used for class names. E.g., CS, EE, ME, BBA")
					
					with col2:
						new_head = st.text_input("Department Head", value=selected_dept['head_name'] or "")
						new_head_email = st.text_input("Head Email", value=selected_dept['head_email'] or "")
					
					new_location = st.text_input("Location", value=selected_dept['location'] or "")
					new_email = st.text_input("Department Email", value=selected_dept['email'] or "")
					new_phone = st.text_input("Phone", value=selected_dept['phone'] or "")
					new_description = st.text_area("Description", value=selected_dept['description'] or "", height=80)
					
					if st.form_submit_button("Save Changes", type="primary"):
						# Validate code is alphabetic
						if not new_code.strip().replace("-", "").isalpha():
							st.error("❌ Department Code must contain only letters (e.g., CS, EE, ME)")
							st.stop()
						
						# Normalize code to uppercase
						normalized_code = new_code.strip().upper()
						
						update_data = {
							"name": new_name,
							"code": normalized_code,
							"short_form": normalized_code,  # Keep synchronized with code
							"head_name": new_head,
							"head_email": new_head_email,
							"location": new_location,
							"email": new_email,
							"phone": new_phone,
							"description": new_description
						}
						success, msg = update_department(dept_id, update_data, cfg=st.session_state.config)
						if success:
							# Store success message in session state
							st.session_state['dept_update_success'] = True
							st.session_state['dept_update_code'] = normalized_code
							st.rerun()
						else:
							st.error(f"❌ {msg}")
			
			# Actions tab
			with detail_tabs[5]:
				st.subheader("⚙️ Actions")
				
				col1, col2, col3 = st.columns(3)
				
				with col1:
					if st.button("📥 Export as CSV", use_container_width=True, key="export_dept"):
						csv_data = export_department_report(dept_id, cfg=st.session_state.config)
						if csv_data:
							st.download_button(
								label="Download Report",
								data=csv_data,
								file_name=f"{selected_dept['code']}_report.csv",
								mime="text/csv"
							)
				
				with col2:
					st.write("")  # Spacing
				
				with col3:
					if st.button("🗑️ Delete Department", use_container_width=True, key="delete_dept", help="This will mark the department as inactive"):
						success, msg = delete_department(dept_id, cfg=st.session_state.config)
						if success:
							st.success("✅ Department deleted successfully!")
							st.rerun()
						else:
							st.error(f"❌ {msg}")


def render_datasets():
	st.title("Available Datasets")
	st.markdown("#### Dataset Overview")

	import os
	from pathlib import Path

	dataset_path = Path("datasets")
	if dataset_path.exists():
		dataset_folders = [f for f in dataset_path.iterdir() if f.is_dir()]

		if dataset_folders:
			st.markdown("### Dataset Folders")
			for folder in dataset_folders:
				with st.expander(f"📁 {folder.name}", expanded=False):
					# Count images in the folder
					image_count = 0
					total_size = 0

					for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
						images = list(folder.rglob(ext))
						image_count += len(images)
						for img in images:
							try:
								total_size += img.stat().st_size
							except:
								pass

					col1, col2, col3 = st.columns(3)
					with col1:
						st.metric("Images", image_count)
					with col2:
						st.metric("Size (MB)", f"{total_size / (1024*1024):.1f}")
					with col3:
						st.metric("Avg Size (KB)", f"{(total_size / max(1, image_count)) / 1024:.1f}" if image_count > 0 else "0")

					# Show sample images if available
					if image_count > 0:
						st.markdown("**Sample Images:**")
						sample_images = list(folder.rglob('*.jpg'))[:5] + list(folder.rglob('*.png'))[:5]
						sample_images = sample_images[:10]  # Limit to 10 samples

						cols = st.columns(min(5, len(sample_images)))
						for i, img_path in enumerate(sample_images):
							try:
								img = Image.open(img_path)
								img.thumbnail((200, 200))
								cols[i % 5].image(img, caption=img_path.name, use_container_width=True)
							except Exception as e:
								cols[i % 5].write(f"❌ {img_path.name}")
		else:
			st.info("No dataset folders found in datasets/ directory")
	else:
		st.error("datasets/ directory not found")

	st.markdown("---")
	st.markdown("#### Dataset Statistics")

	# Overall statistics
	total_images = 0
	total_size = 0
	dataset_stats = []

	for folder in dataset_path.iterdir() if dataset_path.exists() else []:
		if folder.is_dir():
			image_count = 0
			folder_size = 0

			for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
				images = list(folder.rglob(ext))
				image_count += len(images)
				for img in images:
					try:
						folder_size += img.stat().st_size
					except:
						pass

			total_images += image_count
			total_size += folder_size

			dataset_stats.append({
				"Dataset": folder.name,
				"Images": image_count,
				"Size (MB)": round(folder_size / (1024*1024), 1)
			})

	if dataset_stats:
		import pandas as pd
		stats_df = pd.DataFrame(dataset_stats)
		st.dataframe(stats_df)

		col1, col2, col3 = st.columns(3)
		with col1:
			st.metric("Total Datasets", len(dataset_stats))
		with col2:
			st.metric("Total Images", total_images)
		with col3:
			st.metric("Total Size (MB)", f"{total_size / (1024*1024):.1f}")


def render_reports_downloads():
	st.title("Reports & Downloads")
	st.markdown("#### Event Logs")
	events = list_events(limit=1000)
	df = pd.DataFrame(events)
	st.dataframe(df)
	if not df.empty:
		csv_bytes = df.to_csv(index=False).encode("utf-8")
		st.download_button(
			label="Download Events CSV",
			data=csv_bytes,
			file_name="events.csv",
			mime="text/csv",
			key="download_events_csv",
		)
	else:
		st.info("No events to export yet.")

	st.markdown("#### Dataset Snapshot")
	if st.button("Refresh dataset snapshot", key="refresh_dataset_snapshot") or st.session_state.get("dataset_df") is None:
		st.session_state.dataset_df = load_dataset()
	st.dataframe(st.session_state.dataset_df)
	if st.session_state.dataset_df is not None and not st.session_state.dataset_df.empty:
		ds_csv = st.session_state.dataset_df.to_csv(index=False).encode("utf-8")
		st.download_button(
			label="Download Dataset CSV",
			data=ds_csv,
			file_name="dataset.csv",
			mime="text/csv",
			key="download_dataset_csv",
		)
	cfg: AppConfig = st.session_state.config
	st.markdown("#### Model")
	try:
		if os.path.exists(cfg.model_path):
			with open(cfg.model_path, "rb") as f:
				st.download_button(
					label=f"Download Model ({os.path.basename(cfg.model_path)})",
					data=f.read(),
					file_name=os.path.basename(cfg.model_path),
					mime="application/octet-stream",
					key="download_model_file",
				)
		else:
			st.info("No saved model found. Train one in Dataset & Training.")
	except Exception as e:
		st.error(f"Unable to access model file: {e}")


def show_sidebar_navigation() -> None:
	"""Professional sidebar navigation with role-based menu"""
	st.sidebar.title("🏫 Navigation")
	
	user = st.session_state.get('user')
	current_page = get_current_page()
	
	# Show user info if logged in
	if user:
		st.sidebar.markdown("---")
		st.sidebar.markdown(f"**👤 {user.get('full_name', 'User')}**")
		st.sidebar.caption(f"Role: {user.get('role', 'N/A').title()}")
		st.sidebar.markdown("---")
	
	# Build navigation menu based on role
	if not user:
		# Guest menu - Simple 3-button navigation
		menu_items = [
			("🏠 Home", "home"),
			("🎓 Student", "student_portal"),
			("👨‍💼 Admin", "admin_login"),
		]
	elif is_admin():
		# Admin menu
		menu_items = [
			("🏠 Home", "home"),
			("🎓 Student Portal", "student_portal"),
			("📊 Admin Dashboard", "admin_dashboard"),
			("🎓 Verification", "verification"),
			("👤 Profile", "profile"),
		]
	elif is_student():
		# Student menu
		menu_items = [
			("🏠 Home", "home"),
			("🎓 Student Portal", "student_portal"),
			("📋 My Dashboard", "student_dashboard"),
			("👤 Profile", "profile"),
		]
	else:
		# Default menu
		menu_items = [
			("🏠 Home", "home"),
			("🎓 Student", "student_portal"),
			("👨‍💼 Admin", "admin_login"),
		]
	
	# Display navigation buttons
	for label, page_id in menu_items:
		button_type = "primary" if current_page == page_id else "secondary"
		if st.sidebar.button(label, key=f"nav_{page_id}", use_container_width=True, type=button_type):
			navigate_to(page_id)
	
	# Logout button for logged-in users
	if user:
		st.sidebar.markdown("---")
		if st.sidebar.button("🚪 Logout", use_container_width=True, type="secondary"):
			logout_and_redirect()
	
	# Settings for admin
	if is_admin():
		sidebar_settings()


def main():
	st.set_page_config(page_title="Attire & Safety Verification", layout="wide")
	ensure_dirs()
	init_session_state()
	init_db()

	# Handle legacy redirect flags (backward compatibility)
	if st.session_state.get('show_verification'):
		del st.session_state['show_verification']
		navigate_to('verification')
		return
	
	# Sync legacy 'page' state to new 'current_page'
	if 'page' in st.session_state and st.session_state.get('page') != get_current_page():
		st.session_state.current_page = st.session_state['page']
		del st.session_state['page']
	
	# Show sidebar navigation
	show_sidebar_navigation()
	
	# Import UI functions
	from src.ui.auth_ui import show_registration_form
	from src.ui.face_login_ui import show_face_authentication
	from src.ui.student_dashboard import show_student_dashboard
	
	# Get current page
	current_page = get_current_page()
	
	# Route to appropriate page
	if current_page == "home":
		render_home()
	
	elif current_page == "student_portal":
		render_student_portal()
	
	elif current_page == "face_auth":
		# Check for redirect flag from face authentication
		if st.session_state.get('redirect_to_verification'):
			del st.session_state['redirect_to_verification']
			navigate_to('verification')
			return
		
		user = show_face_authentication(st.session_state.config)
		if user:
			st.session_state['user'] = user
			# Redirect based on role after successful login
			if is_admin():
				navigate_to('admin_dashboard')
			elif is_student():
				navigate_to('verification')
			else:
				navigate_to('student_dashboard')
	
	elif current_page == "register":
		user = show_registration_form(st.session_state.config)
		# Registration redirects to face_auth automatically
	
	elif current_page == "student_dashboard":
		if not is_logged_in():
			st.warning("🔒 Please login first to access your dashboard")
			if st.button("🔐 Go to Login"):
				navigate_to("face_auth")
		else:
			show_student_dashboard(st.session_state.config)
	
	elif current_page == "verification":
		render_student_verification()
	
	elif current_page == "admin_login":
		st.title("👨‍💼 Admin Login")
		st.markdown("---")
		st.warning("🔒 This area requires administrator access.")
		st.info("Please enter admin credentials to continue.")

		with st.form("admin_login_form"):
			admin_username = st.text_input("Username")
			admin_password = st.text_input("Password", type="password")
			login_button = st.form_submit_button("Login as Admin", use_container_width=True, type="primary")

		if login_button:
			if admin_username == "admin" and admin_password == "admin123":
				st.session_state['user'] = {
					'username': 'admin',
					'role': 'admin',
					'full_name': 'System Administrator',
					'email': 'admin@system.com'
				}
				st.success("✅ Admin login successful!")
				navigate_to("admin_dashboard")
			else:
				st.error("❌ Invalid admin credentials")
	
	elif current_page == "admin_dashboard":
		if not is_admin():
			navigate_to("admin_login")
		else:
			render_admin_tab()
	
	elif current_page == "profile":
		if not is_logged_in():
			st.warning("🔒 Please login first to access your profile")
			if st.button("🔐 Go to Login"):
				navigate_to("face_auth")
		else:
			user = st.session_state.get('user')
			st.title("👤 My Profile")
			st.markdown("---")
			
			col1, col2 = st.columns(2)
			with col1:
				st.write(f"**Full Name:** {user.get('full_name', 'N/A')}")
				st.write(f"**Email:** {user.get('email', 'N/A')}")
				st.write(f"**Role:** {user.get('role', 'N/A').upper()}")
			
			with col2:
				if user.get('roll_no'):
					st.write(f"**Roll Number:** {user.get('roll_no', 'N/A')}")
				if user.get('department'):
					st.write(f"**Department:** {user.get('department', 'N/A')}")
				if user.get('class'):
					st.write(f"**Class:** {user.get('class', 'N/A')}")
			
			if user.get('auth_method') == 'face' and user.get('auth_time'):
				st.info(f"ℹ️ Last Face Authentication: {user.get('auth_time', 'N/A')}")
			
			st.markdown("---")

			# Historical accuracy chart and recent events
			student_identifier = user.get('roll_no') or user.get('student_id') or user.get('id') or None
			if student_identifier:
				from src.db import get_events_for_student
				events = get_events_for_student(student_identifier, limit=200, cfg=st.session_state.config)
				if events:
					import pandas as _pd
					df_events = _pd.DataFrame(events)
					if 'timestamp' in df_events.columns:
						try:
							df_events['timestamp'] = _pd.to_datetime(df_events['timestamp'])
						except Exception:
							pass
					if 'score' in df_events.columns and not df_events['score'].isnull().all():
						chart_df = df_events.set_index('timestamp')['score'].sort_index()
						with st.expander("📈 Accuracy History", expanded=False):
							st.line_chart(chart_df)
					with st.expander("🗂️ Recent Verification Events", expanded=False):
						st.dataframe(df_events[['timestamp','zone','status','score','label']].sort_values('timestamp', ascending=False))
						csv_bytes = df_events.to_csv(index=False).encode('utf-8')
						st.download_button("Download Events CSV", csv_bytes, file_name=f"{student_identifier}_events.csv")
	
	else:
		# Default to home
		render_home()


if __name__ == "__main__":
	main()
