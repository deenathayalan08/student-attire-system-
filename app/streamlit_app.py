import io
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

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
	init_db,
	insert_event,
	list_events,
	upsert_setting,
	get_setting,
	get_all_students,
	get_compliance_stats,
	add_student,
	update_student_verification,
	add_user,
	get_student,
	get_all_departments,
	generate_student_id,
	parse_student_id,
	delete_student,
)
from src.alerts import notify_non_compliance, get_id_card_status_message, get_detailed_id_card_message, notify_id_card_status
from src.security import check_and_alert_unauthorized_student, check_and_log_entry_time, check_and_log_exit_time, check_and_alert_emergency_violations
from src.biometric import get_biometric_system


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
	with st.sidebar.expander("Settings", expanded=False):
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
			except Exception as e:
				st.error(f"Failed to load model: {e}")

		if st.sidebar.button("Clear session state"):
			for k in list(st.session_state.keys()):
				if k not in ["config"]:
					del st.session_state[k]
				st.rerun()


def render_home():
	st.title("🏫 Student Attire & Safety Verification System")
	st.caption("Streamlit + MediaPipe + scikit-learn")

	st.markdown("---")
	st.markdown("### 👋 Welcome to the Student Verification Portal")

	st.markdown("#### Are you already registered?")
	if "home_auth_mode" not in st.session_state:
		st.session_state.home_auth_mode = "login"

	option = st.radio(
		"Select an option",
		("Already registered", "Need to register"),
		index=0 if st.session_state.home_auth_mode == "login" else 1,
		key="home_auth_mode_radio",
		horizontal=True,
	)

	st.session_state.home_auth_mode = "login" if option == "Already registered" else "register"

	if st.session_state.home_auth_mode == "login":
		render_home_login_section()
	else:
		render_home_registration_section()

	st.markdown("---")
	st.markdown("### 🔧 System Information")
	st.info("This system uses advanced AI to verify student attire compliance and ensure safety standards are met.")

	st.markdown("### 📊 Quick Stats")
	stats = get_compliance_stats(cfg=st.session_state.config)
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Total Students", stats["total_students"])
	with col2:
		st.metric("Verified Today", stats["verified_students"])
	with col3:
		st.metric("Compliance Rate", f"{stats['compliance_percentage']:.1f}%")
	with col4:
		st.metric("Events Today", stats["total_events"])


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


def render_home_login_section() -> None:
	st.subheader("🔐 Student Login & Biometric Verification")
	cfg: AppConfig = st.session_state.config

	default_id = st.session_state.get("prefill_student_id", "")
	student_id_input = st.text_input("Student ID / RFID", value=default_id, key="home_login_student_id")

	if not student_id_input:
		st.info("Enter your registered Student ID to continue.")
		return

	student_info = get_student(student_id_input, cfg)
	if not student_info:
		st.error("Student ID not found. Please double-check or register below.")
		return

	st.success(f"Welcome back, {student_info.get('name', 'Student')}!")
	col_a, col_b, col_c = st.columns(3)
	with col_a:
		st.write(f"**Department:** {student_info.get('department', 'N/A')}")
	with col_b:
		st.write(f"**Class:** {student_info.get('class', 'N/A')}")
	with col_c:
		st.write(f"**Gender:** {student_info.get('gender', 'N/A').title()}")

	bio_system = get_biometric_system(cfg)
	if not bio_system.is_biometric_registered(student_id_input):
		st.warning("Biometric authentication not registered for this student. Please contact an administrator.")
		return

	from src.phone_comm import get_phone_comm_system
	phone_comm = get_phone_comm_system(cfg)

	qr_state = st.session_state.get("home_qr_state")

	if st.button("Generate Unique Biometric QR Code", key="home_generate_biometric_qr"):
		try:
			token, qr_img = phone_comm.generate_biometric_qr(student_id_input)
			buf = io.BytesIO()
			qr_img.save(buf, format="PNG")
			st.session_state.home_qr_state = {
				"student_id": student_id_input,
				"image": buf.getvalue(),
				"generated_token": token,
				"generated_at": datetime.now().isoformat(),
			}
			st.session_state.home_biometric_verified = None
			st.success("QR code generated. Scan it with your phone to continue.")
		except Exception as exc:
			st.error(f"Unable to generate QR code: {exc}")

	qr_state = st.session_state.get("home_qr_state")
	if qr_state and qr_state.get("student_id") == student_id_input:
		st.markdown("##### 📱 Scan & Authenticate")
		st.image(qr_state["image"], caption="Scan this QR code on your phone", use_column_width=False)
		st.info(
			"Each student receives a unique QR code. Scan it on your phone, complete the fingerprint prompt, "
			"and enter the verification token displayed on your phone below."
		)

		token_input = st.text_input("Verification token from phone", key="home_token_input")
		if st.button("Verify Biometric Token", key="home_verify_token_btn") and token_input:
			verified_student = phone_comm.verify_token_and_get_student_id(token_input.strip())
			if verified_student is None:
				st.error("Invalid or expired token. Generate a new QR code and try again.")
			elif verified_student != student_id_input:
				st.error("Token does not match this student. Please regenerate the QR code.")
			else:
				st.session_state.home_biometric_verified = student_id_input
				st.success("Biometric verification successful! You can proceed to attire verification.")
				st.session_state.login_student_id = student_id_input
				st.session_state.nav_selection = "Student Verification"
				st.rerun()
	else:
		st.info("Click **Generate Unique Biometric QR Code** to begin the verification process.")


def render_home_registration_section() -> None:
	st.subheader("🆕 Student Registration")
	cfg: AppConfig = st.session_state.config
	departments = get_all_departments(cfg)

	if not departments:
		st.info("No departments have been configured yet. Please ask an administrator to add departments first.")
		return

	with st.form("home_student_registration_form"):
		st.markdown("#### 📋 Student Information")
		col1, col2 = st.columns(2)
		with col1:
			name = st.text_input("Full Name *", key="home_reg_name")
			gender = st.selectbox("Gender *", ["male", "female"], index=0, key="home_reg_gender")
		with col2:
			email = st.text_input("Email", key="home_reg_email")
			phone = st.text_input("Phone", key="home_reg_phone")

		st.markdown("#### 🎓 Academic Information")
		col3, col4, col5, col6 = st.columns(4)
		with col3:
			batch_year = st.text_input("Batch Year (e.g., 2025)", key="home_reg_batch_year")
		with col4:
			dept_options = [f"{d['department_id']} - {d['name']}" for d in departments]
			selected_dept = st.selectbox("Department *", dept_options, key="home_reg_department")
			dept_id = selected_dept.split(" - ")[0] if selected_dept else ""
		with col5:
			class_section = st.text_input("Class Section (A, B, ...)", key="home_reg_class_section")
		with col6:
			student_number = st.text_input("Student Number (2 digits)", key="home_reg_student_number", max_chars=2)

		if st.form_submit_button("Generate Student ID", type="secondary"):
			if batch_year and dept_id and class_section and student_number:
				generated_id = generate_student_id(batch_year, dept_id, class_section, student_number, cfg=cfg)
				st.session_state.generated_student_id = generated_id
				st.success(f"Generated Student ID: {generated_id}")
			else:
				st.error("Please fill batch year, department, class section, and student number to generate Student ID.")

		if st.session_state.get("generated_student_id"):
			student_id = st.text_input(
				"Student ID *",
				value=st.session_state.generated_student_id,
				key="home_reg_student_id",
			)
		else:
			student_id = st.text_input("Student ID *", key="home_reg_student_id_manual")

		st.markdown("---")
		st.markdown("**Biometric Registration**")
		biometric_type = st.selectbox("Biometric Type", ["fingerprint", "face"], index=0, key="home_reg_biometric")

		if st.form_submit_button("Complete Registration", type="primary"):
			if not (student_id and name and gender):
				st.error("Name, gender, and student ID are required.")
			else:
				parsed = parse_student_id(student_id, cfg=cfg)
				class_name = f"{parsed.get('batch_year', '')} - {parsed.get('class_section', '')}" if parsed else ""
				department_name = next((d['name'] for d in departments if d['department_id'] == parsed.get('department_id')), "") if parsed else ""

				add_student({
					"id": student_id,
					"name": name,
					"class": class_name or student_id,
					"department": department_name or selected_dept,
					"gender": gender,
					"email": email,
					"phone": phone,
					"contact_info": "",
				}, cfg=cfg)

				bio_system = get_biometric_system(cfg)
				bio_success = bio_system.register_biometric(student_id, biometric_type)

				st.success(f"Registration completed successfully for {name}!")
				if bio_success:
					st.info("Biometric data registered. You can now log in.")
				else:
					st.warning("Student registered, but biometric registration failed. Please retry from admin panel.")

				st.session_state.prefill_student_id = student_id
				st.session_state.home_auth_mode = "login"
				if "generated_student_id" in st.session_state:
					del st.session_state["generated_student_id"]
				st.rerun()
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
	student_id = st.text_input("Student ID / RFID (optional)", key="student_id_image")
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


def render_webcam_tab():
	st.subheader("Webcam")
	student_id = st.text_input("Student ID / RFID (optional)", key="student_id_webcam")
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


def render_video_tab():
	st.subheader("Video")
	student_id = st.text_input("Student ID / RFID (optional)", key="student_id_video")
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




def render_phone_verification():
	st.title("Phone-PC Biometric Verification")
	st.markdown("### Step 1: Phone Authentication")
	st.markdown("Use your mobile phone to authenticate via fingerprint, then enter the verification token below.")

	# Phone communication setup
	from src.phone_comm import get_phone_comm_system
	phone_comm = get_phone_comm_system(st.session_state.config)

	# Student ID input for QR generation
	student_id_input = st.text_input("Enter Student ID to generate QR code", key="student_id_qr")

	if student_id_input:
		# Check if student exists and has biometric registered
		from src.db import get_student
		from src.biometric import get_biometric_system

		student_info = get_student(student_id_input, st.session_state.config)
		bio_system = get_biometric_system(st.session_state.config)

		if not student_info:
			st.error("❌ Student not found in database")
		elif not bio_system.is_biometric_registered(student_id_input):
			st.error("❌ Biometric not registered for this student")

			# Guide to biometric registration
			st.warning("🔐 **Biometric Registration Required**")
			st.info("You need to register your biometric data before proceeding with verification.")

			st.markdown("### 📱 Biometric Registration Process")

			with st.expander("📋 How to Register Biometric Data", expanded=True):
				st.write("**Option 1: Register via Admin Dashboard**")
				st.write("1. Go to **Admin Dashboard** → **Add Student** tab")
				st.write("2. Find your student record or add yourself if not present")
				st.write("3. Check **'Register biometric data now'** checkbox")
				st.write("4. Click **'Register Biometric'** button")
				st.write("5. Follow the fingerprint registration prompts")
				st.write("6. Return here after registration to continue verification")

				st.write("")
				st.write("**Option 2: Contact Administrator**")
				st.write("• Ask a system administrator to register your biometric data")
				st.write("• Provide them with your Student ID: **" + student_id_input + "**")

			# Quick registration option (if admin access)
			st.markdown("---")
			st.markdown("### 🔧 Quick Biometric Registration")

			if st.checkbox("I have administrator access - register biometric now", key="admin_register"):
				st.warning("⚠️ **Administrator Access Required**")
				st.write("Only system administrators should use this option.")

				col1, col2 = st.columns(2)
				with col1:
					admin_username = st.text_input("Admin Username", key="admin_user")
					admin_password = st.text_input("Admin Password", type="password", key="admin_pass")

				with col2:
					confirm_student_id = st.text_input("Confirm Student ID", value=student_id_input, key="confirm_student")
					biometric_type = st.selectbox("Biometric Type", ["fingerprint", "face"], index=0, key="bio_type")

				if st.button("🔐 Register Biometric (Admin Only)", key="register_bio_admin"):
					if admin_username and admin_password and confirm_student_id == student_id_input:
						# Verify admin credentials
						from src.db import get_user
						admin_user = get_user(admin_username, st.session_state.config)

						if admin_user and admin_user.get('password') == admin_password and admin_user.get('role') == 'admin':
							# Register biometric
							success = bio_system.register_biometric(student_id_input, biometric_type)
							if success:
								st.success(f"✅ Biometric ({biometric_type}) registered successfully for student {student_id_input}!")
								st.balloons()
								st.rerun()  # Refresh to show updated status
							else:
								st.error("❌ Failed to register biometric data")
						else:
							st.error("❌ Invalid administrator credentials")
					else:
						st.error("❌ Please fill all fields and ensure Student ID matches")

			st.markdown("---")
			st.info("💡 **After biometric registration, return to this page and enter your Student ID again to continue with verification.**")
		else:
			st.success(f"✅ **Student Found:** {student_info.get('name', 'Unknown')}")

			# Generate QR code
			if st.button("📱 Generate QR Code for Phone Authentication", key="generate_qr"):
				qr_img = phone_comm.generate_qr_code(student_id_input)

				if qr_img:
					st.image(qr_img, caption="Scan this QR code with your phone", width=300)
					st.info("📱 **Instructions:**")
					st.write("1. Open your phone's camera or QR scanner app")
					st.write("2. Scan the QR code above")
					st.write("3. Your phone will perform fingerprint authentication")
					st.write("4. Enter the verification token below")

					# Store the student ID for later use
					st.session_state.current_student_id = student_id_input
				else:
					st.error("Failed to generate QR code")

	# Token input (only show if we have a student ID)
	if hasattr(st.session_state, 'current_student_id') and st.session_state.current_student_id:
		st.markdown("---")
		st.markdown("### Step 2: Enter Verification Token")
		token = st.text_input("Enter verification token from phone", key="phone_token")

		if token:
			# Verify token and get student ID
			verified_student_id = phone_comm.verify_token_and_get_student_id(token)

			if verified_student_id and verified_student_id == st.session_state.current_student_id:
				st.success(f"✅ **Biometric Verified!** Student ID: {verified_student_id}")

				# Get student info
				student_info = get_student(verified_student_id, st.session_state.config)

				if student_info:
					st.info(f"👤 **Student:** {student_info.get('name', 'Unknown')} | **Class:** {student_info.get('class', 'Unknown')}")

					# Step 3: Attire Verification
					st.markdown("---")
					st.markdown("### Step 3: Attire Verification")
					st.markdown("Upload a photo for attire analysis:")

					zone = st.selectbox("Verification Zone", st.session_state.config.zones, index=0, key="zone_phone")
					upload = st.file_uploader("Upload student photo", type=["jpg", "jpeg", "png"], key="upload_phone")

					if upload is not None:
						img = Image.open(upload).convert("RGB")
						bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

						# Perform full attire verification
						result = verify_attire_and_safety(bgr, verified_student_id, zone, st.session_state.config, st.session_state.classifier)

						col1, col2 = st.columns(2)
						with col1:
							st.image(img, caption="Student Photo")

						with col2:
							# Create annotated image
							pose = extract_pose(bgr)
							annotated = draw_pose_annotations(bgr.copy(), pose)

							# Add violation indicators
							violations = result.get("violations", {}).get("details", [])
							annotated = draw_violation_indicators(annotated, pose, violations)
							annotated = overlay_detailed_badge(annotated, result)

							st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Analysis Result")

						# Display comprehensive results
						st.markdown("---")
						st.markdown("### 📊 Verification Results")

						# Status overview
						status_col, score_col, policy_col = st.columns(3)
						with status_col:
							status_color = {"PASS": "🟢", "WARNING": "🟡", "FAIL": "🔴"}.get(result["status"], "⚪")
							st.metric("Status", f"{status_color} {result['status']}")

						with score_col:
							st.metric("Compliance Score", f"{result['overall_score']:.1%}")

						with policy_col:
							st.metric("Policy", result.get("day_policy", "Unknown").title())

						# Student details
						st.markdown("#### 👤 Student Information")
						info_col1, info_col2, info_col3 = st.columns(3)
						with info_col1:
							st.write(f"**ID:** {result['student_id']}")
							st.write(f"**Name:** {result['student_name']}")
						with info_col2:
							st.write(f"**Gender:** {result['gender'].title()}")
							st.write(f"**Class:** {result['class']}")
						with info_col3:
							st.write(f"**Department:** {result['department']}")
							st.write(f"**Zone:** {result['zone']}")

						# Violations summary
						st.markdown("#### 🚨 Violations Summary")
						violations_data = result["violations"]
						vio_col1, vio_col2, vio_col3, vio_col4 = st.columns(4)
						with vio_col1:
							st.metric("Total Violations", violations_data["total_violations"])
						with vio_col2:
							st.metric("Critical", violations_data["critical"])
						with vio_col3:
							st.metric("High", violations_data["high"])
						with vio_col4:
							st.metric("Medium", violations_data["medium"])

						# Detailed violations
						if violations_data["total_violations"] > 0:
							st.error("❌ **Dress Code Violations Detected**")

							for i, violation in enumerate(violations_data["details"], 1):
								severity = violation.get("severity", "medium")
								severity_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵"}.get(severity, "⚪")

								with st.expander(f"{severity_emoji} Violation {i}: {violation['item']}", expanded=True):
									col1, col2 = st.columns(2)
									with col1:
										st.write(f"**Detected:** {violation['detected']}")
										st.write(f"**Score:** {violation['score']:.1%}")
									with col2:
										st.write(f"**Severity:** {severity.upper()}")
						else:
							st.success("✅ **No Dress Code Violations**")

						# Technical details
						st.markdown("#### 🔧 Technical Details")
						tech_col1, tech_col2, tech_col3 = st.columns(3)
						with tech_col1:
							st.write(f"**Biometric:** {'✅ Verified' if result['biometric_verified'] else '❌ Failed'}")
						with tech_col2:
							st.write(f"**ID Card:** {'✅ Detected' if result['id_card_detected'] else '❌ Not Found'}")
						with tech_col3:
							st.write(f"**Pose:** {'✅ Detected' if result['pose_detected'] else '❌ Not Detected'}")

						# Security checks
						st.markdown("#### 🔒 Security Checks")
						security = result["security_checks"]
						sec_col1, sec_col2, sec_col3 = st.columns(3)
						with sec_col1:
							auth_status = "✅ Authorized" if security.get("authorized", {}).get("status") == "OK" else "❌ Unauthorized"
							st.write(f"**Access:** {auth_status}")
						with sec_col2:
							time_status = "✅ OK" if security.get("time_check", {}).get("status") == "OK" else "⚠️ Check Required"
							st.write(f"**Timing:** {time_status}")
						with sec_col3:
							emergency_status = "✅ OK" if security.get("emergency_check", {}).get("status") == "OK" else "🚨 Alert"
							st.write(f"**Emergency:** {emergency_status}")

						# Recommendations
						st.markdown("#### 💡 Recommendations")
						for rec in result.get("recommendations", []):
							st.info(f"• {rec}")

						# Event logging
						st.caption(f"📝 Event logged with ID: {result['event_id']}")

						# Generate report button
						if st.button("📄 Generate Official Report", key="generate_report"):
							from src.report_generator import generate_html_report, save_report
							
							# Generate and save report
							html_report = generate_html_report(result, cfg=st.session_state.config)
							report_path = save_report(result, format_type="html", cfg=st.session_state.config)
							
							if report_path:
								st.success(f"✅ Report generated successfully!")
								
								# Display report
								with st.expander("📋 View Report", expanded=True):
									st.markdown(html_report, unsafe_allow_html=True)
								
								# Download buttons
								col1, col2, col3 = st.columns(3)
								
								with col1:
									st.download_button(
										label="📥 Download as HTML",
										data=html_report,
										file_name=f"report_{verified_student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
										mime="text/html",
										key="download_html"
									)
								
								with col2:
									from src.report_generator import generate_text_report
									text_report = generate_text_report(result, cfg=st.session_state.config)
									st.download_button(
										label="📥 Download as TXT",
										data=text_report,
										file_name=f"report_{verified_student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
										mime="text/plain",
										key="download_txt"
									)
								
								with col3:
									import json
									json_report = json.dumps(result, indent=2, default=str)
									st.download_button(
										label="📥 Download as JSON",
										data=json_report,
										file_name=f"report_{verified_student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
										mime="application/json",
										key="download_json"
									)

					else:
						st.info("📷 Please upload a student photo to continue with attire verification")
				else:
					st.error("❌ Student information not found in database")
			else:
				st.error("❌ **Invalid or expired verification token**")
				st.info("Please authenticate again on your phone and enter a new token")


def render_student_registration():
	st.title("📝 Student Registration")
	st.markdown("### Register as a New Student")

	st.info("**Welcome!** Please fill out the registration form below. Your biometric data will be registered automatically during the process.")

	# Get departments for dropdown
	from src.db import get_all_departments
	departments = get_all_departments(st.session_state.config)

	with st.form("student_registration_form"):
		st.markdown("#### 📋 Student Information")

		col1, col2 = st.columns(2)
		with col1:
			name = st.text_input("Full Name *")
			gender = st.selectbox("Gender *", ["male", "female"], index=0)

		with col2:
			email = st.text_input("Email")
			phone = st.text_input("Phone")

		st.markdown("#### 🎓 Academic Information")

		col1, col2, col3, col4 = st.columns(4)
		with col1:
			batch_year = st.text_input("Batch Year (2-digit)", placeholder="25", max_chars=2)
		with col2:
			dept_options = [f"{d['short_form']} ({d['name']})" for d in departments]
			selected_dept = st.selectbox("Department", dept_options, index=0 if departments else None)
			dept_id = next((d['department_id'] for d in departments if f"{d['short_form']} ({d['name']})" == selected_dept), None)
		with col3:
			class_section = st.text_input("Class Section (A, B, C, etc.)", placeholder="A")
		with col4:
			student_number = st.text_input("Student Number (2-digit)", placeholder="01", max_chars=2)

		# Generate Student ID button
		if st.form_submit_button("Generate Student ID", width='stretch'):
			if batch_year and dept_id and class_section and student_number:
				from src.db import generate_student_id
				generated_id = generate_student_id(batch_year, dept_id, class_section, student_number, cfg=st.session_state.config)
				st.session_state.generated_student_id = generated_id
				st.success(f"Generated Student ID: {generated_id}")
				st.rerun()
			else:
				st.error("Please fill all academic information fields to generate Student ID")

		# Show generated or manual Student ID input
		if hasattr(st.session_state, 'generated_student_id') and st.session_state.generated_student_id:
			student_id = st.text_input("Student ID *", value=st.session_state.generated_student_id, disabled=True)
		else:
			student_id = st.text_input("Student ID *", placeholder="Enter manually or generate above")

		# Auto-fill class and department based on generated ID
		if hasattr(st.session_state, 'generated_student_id') and st.session_state.generated_student_id:
			from src.db import parse_student_id
			parsed = parse_student_id(st.session_state.generated_student_id)
			if parsed:
				auto_class = f"{parsed['batch_year']} - {parsed['class_section']}"
				auto_dept = next((d['name'] for d in departments if d['department_id'] == parsed['department_id']), "")
				st.text_input("Auto-filled Class", value=auto_class, disabled=True)
				st.text_input("Auto-filled Department", value=auto_dept, disabled=True)

		# Biometric registration section
		st.markdown("---")
		st.markdown("**Biometric Registration**")
		st.markdown("Biometric data will be registered automatically when the student is added.")

		biometric_type = st.selectbox("Biometric Type", ["fingerprint", "face"], index=0, key="biometric_type")
		st.info("💡 **Note:** Biometric registration is required for student verification. Make sure you are ready for biometric capture.")

		if st.form_submit_button("Complete Registration"):
			# Parse student ID to get class and department info
			from src.db import parse_student_id
			parsed_id = parse_student_id(student_id)

			if parsed_id:
				class_name = f"{parsed_id['batch_year']} - {parsed_id['class_section']}"
				department_name = next((d['name'] for d in departments if d['department_id'] == parsed_id['department_id']), "")
			else:
				class_name = ""
				department_name = ""

			if student_id and name and gender:
				# Add student to database
				add_student({
					"id": student_id,
					"name": name,
					"class": class_name,
					"department": department_name,
					"gender": gender,
					"email": email,
					"phone": phone,
					"contact_info": ""
				}, cfg=st.session_state.config)

				# Register biometric data automatically
				from src.biometric import get_biometric_system
				bio_system = get_biometric_system(st.session_state.config)
				biometric_success = bio_system.register_biometric(student_id, biometric_type)

				if biometric_success:
					st.success(f"✅ Registration completed successfully! Welcome {name}!")
					st.info(f"**Your Student ID:** {student_id}")
					st.info("You can now use the Student Login to access verification features.")
				else:
					st.warning(f"⚠️ Student registered, but biometric registration failed. Please contact administrator.")

				# Clear generated ID after successful submission
				if hasattr(st.session_state, 'generated_student_id'):
					del st.session_state.generated_student_id
			else:
				st.error("Please fill required fields (*) and ensure Student ID is valid")

	st.markdown("---")
	st.markdown("#### 🔙 Navigation")
	col1, col2 = st.columns(2)
	with col1:
		if st.button("🏠 Back to Home", width='stretch'):
			st.session_state.nav_selection = "Home"
			st.rerun()
	with col2:
		if st.button("🔐 Go to Student Login", width='stretch'):
			st.session_state.nav_selection = "Student Verification"
			st.rerun()


def render_student_verification():
	st.title("🔐 Student Login & Verification")
	st.markdown("### Welcome back! Please verify your identity")

	# Student ID input for verification
	student_id_input = st.text_input("Enter your Student ID", key="login_student_id", placeholder="e.g., 2301208")

	if student_id_input:
		# Check if student exists
		from src.db import get_student
		from src.biometric import get_biometric_system

		student_info = get_student(student_id_input, st.session_state.config)
		bio_system = get_biometric_system(st.session_state.config)

		if not student_info:
			st.error("❌ Student ID not found in the system")
			st.info("Please check your Student ID or contact administrator if you haven't registered yet")
		else:
			st.success(f"✅ Welcome back, {student_info.get('name', 'Student')}!")
			st.info(f"**Department:** {student_info.get('department', 'N/A')} | **Class:** {student_info.get('class', 'N/A')}")

			# Check biometric registration
			if not bio_system.is_biometric_registered(student_id_input):
				st.warning("⚠️ Biometric authentication not registered")
				st.info("Please contact your administrator to register biometric data before proceeding")
			else:
				st.success("✅ Biometric authentication is registered")

				# Proceed with verification tabs
				st.markdown("---")
				st.markdown("### 📸 Choose Verification Method")
				tabs = st.tabs(["Phone-PC Verification", "Direct Image", "Webcam", "Video"])
				with tabs[0]:
					render_phone_verification()
				with tabs[1]:
					render_image_tab()
				with tabs[2]:
					render_webcam_tab()
				with tabs[3]:
					render_video_tab()
	else:
		st.info("👆 Please enter your Student ID to continue")


def render_admin_login():
	st.title("🔐 Admin Login")
	st.markdown("### Administrator Access Required")

	st.markdown("---")

	# Admin login form
	with st.form("admin_login_form"):
		st.markdown("#### Enter Admin Credentials")

		username = st.text_input("Username", placeholder="admin")
		password = st.text_input("Password", type="password", placeholder="••••••••")

		# Login button
		if st.form_submit_button("Login as Administrator", width='stretch', type="primary"):
			if username == st.session_state.admin_username and password == st.session_state.admin_password:
				st.session_state.admin_authenticated = True
				st.success("✅ Admin login successful!")
				st.rerun()
			else:
				st.error("❌ Invalid username or password")

	st.markdown("---")
	st.markdown("#### 🔙 Navigation")
	if st.button("🏠 Back to Home", width='stretch'):
		st.session_state.nav_selection = "Home"
		st.rerun()


def render_admin_tab():
	import pandas as pd

	# Check if admin is authenticated
	if not st.session_state.get('admin_authenticated', False):
		render_admin_login()
		return

	st.title("👨‍💼 Admin Dashboard")

	# Logout button and Settings
	col1, col2, col3 = st.columns([2, 1, 1])
	with col2:
		if st.button("⚙️ Settings", width='stretch'):
			st.session_state.show_admin_settings = not st.session_state.get('show_admin_settings', False)
			st.rerun()
	with col3:
		if st.button("🚪 Logout", width='stretch'):
			st.session_state.admin_authenticated = False
			st.session_state.nav_selection = "Home"
			st.rerun()

	# Admin Settings Section
	if st.session_state.get('show_admin_settings', False):
		st.markdown("---")
		st.markdown("### 🔧 Admin Settings")

		with st.form("admin_settings_form"):
			st.markdown("#### Change Admin Credentials")

			col1, col2 = st.columns(2)
			with col1:
				new_username = st.text_input("New Username", value=st.session_state.admin_username)
			with col2:
				new_password = st.text_input("New Password", type="password", value=st.session_state.admin_password)

			st.markdown("**Security Note:** Changes take effect immediately and apply to the entire system.")

			if st.form_submit_button("Update Credentials", width='stretch', type="primary"):
				if new_username and new_password:
					st.session_state.admin_username = new_username
					st.session_state.admin_password = new_password
					st.success("✅ Admin credentials updated successfully!")
					st.info(f"**New Username:** {new_username}")
					# Hide settings after successful update
					st.session_state.show_admin_settings = False
					st.rerun()
				else:
					st.error("❌ Both username and password are required")

		st.markdown("---")

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
	tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Students", "Compliance Reports", "Add Student", "Add User", "Departments & Classes", "Policy Settings"])
	
	with tab1:
		students = get_all_students(cfg=st.session_state.config)
		st.subheader("All Students")

		# Search functionality
		search_term = st.text_input("🔍 Search by Student ID", key="search_student_id", placeholder="Enter student ID to search...")

		# Filter students based on search term
		if search_term:
			filtered_students = [s for s in students if search_term.lower() in s.get('id', '').lower()]
			st.info(f"Found {len(filtered_students)} student(s) matching '{search_term}'")
		else:
			filtered_students = students

		if filtered_students:
			# Re-order columns to match registration fields
			df = pd.DataFrame(filtered_students)
			desired_cols = [
				"id",
				"name",
				"gender",
				"class",
				"department",
				"email",
				"phone",
				"verified",
				"uniform_type",
			]
			cols = [c for c in desired_cols if c in df.columns] + [c for c in df.columns if c not in desired_cols]
			df = df[cols]
			st.dataframe(df, width='stretch')

			# Verification status for filtered results
			verified_count = sum(1 for s in filtered_students if s.get('verified', 0))
			st.info(f"📊 {verified_count}/{len(filtered_students)} students verified")

			# Delete student controls
			st.markdown("---")
			st.markdown("### 🗑️ Delete Student")
			delete_ids = [s["id"] for s in filtered_students if s.get("id")]
			if delete_ids:
				col_del1, col_del2 = st.columns([2, 1])
				with col_del1:
					selected_delete_id = st.selectbox(
						"Select Student ID to delete",
						delete_ids,
						key="admin_delete_student_id",
					)
				with col_del2:
					if st.button("Delete Selected Student", key="admin_delete_student_btn"):
						delete_student(selected_delete_id, cfg=st.session_state.config)
						st.success(f"Student {selected_delete_id} deleted successfully.")
						st.rerun()
		else:
			if search_term:
				st.warning(f"No students found with ID containing '{search_term}'")
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

		# Student ID Format Information
		st.info("**Student ID Format:** YYDDCSNN")
		st.write("• **YY**: Last 2 digits of batch year (e.g., 23 for 2023)")
		st.write("• **DD**: Department ID (auto-generated when department is created)")
		st.write("• **C**: Class section (1=A, 2=B, 3=C, etc.)")
		st.write("• **NN**: Student number (2-digit, e.g., 08)")
		st.write("**Example:** 2301208 = 2023 batch, Dept 01, Class A, Student 08")

		# Get available departments for dropdown
		from src.db import get_all_departments
		departments = get_all_departments(cfg=st.session_state.config)
		dept_options = [""] + [f"{d['department_id']} - {d['name']}" for d in departments]

		with st.form("add_student_form"):
			col1, col2 = st.columns(2)
			with col1:
				# Student ID generation section
				st.markdown("**Student ID Generation**")
				batch_year = st.text_input("Batch Year (e.g., 2023)", placeholder="2023")
				selected_dept = st.selectbox("Department", dept_options, help="Select department to auto-fill Department ID")

				# Extract department ID from selection
				dept_id = ""
				if selected_dept and selected_dept != "":
					dept_id = selected_dept.split(" - ")[0]

				class_section = st.text_input("Class Section (A, B, C, etc.)", placeholder="A")
				student_number = st.text_input("Student Number (2-digit)", placeholder="08", max_chars=2)

				# Generate Student ID button
				if st.form_submit_button("Generate Student ID", width='stretch'):
					if batch_year and dept_id and class_section and student_number:
						from src.db import generate_student_id
						generated_id = generate_student_id(batch_year, dept_id, class_section, student_number, cfg=st.session_state.config)
						st.session_state.generated_student_id = generated_id
						st.success(f"Generated Student ID: **{generated_id}**")
						st.rerun()
					else:
						st.error("Please fill all fields to generate Student ID")

				# Display generated or manual Student ID
				if hasattr(st.session_state, 'generated_student_id'):
					student_id = st.text_input("Student ID *", value=st.session_state.generated_student_id)
				else:
					student_id = st.text_input("Student ID *", placeholder="2301208")

				name = st.text_input("Full Name *")
				gender = st.selectbox("Gender *", ["male", "female"], index=0)

			with col2:
				email = st.text_input("Email")
				phone = st.text_input("Phone")

				# Auto-fill class and department based on generated ID
				if hasattr(st.session_state, 'generated_student_id') and st.session_state.generated_student_id:
					from src.db import parse_student_id
					parsed = parse_student_id(st.session_state.generated_student_id)
					if parsed:
						auto_class = f"{parsed['batch_year']} - {parsed['class_section']}"
						auto_dept = next((d['name'] for d in departments if d['department_id'] == parsed['department_id']), "")
						st.text_input("Auto-filled Class", value=auto_class, disabled=True)
						st.text_input("Auto-filled Department", value=auto_dept, disabled=True)

			# Biometric registration section
			st.markdown("---")
			st.markdown("**Biometric Registration**")
			st.markdown("Biometric data will be registered automatically when the student is added.")

			biometric_type = st.selectbox("Biometric Type", ["fingerprint", "face"], index=0, key="biometric_type")
			st.info("💡 **Note:** Biometric registration is required for student verification. Make sure the student is present for registration.")

			if st.form_submit_button("Add/Update Student"):
				# Parse student ID to get class and department info
				from src.db import parse_student_id
				parsed_id = parse_student_id(student_id)

				if parsed_id:
					class_name = f"{parsed_id['batch_year']} - {parsed_id['class_section']}"
					department_name = next((d['name'] for d in departments if d['department_id'] == parsed_id['department_id']), "")
				else:
					class_name = ""
					department_name = ""

				if student_id and name and gender:
					# Add student to database
					add_student({
						"id": student_id,
						"name": name,
						"class": class_name,
						"department": department_name,
						"gender": gender,
						"email": email,
						"phone": phone,
						"contact_info": ""
					}, cfg=st.session_state.config)

					# Register biometric data automatically
					from src.biometric import get_biometric_system
					bio_system = get_biometric_system(st.session_state.config)
					biometric_success = bio_system.register_biometric(student_id, biometric_type)

					if biometric_success:
						st.success(f"✅ Student {name} added successfully with {biometric_type} biometric registration!")
					else:
						st.warning(f"⚠️ Student {name} added, but biometric registration failed. Please try again.")

					# Clear generated ID after successful submission
					if hasattr(st.session_state, 'generated_student_id'):
						del st.session_state.generated_student_id
				else:
					st.error("Please fill required fields (*) and ensure Student ID is valid")
	
	with tab4:
		st.subheader("Add User (Role-Based Access)")
		with st.form("add_user_form"):
			username = st.text_input("Username *")
			password = st.text_input("Password *", type="password")
			role = st.selectbox("Role *", ["admin", "teacher", "security_staff"])
			full_name = st.text_input("Full Name *")
			email = st.text_input("Email *")
			assigned_class = st.text_input("Assigned Class (for teachers)")
			
			if st.form_submit_button("Add User"):
				if username and password and role and full_name and email:
					add_user(username, password, role, full_name, email, assigned_class, cfg=st.session_state.config)
					st.success("User added successfully!")
				else:
					st.error("Please fill all required fields (*)")

	
	with tab5:
		st.subheader("🏫 Departments & Classes Management")

		st.markdown("### ➕ Add New Department")

		with st.form("add_department_form"):
			col1, col2 = st.columns(2)
			with col1:
				dept_name = st.text_input("Department Name *", placeholder="e.g., Computer Science")
				short_form = st.text_input("Short Form *", placeholder="e.g., CS")
			with col2:
				num_classes = st.number_input("Number of Classes", min_value=1, max_value=10, value=3)

			st.markdown("**Class Sections:**")
			classes = [chr(65 + i) for i in range(num_classes)]  # A, B, C, etc.
			st.write(f"Generated classes: {', '.join(classes)}")

			if st.form_submit_button("➕ Add Department"):
				if dept_name and short_form:
					from src.db import add_department
					try:
						dept_id = add_department(dept_name, short_form, num_classes, cfg=st.session_state.config)
						st.success(f"✅ Department '{dept_name}' ({short_form}) added successfully!")
						st.info(f"Classes created: {', '.join(classes)}")
						st.rerun()  # Refresh to show updated list
					except Exception as e:
						st.error(f"Failed to add department: {e}")
				else:
					st.error("Please fill all required fields (*)")

		st.markdown("---")
		st.markdown("### 📋 Existing Departments")

		from src.db import get_all_departments
		existing_departments = get_all_departments(cfg=st.session_state.config)

		if existing_departments:
			# Process departments to show classes
			for dept in existing_departments:
				# Generate classes based on num_classes (A, B, C, etc.)
				classes = [chr(65 + i) for i in range(dept['num_classes'])]  # A, B, C, etc.
				dept['classes'] = classes

			# Display existing departments in a table
			dept_df = pd.DataFrame(existing_departments)
			st.dataframe(dept_df, width='stretch')

			# Department management actions
			st.markdown("### ⚙️ Department Actions")
			selected_dept = st.selectbox("Select Department to Manage", [d["name"] for d in existing_departments])

			col1, col2, col3 = st.columns(3)
			with col1:
				if st.button("✏️ Edit Department"):
					st.info(f"Edit functionality for {selected_dept} would be implemented here")
			with col2:
				if st.button("🗑️ Delete Department"):
					from src.db import delete_department
					selected_dept_data = next((d for d in existing_departments if d["name"] == selected_dept), None)
					if selected_dept_data:
						try:
							delete_department(selected_dept_data["id"], cfg=st.session_state.config)
							st.success(f"✅ Department '{selected_dept}' deleted successfully!")
							st.rerun()
						except Exception as e:
							st.error(f"Failed to delete department: {e}")
			with col3:
				if st.button("👥 View Students"):
					st.info(f"View students in {selected_dept} would be implemented here")
		else:
			st.info("No departments added yet. Use the form above to add your first department.")

		st.markdown("---")
		st.markdown("### 📊 Department Statistics")

		if existing_departments:
			total_depts = len(existing_departments)
			total_students = sum(d.get("students", 0) for d in existing_departments)
			total_classes = sum(len(d.get("classes", [])) for d in existing_departments)

			col1, col2, col3, col4 = st.columns(4)
			with col1:
				st.metric("Total Departments", total_depts)
			with col2:
				st.metric("Total Classes", total_classes)
			with col3:
				st.metric("Total Students", total_students)
			with col4:
				avg_students = total_students / total_depts if total_depts > 0 else 0
				st.metric("Avg Students/Dept", f"{avg_students:.1f}")

		st.markdown("---")
		st.markdown("### 🎓 Student Registration Guidelines")

		st.info("**For Student Registration:**")
		st.write("• **Department Selection:** Choose from the departments you add above")
		st.write("• **Class Selection:** Classes are automatically generated when adding departments")
		st.write("• **Student ID Format:** Typically includes department shortform (e.g., CS001, IT045, ME123)")
		st.write("• **Registration Process:** Use the 'Add Student' tab to register new students")

	with tab6:
		st.subheader("⚙️ Policy Settings")
		st.markdown("#### Days & Policy Configuration")
		
		cfg = st.session_state.config
		
		# Load current settings from DB
		from src.db import get_policy_settings
		db_settings = get_policy_settings(cfg)
		
		# Form for editing policy
		with st.form("policy_settings_form"):
			col1, col2 = st.columns(2)
			
			with col1:
				uniform_day = st.selectbox(
					"Uniform Day",
					["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
					index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(
						db_settings.get("policy_uniform_day", "Wednesday").title()
					)
				)
				
				entry_start = st.time_input(
					"Normal Entry Start Time",
					value=datetime.strptime(cfg.normal_entry_start_time, "%H:%M").time()
				)
			
			with col2:
				casual_day = st.selectbox(
					"Casual Day",
					["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
					index=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"].index(
						db_settings.get("policy_casual_day", "Friday").title()
					)
				)
				
				entry_end = st.time_input(
					"Normal Entry End Time",
					value=datetime.strptime(cfg.normal_entry_end_time, "%H:%M").time()
				)
			
			st.markdown("---")
			st.markdown("#### Uniform Images")
			
			col1, col2 = st.columns(2)
			
			with col1:
				st.markdown("**Male Uniform Reference**")
				male_uniform = st.file_uploader("Upload male uniform image", type=["jpg", "jpeg", "png"], key="male_uniform")
				if male_uniform:
					st.image(male_uniform, caption="Male Uniform", width=200)
				
				st.markdown("**Male Casual Reference**")
				male_casual = st.file_uploader("Upload male casual image", type=["jpg", "jpeg", "png"], key="male_casual")
				if male_casual:
					st.image(male_casual, caption="Male Casual", width=200)
			
			with col2:
				st.markdown("**Female Uniform Reference**")
				female_uniform = st.file_uploader("Upload female uniform image", type=["jpg", "jpeg", "png"], key="female_uniform")
				if female_uniform:
					st.image(female_uniform, caption="Female Uniform", width=200)
				
				st.markdown("**Female Casual Reference**")
				female_casual = st.file_uploader("Upload female casual image", type=["jpg", "jpeg", "png"], key="female_casual")
				if female_casual:
					st.image(female_casual, caption="Female Casual", width=200)
			
			st.markdown("---")
			
			if st.form_submit_button("💾 Save Policy Settings"):
				# Update settings in DB
				from src.db import update_policy_settings
				
				update_policy_settings({
					"policy_uniform_day": uniform_day.lower(),
					"policy_casual_day": casual_day.lower(),
					"normal_entry_start_time": entry_start.strftime("%H:%M"),
					"normal_entry_end_time": entry_end.strftime("%H:%M"),
				}, cfg=cfg)
				
				# Save uniform images if provided
				if male_uniform:
					import cv2
					import numpy as np
					male_uniform_img = Image.open(male_uniform)
					male_uniform_arr = cv2.cvtColor(np.array(male_uniform_img), cv2.COLOR_RGB2BGR)
					cv2.imwrite(str(cfg.male_uniform_image), male_uniform_arr)
				
				if female_uniform:
					female_uniform_img = Image.open(female_uniform)
					female_uniform_arr = cv2.cvtColor(np.array(female_uniform_img), cv2.COLOR_RGB2BGR)
					cv2.imwrite(str(cfg.female_uniform_image), female_uniform_arr)
				
				if male_casual:
					male_casual_img = Image.open(male_casual)
					male_casual_arr = cv2.cvtColor(np.array(male_casual_img), cv2.COLOR_RGB2BGR)
					cv2.imwrite(str(cfg.male_casual_image), male_casual_arr)
				
				if female_casual:
					female_casual_img = Image.open(female_casual)
					female_casual_arr = cv2.cvtColor(np.array(female_casual_img), cv2.COLOR_RGB2BGR)
					cv2.imwrite(str(cfg.female_casual_image), female_casual_arr)
				
				st.success("✅ Policy settings updated successfully!")
				st.balloons()
		
		st.markdown("---")
		st.markdown("#### Current Settings")
		
		current_settings = {
			"Uniform Day": db_settings.get("policy_uniform_day", "wednesday"),
			"Casual Day": db_settings.get("policy_casual_day", "friday"),
			"Entry Start": cfg.normal_entry_start_time,
			"Entry End": cfg.normal_entry_end_time,
		}
		
		settings_df = pd.DataFrame([current_settings])
		st.dataframe(settings_df, width='stretch')


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
								cols[i % 5].image(img, caption=img_path.name, width='stretch')
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


def main():
	st.set_page_config(page_title="Student Attire & Safety Verification", layout="wide")
	ensure_dirs()
	init_session_state()
	init_db()

	# Initialize navigation state
	if "nav_selection" not in st.session_state:
		st.session_state.nav_selection = "Home"

	# Initialize admin authentication state
	if "admin_authenticated" not in st.session_state:
		st.session_state.admin_authenticated = False

	# Initialize admin credentials (stored in session for demo - in production, use secure storage)
	if "admin_username" not in st.session_state:
		st.session_state.admin_username = "admin"
	if "admin_password" not in st.session_state:
		st.session_state.admin_password = "admin123"

	# Initialize settings visibility
	if "show_admin_settings" not in st.session_state:
		st.session_state.show_admin_settings = False

	st.sidebar.title("Navigation")
	nav_options = ["Home", "Student Verification", "Student Registration", "Admin Dashboard", "Reports & Downloads", "Datasets", "Dataset & Training"]
	nav = st.sidebar.radio("Go to", nav_options, index=nav_options.index(st.session_state.nav_selection) if st.session_state.nav_selection in nav_options else 0)

	# Update navigation state when sidebar changes
	st.session_state.nav_selection = nav

	sidebar_settings()

	if nav == "Home":
		render_home()
	elif nav == "Student Verification":
		render_student_verification()
	elif nav == "Student Registration":
		render_student_registration()
	elif nav == "Admin Dashboard":
		render_admin_tab()
	elif nav == "Reports & Downloads":
		render_reports_downloads()
	elif nav == "Datasets":
		render_datasets()
	elif nav == "Dataset & Training":
		render_dataset_tab()


if __name__ == "__main__":
	main()
