

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
from src.face_recognition import get_face_system


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
	if "face_verified_student" not in st.session_state:
		st.session_state.face_verified_student = None
	if "sidebar_settings_expanded" not in st.session_state:
		st.session_state.sidebar_settings_expanded = False


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
	# Settings toggle button with arrow
	arrow = "▼" if st.session_state.sidebar_settings_expanded else "▶"
	if st.sidebar.button(f"{arrow} **Settings**", key="settings_toggle", use_container_width=True):
		st.session_state.sidebar_settings_expanded = not st.session_state.sidebar_settings_expanded
		st.rerun()

	if st.session_state.sidebar_settings_expanded:
		cfg: AppConfig = ensure_config_defaults(st.session_state.config)

		# General Settings Section
		st.sidebar.markdown("---")
		st.sidebar.markdown("### ⚙️ **General Settings**")

		col1, col2 = st.sidebar.columns(2)
		with col1:
			cfg.policy_profile = st.selectbox(
				"Policy Profile",
				["regular", "sports", "lab"],
				index=["regular", "sports", "lab"].index(cfg.policy_profile),
				help="Adjusts expected attire and safety items based on activity type"
			)
		with col2:
			cfg.hist_bins = st.slider(
				"Histogram Bins",
				8, 64, cfg.hist_bins, 4,
				help="Number of bins for color histogram analysis"
			)

		col3, col4 = st.sidebar.columns(2)
		with col3:
			cfg.expected_top = st.text_input(
				"Expected Top Color",
				value=cfg.expected_top,
				help="Color keyword for top attire (e.g., white, blue)"
			)
		with col4:
			cfg.expected_bottom = st.text_input(
				"Expected Bottom Color",
				value=cfg.expected_bottom,
				help="Color keyword for bottom attire (e.g., black, navy)"
			)

		# Model Settings Section
		st.sidebar.markdown("---")
		st.sidebar.markdown("### 🤖 **Model Settings**")

		col5, col6 = st.sidebar.columns(2)
		with col5:
			cfg.confidence_threshold = st.slider(
				"Decision Threshold",
				0.5, 0.95, float(cfg.confidence_threshold), 0.01,
				help="Minimum confidence score for classification"
			)
		with col6:
			cfg.max_video_fps = st.slider(
				"Max Video FPS",
				5, 30, cfg.max_video_fps, 1,
				help="Maximum frames per second for video processing"
			)

		st.sidebar.markdown("**Model Controls**")
		col7, col8 = st.sidebar.columns(2)
		with col7:
			cfg.enable_rules = st.checkbox(
				"Enable Rule-based Checks",
				value=cfg.enable_rules,
				help="Use predefined rules for attire verification"
			)
		with col8:
			cfg.enable_model = st.checkbox(
				"Enable ML Model",
				value=cfg.enable_model,
				help="Use machine learning model for classification"
			)

		cfg.save_frames = st.sidebar.checkbox(
			"Save Frames to Dataset",
			value=False,
			help="Automatically save processed frames for training"
		)
		cfg.current_label = st.sidebar.text_input(
			"Dataset Label",
			value=cfg.current_label,
			help="Label for saved training samples"
		)

		# ID Card Detection Section
		st.sidebar.markdown("---")
		st.sidebar.markdown("### 🆔 **ID Card Detection**")

		col9, col10 = st.sidebar.columns(2)
		with col9:
			cfg.enable_id_card_detection = st.checkbox(
				"Enable Detection",
				value=cfg.enable_id_card_detection,
				help="Activate automatic ID card detection"
			)
		with col10:
			cfg.id_card_required = st.checkbox(
				"ID Card Required",
				value=cfg.id_card_required,
				help="Require visible student ID card"
			)

		cfg.id_card_confidence_threshold = st.sidebar.slider(
			"Confidence Threshold",
			0.1, 0.9, float(cfg.id_card_confidence_threshold), 0.05,
			help="Minimum confidence for ID card detection"
		)

		# Uniform Policy Section
		st.sidebar.markdown("---")
		st.sidebar.markdown("### 👔 **Uniform Policy**")

		cfg.policy_gender = st.sidebar.selectbox(
			"Policy Gender Focus",
			["male", "female"],
			index=0 if cfg.policy_gender == "male" else 1,
			help="Gender-specific uniform requirements to configure"
		)

		st.sidebar.markdown("**Male Uniform Requirements**")
		col11, col12 = st.sidebar.columns(2)
		with col11:
			cfg.require_shirt_for_male = st.checkbox(
				"Require Formal Shirt",
				value=getattr(cfg, "require_shirt_for_male", True),
				help="Require formal shirt for male students"
			)
			cfg.require_black_shoes_male = st.checkbox(
				"Require Black Shoes",
				value=getattr(cfg, "require_black_shoes_male", True),
				help="Require black formal shoes for males"
			)
		with col12:
			cfg.allow_any_color_pants_male = st.checkbox(
				"Allow Any Color Pants",
				value=getattr(cfg, "allow_any_color_pants_male", True),
				help="Allow pants in any color for males"
			)
			cfg.require_footwear_male = st.checkbox(
				"Require Footwear",
				value=cfg.require_footwear_male,
				help="Require any type of footwear for males"
			)

		# Action Buttons Section
		st.sidebar.markdown("---")
		st.sidebar.markdown("### 🔧 **Actions**")

		col13, col14 = st.sidebar.columns(2)
		with col13:
			if st.sidebar.button("📥 Load Model", use_container_width=True):
				try:
					st.session_state.classifier.load(cfg.model_path)
					st.sidebar.success("✅ Model loaded successfully")
				except Exception as e:
					st.sidebar.error(f"❌ Failed to load model: {e}")

		with col14:
			if st.sidebar.button("🗑️ Clear Session", use_container_width=True):
				for k in list(st.session_state.keys()):
					if k not in ["config", "sidebar_settings_expanded"]:
						del st.session_state[k]
				st.sidebar.success("✅ Session cleared")
				st.rerun()


def render_home():
	# Professional header
	render_professional_header()
	
	# Welcome section with professional styling
	st.markdown("""
	<div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); 
	            padding: 2rem; 
	            border-radius: 10px; 
	            margin: 2rem 0;
	            border-left: 5px solid #1f77b4;">
		<h2 style="color: #1f77b4; margin-top: 0;">👋 Welcome to the Student Verification Portal</h2>
		<p style="font-size: 1.1rem; color: #495057; margin-bottom: 0;">
			Secure, efficient, and AI-powered student identity and attire verification system.
		</p>
	</div>
	""", unsafe_allow_html=True)

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
	
	# System Information with professional styling
	st.markdown("""
	<div style="background-color: #e7f3ff; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #1f77b4; margin: 2rem 0;">
		<h3 style="color: #1f77b4; margin-top: 0;">🔧 System Information</h3>
		<p style="color: #495057; margin-bottom: 0;">
			This system uses advanced AI and computer vision technologies to verify student attire compliance 
			and ensure safety standards are met. Powered by MediaPipe, scikit-learn, and Streamlit.
		</p>
	</div>
	""", unsafe_allow_html=True)

	# Quick Stats with professional styling
	st.markdown("### 📊 System Statistics")
	stats = get_compliance_stats(cfg=st.session_state.config)
	
	# Professional metric cards
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.markdown("""
		<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
		            padding: 1.5rem; 
		            border-radius: 10px; 
		            text-align: center;
		            color: white;
		            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
			<div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">{}</div>
			<div style="font-size: 0.9rem; opacity: 0.9;">Total Students</div>
		</div>
		""".format(stats["total_students"]), unsafe_allow_html=True)
	
	with col2:
		st.markdown("""
		<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
		            padding: 1.5rem; 
		            border-radius: 10px; 
		            text-align: center;
		            color: white;
		            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
			<div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">{}</div>
			<div style="font-size: 0.9rem; opacity: 0.9;">Verified Today</div>
		</div>
		""".format(stats["verified_students"]), unsafe_allow_html=True)
	
	with col3:
		compliance_color = "#2ca02c" if stats['compliance_percentage'] >= 80 else "#ff7f0e" if stats['compliance_percentage'] >= 60 else "#d62728"
		st.markdown("""
		<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
		            padding: 1.5rem; 
		            border-radius: 10px; 
		            text-align: center;
		            color: white;
		            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
			<div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; color: {};">{:.1f}%</div>
			<div style="font-size: 0.9rem; opacity: 0.9;">Compliance Rate</div>
		</div>
		""".format(compliance_color, stats['compliance_percentage']), unsafe_allow_html=True)
	
	with col4:
		st.markdown("""
		<div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
		            padding: 1.5rem; 
		            border-radius: 10px; 
		            text-align: center;
		            color: white;
		            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
			<div style="font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;">{}</div>
			<div style="font-size: 0.9rem; opacity: 0.9;">Events Today</div>
		</div>
		""".format(stats["total_events"]), unsafe_allow_html=True)
	
	# Add footer
	render_professional_footer()


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
	st.subheader("🔐 Student Login & Face Verification")
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

	face_system = get_face_system(cfg)
	if not face_system.has_face_data(student_id_input):
		st.warning("Face data not registered for this student. Please complete face registration first.")
		return

	st.markdown("#### 🧑‍💻 Face Verification")
	st.info(
		"Click the button below to enable camera for face verification. "
		"Upload a **recent selfie** taken right now (or use the camera input). "
		"This will be matched against the stored face data for the selected student."
	)
	
	# Initialize camera enabled state for this student
	camera_key = f"camera_enabled_login_{student_id_input}"
	if camera_key not in st.session_state:
		st.session_state[camera_key] = False
	
	# Button to enable camera
	if not st.session_state[camera_key]:
		if st.button("📷 Enable Camera for Face Verification", key="enable_camera_login_btn", type="primary"):
			st.session_state[camera_key] = True
			st.rerun()
	
	face_upload = st.file_uploader(
		"Upload a selfie (JPG/PNG)", type=["jpg", "jpeg", "png"], key="home_face_login_upload"
	)
	
	# Only show camera if enabled
	camera_capture = None
	if st.session_state[camera_key]:
		camera_capture = st.camera_input("Or capture using your webcam", key="home_face_login_camera")

	selected_image = face_upload or camera_capture

	if selected_image and st.button("✅ Verify Face Identity", key="home_face_verify_btn", type="primary", use_container_width=True):
		with st.spinner("🔄 Verifying face identity... Please wait."):
			img = Image.open(selected_image).convert("RGB")
			result = face_system.verify_face(student_id_input, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
		if result.success:
			st.success("✅ Face verified successfully! You can proceed to attire verification.")
			st.session_state.face_verified_student = student_id_input
			st.session_state.login_student_id = student_id_input
			st.session_state.nav_selection = "Student Verification"
			st.rerun()
		else:
			st.error(
				f"❌ Face verification failed. {result.message} "
				f"(difference score: {result.distance:.3f})"
			)


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

		if st.form_submit_button("Complete Registration", type="primary"):
			if not (student_id and name and gender):
				st.error("Name, gender, and student ID are required.")
			else:
				# Store form data in session state for processing after face capture
				st.session_state.reg_form_data = {
					"student_id": student_id,
					"name": name,
					"gender": gender,
					"email": email,
					"phone": phone,
					"selected_dept": selected_dept,
					"dept_id": dept_id,
				}
				st.rerun()

	# ============================================================================
	# Face registration section (OUTSIDE FORM - form context ends above)
	# ============================================================================
	st.markdown("---")
	st.markdown("#### 🙂 Face Registration (Required)")
	
	# Check if form was submitted and data is stored
	form_data = st.session_state.get("reg_form_data", {})
	name_val = form_data.get("name", "")
	gender_val = form_data.get("gender", "")
	student_id_val = form_data.get("student_id", "")
	
	# Also check if values exist in session state (for immediate display)
	if not name_val:
		name_val = st.session_state.get("home_reg_name", "")
	if not gender_val:
		gender_val = st.session_state.get("home_reg_gender", "")
	if not student_id_val:
		student_id_val = st.session_state.get("home_reg_student_id") or st.session_state.get("home_reg_student_id_manual", "")
	
	# Check if first stage is complete (student ID, name, gender filled)
	first_stage_complete = bool(student_id_val and name_val and gender_val)
	
	if not first_stage_complete:
		st.info("⚠️ Please complete the student information above (Name, Gender, and Student ID) and click 'Complete Registration' before proceeding to face registration.")
	else:
		# Initialize camera enabled state for registration
		camera_key_reg = "camera_enabled_registration"
		if camera_key_reg not in st.session_state:
			st.session_state[camera_key_reg] = False
		
		st.info("Click the button below to enable camera for face registration. Upload a clear face photo or capture one using your webcam. This will be used for future logins.")
		
		# Button to enable camera (outside form - this is safe now)
		if not st.session_state[camera_key_reg]:
			if st.button("📷 Enable Camera for Face Registration", key="enable_camera_reg_btn", type="primary"):
				st.session_state[camera_key_reg] = True
				st.rerun()
		
		face_upload = st.file_uploader("Upload face photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="home_face_upload")
		
		# Only show camera if enabled
		face_capture = None
		if st.session_state[camera_key_reg]:
			face_capture = st.camera_input("Or capture using your webcam", key="home_face_capture")

		face_image = None
		if face_capture is not None:
			face_image = Image.open(io.BytesIO(face_capture.getvalue())).convert("RGB")
		elif face_upload is not None:
			face_image = Image.open(face_upload).convert("RGB")

		if face_image is not None:
			st.success("✅ Face image ready for registration.")
			st.image(face_image, caption="Face Preview", width=250)
			
			# Process registration if form data is available
			if st.session_state.get("reg_form_data"):
				form_data = st.session_state.reg_form_data
				parsed = parse_student_id(form_data["student_id"], cfg=cfg)
				class_name = f"{parsed.get('batch_year', '')} - {parsed.get('class_section', '')}" if parsed else ""
				department_name = next((d['name'] for d in departments if d['department_id'] == parsed.get('department_id')), "") if parsed else ""

				add_student({
					"id": form_data["student_id"],
					"name": form_data["name"],
					"class": class_name or form_data["student_id"],
					"department": department_name or form_data["selected_dept"],
					"gender": form_data["gender"],
					"email": form_data["email"],
					"phone": form_data["phone"],
					"contact_info": "",
				}, cfg=cfg)

				face_system = get_face_system(cfg)
				face_array = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
				
				# Show processing message
				with st.spinner("🔄 Processing face image and storing data..."):
					face_result = face_system.register_face(form_data["student_id"], face_array)
				
				if not face_result.success:
					st.error(f"❌ Failed to store face data: {face_result.message}")
					st.warning("⚠️ **Possible reasons:**")
					st.warning("• No face detected in the image")
					st.warning("• Face is not clearly visible")
					st.warning("• Poor lighting conditions")
					st.warning("• MediaPipe library not available")
					st.info("💡 **Solution:** Please try again with a clearer face photo with good lighting.")
				else:
					# Registration completed successfully
					st.balloons()
					st.success(f"🎉 **REGISTRATION COMPLETED SUCCESSFULLY!**")
					st.success(f"Welcome, **{form_data['name']}**! Your account has been created.")
					
					# Display student information
					st.markdown("---")
					st.markdown("### 📋 Your Registration Details")
					col1, col2 = st.columns(2)
					with col1:
						st.info(f"**Student ID:** {form_data['student_id']}")
						st.info(f"**Name:** {form_data['name']}")
						st.info(f"**Department:** {department_name or form_data.get('selected_dept', 'N/A')}")
					with col2:
						st.info(f"**Class:** {class_name or 'N/A'}")
						st.info(f"**Gender:** {form_data['gender'].title()}")
						st.info(f"**Email:** {form_data.get('email', 'N/A')}")
					
					# Explain how image data is stored
					st.markdown("---")
					st.markdown("### 💾 How Your Face Data is Stored")
					with st.expander("📖 Learn about face data storage", expanded=True):
						st.markdown("""
						**Your face image data is securely stored in two ways:**
						
						1. **Face Embeddings in Database** 📊
						   - Your face image is processed using advanced AI (MediaPipe Face Mesh)
						   - Unique facial features are extracted and converted into a mathematical representation (embedding vector)
						   - This embedding is stored in the database (`attire.db`) linked to your Student ID
						   - **No actual images are stored** - only the mathematical representation for security and privacy
						
						2. **Backup JSON File** 📄
						   - A backup copy of your face embedding is also saved in `data/face_embeddings.json`
						   - This ensures data redundancy and recovery options
						
						**Security Features:**
						- ✅ Your face data is encrypted and cannot be reverse-engineered to recreate your image
						- ✅ Only the mathematical representation is stored, not the actual photo
						- ✅ Data is linked securely to your Student ID
						- ✅ Used only for identity verification during login
						""")
					
					st.markdown("---")
					
					# Verify face data was actually stored in database
					from src.db import get_student_face_embedding
					stored_face_check = get_student_face_embedding(form_data["student_id"], cfg=cfg)
					if stored_face_check is not None:
						st.markdown("""
						<div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
						            padding: 1.5rem; 
						            border-radius: 8px; 
						            border-left: 5px solid #2ca02c;
						            margin: 1rem 0;">
							<h4 style="color: #155724; margin-top: 0;">✅ Face Data Stored Successfully</h4>
							<p style="color: #155724; margin-bottom: 0.5rem;">
								Your face data has been securely stored in the database. You can now log in using face verification.
							</p>
							<p style="color: #155724; margin-bottom: 0; font-weight: 600;">
								👉 <strong>Next Step:</strong> Go to the Student Login page to verify your identity and access the system.
							</p>
						</div>
						""", unsafe_allow_html=True)
					else:
						st.markdown("""
						<div style="background-color: #f8d7da; padding: 1.5rem; border-radius: 8px; border-left: 5px solid #d62728; margin: 1rem 0;">
							<h4 style="color: #721c24; margin-top: 0;">⚠️ Warning: Face Data Verification Failed</h4>
							<p style="color: #721c24; margin-bottom: 0;">
								The face embedding may not have been stored properly. Please try registering your face again or contact the administrator.
							</p>
						</div>
						""", unsafe_allow_html=True)
					
					st.session_state.prefill_student_id = form_data["student_id"]
					st.session_state.home_auth_mode = "login"
					# Clear form data and generated ID
					if "reg_form_data" in st.session_state:
						del st.session_state["reg_form_data"]
					if "generated_student_id" in st.session_state:
						del st.session_state["generated_student_id"]
					if camera_key_reg in st.session_state:
						del st.session_state[camera_key_reg]
					st.rerun()
		else:
			if st.session_state.get("reg_form_data"):
				st.warning("⚠️ Face photo is required to complete registration.")
def handle_image(image: Image.Image, zone: str, student_id: Optional[str]) -> Dict[str, Any]:
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
	result = verify_attire_and_safety(bgr, student_id, zone, cfg, st.session_state.classifier)

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
	
	# Check if face verification is complete (for student attire verification)
	face_verified = st.session_state.get("face_verified_student") is not None
	
	if not face_verified:
		st.warning("⚠️ Please complete face verification first before using the camera for attire verification.")
		st.info("Go to the Student Verification page and complete face verification to enable the camera.")
		return
	
	# Initialize camera enabled state for attire verification
	camera_key_attire = "camera_enabled_attire_verification"
	if camera_key_attire not in st.session_state:
		st.session_state[camera_key_attire] = False
	
	st.info("Click the button below to enable camera for attire verification. Use the camera to capture a frame.")
	
	# Button to enable camera
	if not st.session_state[camera_key_attire]:
		if st.button("📷 Enable Camera for Attire Verification", key="enable_camera_attire_btn", type="primary"):
			st.session_state[camera_key_attire] = True
			st.rerun()
	
	# Only show camera if enabled
	cam = None
	if st.session_state[camera_key_attire]:
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




def render_student_registration():
	st.title("📝 Student Registration")
	st.markdown("### Register as a New Student")

	st.info("**Welcome!** Please fill out the registration form below. Your face data will be registered automatically during the process.")

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

		if st.form_submit_button("Complete Registration"):
			if not (student_id and name and gender):
				st.error("Please fill required fields (*) and ensure Student ID is valid")
			else:
				# Store form data in session state for processing after face capture
				st.session_state.student_reg_form_data = {
					"student_id": student_id,
					"name": name,
					"gender": gender,
					"email": email,
					"phone": phone,
					"dept_id": dept_id,
					"selected_dept": selected_dept,
				}
				st.rerun()

	# Face registration section (outside form)
	st.markdown("---")
	st.markdown("**Face Registration**")
	st.markdown("Face data will be registered automatically when the student is added.")
	
	# Get values from session state
	form_data = st.session_state.get("student_reg_form_data", {})
	name_val = form_data.get("name", "")
	gender_val = form_data.get("gender", "")
	student_id_val = form_data.get("student_id", "")
	
	# Check if first stage is complete (student ID, name, gender filled)
	first_stage_complete = bool(student_id_val and name_val and gender_val)
	
	if not first_stage_complete:
		st.info("⚠️ Please complete the student information above (Name, Gender, and Student ID) and click 'Complete Registration' before proceeding to face registration.")
	else:
		# Initialize camera enabled state for student registration
		camera_key_student_reg = "camera_enabled_student_registration"
		if camera_key_student_reg not in st.session_state:
			st.session_state[camera_key_student_reg] = False
		
		st.info("💡 **Note:** Click the button below to enable camera. Please capture a clear face photo for identity verification.")
		
		# Button to enable camera (outside form)
		if not st.session_state[camera_key_student_reg]:
			if st.button("📷 Enable Camera for Face Registration", key="enable_camera_student_reg_btn", type="primary"):
				st.session_state[camera_key_student_reg] = True
				st.rerun()
		
		reg_face_image = None
		reg_face_upload = st.file_uploader("Upload face photo (JPG/PNG)", type=["jpg", "jpeg", "png"], key="student_reg_face_upload")
		
		# Only show camera if enabled
		reg_face_capture = None
		if st.session_state[camera_key_student_reg]:
			reg_face_capture = st.camera_input("Or capture using your webcam", key="student_reg_face_capture")
		
		if reg_face_capture is not None:
			reg_face_image = Image.open(io.BytesIO(reg_face_capture.getvalue())).convert("RGB")
		elif reg_face_upload is not None:
			reg_face_image = Image.open(reg_face_upload).convert("RGB")

		if reg_face_image is not None:
			st.image(reg_face_image, caption="Face Preview", width=250)
			
			# Process registration if form data is available
			if form_data:
				# Parse student ID to get class and department info
				from src.db import parse_student_id
				parsed_id = parse_student_id(form_data["student_id"])

				if parsed_id:
					class_name = f"{parsed_id['batch_year']} - {parsed_id['class_section']}"
					department_name = next((d['name'] for d in departments if d['department_id'] == parsed_id['department_id']), "")
				else:
					class_name = ""
					department_name = ""

				# Add student to database
				add_student({
					"id": form_data["student_id"],
					"name": form_data["name"],
					"class": class_name,
					"department": department_name,
					"gender": form_data["gender"],
					"email": form_data["email"],
					"phone": form_data["phone"],
					"contact_info": ""
				}, cfg=st.session_state.config)

				face_system = get_face_system(st.session_state.config)
				face_result = face_system.register_face(form_data["student_id"], cv2.cvtColor(np.array(reg_face_image), cv2.COLOR_RGB2BGR))

				if face_result.success:
					# Registration completed successfully
					st.balloons()
					st.success(f"🎉 **REGISTRATION COMPLETED SUCCESSFULLY!**")
					st.success(f"Welcome, **{form_data['name']}**! Your account has been created.")
					
					# Display student information
					st.markdown("---")
					st.markdown("### 📋 Your Registration Details")
					col1, col2 = st.columns(2)
					with col1:
						st.info(f"**Student ID:** {form_data['student_id']}")
						st.info(f"**Name:** {form_data['name']}")
						st.info(f"**Department:** {department_name or 'N/A'}")
					with col2:
						st.info(f"**Class:** {class_name or 'N/A'}")
						st.info(f"**Gender:** {form_data['gender'].title()}")
						st.info(f"**Email:** {form_data.get('email', 'N/A')}")
					
					# Explain how image data is stored
					st.markdown("---")
					st.markdown("### 💾 How Your Face Data is Stored")
					with st.expander("📖 Learn about face data storage", expanded=True):
						st.markdown("""
						**Your face image data is securely stored in two ways:**
						
						1. **Face Embeddings in Database** 📊
						   - Your face image is processed using advanced AI (MediaPipe Face Mesh)
						   - Unique facial features are extracted and converted into a mathematical representation (embedding vector)
						   - This embedding is stored in the database (`attire.db`) linked to your Student ID
						   - **No actual images are stored** - only the mathematical representation for security and privacy
						
						2. **Backup JSON File** 📄
						   - A backup copy of your face embedding is also saved in `data/face_embeddings.json`
						   - This ensures data redundancy and recovery options
						
						**Security Features:**
						- ✅ Your face data is encrypted and cannot be reverse-engineered to recreate your image
						- ✅ Only the mathematical representation is stored, not the actual photo
						- ✅ Data is linked securely to your Student ID
						- ✅ Used only for identity verification during login
						""")
					
					st.markdown("---")
					
					# Verify face data was actually stored in database
					from src.db import get_student_face_embedding
					stored_face_check = get_student_face_embedding(form_data["student_id"], cfg=st.session_state.config)
					if stored_face_check is not None:
						st.markdown("""
						<div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
						            padding: 1.5rem; 
						            border-radius: 8px; 
						            border-left: 5px solid #2ca02c;
						            margin: 1rem 0;">
							<h4 style="color: #155724; margin-top: 0;">✅ Face Data Stored Successfully</h4>
							<p style="color: #155724; margin-bottom: 0.5rem;">
								Your face data has been securely stored in the database. You can now log in using face verification.
							</p>
							<p style="color: #155724; margin-bottom: 0; font-weight: 600;">
								👉 <strong>Next Step:</strong> Go to the Student Login page to verify your identity and access the system.
							</p>
						</div>
						""", unsafe_allow_html=True)
					else:
						st.error("⚠️ **Warning:** Face data verification failed. The embedding may not have been stored properly.")
						st.warning("Please try registering your face again or contact the administrator.")
					
					# Clear form data and generated ID
					if "student_reg_form_data" in st.session_state:
						del st.session_state["student_reg_form_data"]
					if hasattr(st.session_state, 'generated_student_id'):
						del st.session_state["generated_student_id"]
					if camera_key_student_reg in st.session_state:
						del st.session_state[camera_key_student_reg]
					st.rerun()
				else:
					st.error(f"❌ Failed to store face data: {face_result.message}")
		else:
			if form_data:
				st.warning("⚠️ Please upload or capture a face photo.")

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
		from src.db import get_student
		student_info = get_student(student_id_input, st.session_state.config)
		face_system = get_face_system(st.session_state.config)

		if not student_info:
			st.error("❌ Student ID not found in the system")
			st.info("Please check your Student ID or contact administrator if you haven't registered yet")
		else:
			st.success(f"✅ Welcome back, {student_info.get('name', 'Student')}!")
			st.info(f"**Department:** {student_info.get('department', 'N/A')} | **Class:** {student_info.get('class', 'N/A')}")

			if not face_system.has_face_data(student_id_input):
				st.warning("⚠️ Face data not registered yet.")
				st.info("Please complete face registration from the Student Registration page before logging in.")
				return

			if st.session_state.get("face_verified_student") != student_id_input:
				st.markdown("---")
				st.markdown("### 🙂 Face Verification Required")
				st.info("Click the button below to enable camera for face verification. Upload or capture a selfie to verify your identity.")
				
				# Initialize camera enabled state for verification
				camera_key_verify = f"camera_enabled_verify_{student_id_input}"
				if camera_key_verify not in st.session_state:
					st.session_state[camera_key_verify] = False
				
				# Button to enable camera
				if not st.session_state[camera_key_verify]:
					if st.button("📷 Enable Camera for Face Verification", key="enable_camera_verify_btn", type="primary"):
						st.session_state[camera_key_verify] = True
						st.rerun()
				
				face_upload = st.file_uploader("Upload selfie (JPG/PNG)", type=["jpg", "jpeg", "png"], key="login_face_upload")
				
				# Only show camera if enabled
				face_camera = None
				if st.session_state[camera_key_verify]:
					face_camera = st.camera_input("...or capture using your webcam", key="login_face_camera")
				
				selected_face = face_camera or face_upload
				if selected_face and st.button("✅ Verify Face", key="login_face_btn", type="primary", use_container_width=True):
					with st.spinner("🔄 Verifying face identity... Please wait."):
						img = Image.open(selected_face).convert("RGB")
						result = face_system.verify_face(student_id_input, cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
					if result.success:
						st.success("✅ Face verified!")
						st.session_state.face_verified_student = student_id_input
						st.rerun()
					else:
						st.error(f"Face verification failed: {result.message} (difference {result.distance:.3f})")
				return

			st.success("✅ Face verification completed.")
			st.markdown("---")
			st.markdown("### 📸 Choose Verification Method")
			tabs = st.tabs(["Direct Image", "Webcam", "Video"])
			with tabs[0]:
				render_image_tab()
			with tabs[1]:
				render_webcam_tab()
			with tabs[2]:
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
	tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["Students", "Compliance Reports", "Add Student", "Add User", "Departments & Classes", "Policy Settings", "Student History", "Reports & Downloads", "Datasets", "Dataset & Training"])
	
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

			if st.form_submit_button("Add/Update Student"):
				if not (student_id and name and gender):
					st.error("Please fill required fields (*) and ensure Student ID is valid")
				else:
					# Store form data in session state for processing after face capture
					st.session_state.admin_add_student_form_data = {
						"student_id": student_id,
						"name": name,
						"gender": gender,
						"email": email,
						"phone": phone,
						"dept_id": dept_id,
						"selected_dept": selected_dept,
					}
					st.rerun()

		# Face registration section (outside form)
		st.markdown("---")
		st.markdown("**Face Registration**")
		st.markdown("Face data will be registered automatically when the student is added.")
		
		# Get values from session state
		form_data = st.session_state.get("admin_add_student_form_data", {})
		name_val = form_data.get("name", "")
		gender_val = form_data.get("gender", "")
		student_id_val = form_data.get("student_id", "")
		
		# Check if first stage is complete (student ID, name, gender filled)
		first_stage_complete = bool(student_id_val and name_val and gender_val)
		
		if not first_stage_complete:
			st.info("⚠️ Please complete the student information above (Name, Gender, and Student ID) and click 'Add/Update Student' before proceeding to face registration.")
		else:
			st.info("💡 **Note:** Click the button below to enable camera. Make sure the student is present to capture a face photo.")
			
			# Initialize camera enabled state for admin registration
			camera_key_admin = "camera_enabled_admin_registration"
			if camera_key_admin not in st.session_state:
				st.session_state[camera_key_admin] = False
			
			# Button to enable camera (outside form)
			if not st.session_state[camera_key_admin]:
				if st.button("📷 Enable Camera for Face Capture", key="enable_camera_admin_btn", type="primary"):
					st.session_state[camera_key_admin] = True
					st.rerun()
			
			admin_face_image = None
			st.markdown("**📸 Face Capture**")
			st.info("📷 Please have the student take a clear photo of their face.")
			
			# Only show camera if enabled
			admin_face_capture = None
			if st.session_state[camera_key_admin]:
				admin_face_capture = st.camera_input("Capture face photo", key="admin_face_capture")
			
			if admin_face_capture is not None:
				admin_face_image = Image.open(io.BytesIO(admin_face_capture.getvalue())).convert("RGB")
				st.success("✅ Face captured successfully!")
				col1, col2 = st.columns(2)
				with col1:
					st.image(admin_face_image, caption="Captured Face", width=200)
				with col2:
					st.info("💡 **Tips:**\n• Good lighting\n• Face centered\n• No glasses or hats\n• Neutral expression")
				
				# Process registration if form data is available
				if form_data:
					# Parse student ID to get class and department info
					from src.db import parse_student_id
					parsed_id = parse_student_id(form_data["student_id"])

					if parsed_id:
						class_name = f"{parsed_id['batch_year']} - {parsed_id['class_section']}"
						department_name = next((d['name'] for d in departments if d['department_id'] == parsed_id['department_id']), "")
					else:
						class_name = ""
						department_name = ""

					add_student({
						"id": form_data["student_id"],
						"name": form_data["name"],
						"class": class_name,
						"department": department_name,
						"gender": form_data["gender"],
						"email": form_data["email"],
						"phone": form_data["phone"],
						"contact_info": ""
					}, cfg=st.session_state.config)

					face_system = get_face_system(st.session_state.config)
					face_result = face_system.register_face(form_data["student_id"], cv2.cvtColor(np.array(admin_face_image), cv2.COLOR_RGB2BGR))
					if face_result.success:
						# Registration completed successfully
						st.balloons()
						st.success(f"🎉 **STUDENT REGISTRATION COMPLETED SUCCESSFULLY!**")
						st.success(f"Student **{form_data['name']}** has been added to the system with face data.")
						
						# Display student information
						st.markdown("---")
						st.markdown("### 📋 Student Registration Details")
						col1, col2 = st.columns(2)
						with col1:
							st.info(f"**Student ID:** {form_data['student_id']}")
							st.info(f"**Name:** {form_data['name']}")
							st.info(f"**Department:** {department_name or 'N/A'}")
						with col2:
							st.info(f"**Class:** {class_name or 'N/A'}")
							st.info(f"**Gender:** {form_data['gender'].title()}")
							st.info(f"**Email:** {form_data.get('email', 'N/A')}")
						
						# Explain how image data is stored
						st.markdown("---")
						st.markdown("### 💾 How Face Data is Stored")
						with st.expander("📖 Learn about face data storage", expanded=True):
							st.markdown("""
							**The student's face image data is securely stored in two ways:**
							
							1. **Face Embeddings in Database** 📊
							   - The face image is processed using advanced AI (MediaPipe Face Mesh)
							   - Unique facial features are extracted and converted into a mathematical representation (embedding vector)
							   - This embedding is stored in the database (`attire.db`) linked to the Student ID
							   - **No actual images are stored** - only the mathematical representation for security and privacy
							
							2. **Backup JSON File** 📄
							   - A backup copy of the face embedding is also saved in `data/face_embeddings.json`
							   - This ensures data redundancy and recovery options
							
							**Security Features:**
							- ✅ Face data is encrypted and cannot be reverse-engineered to recreate the image
							- ✅ Only the mathematical representation is stored, not the actual photo
							- ✅ Data is linked securely to the Student ID
							- ✅ Used only for identity verification during login
							""")
						
						st.markdown("---")
						
						# Verify face data was actually stored in database
						from src.db import get_student_face_embedding
						stored_face_check = get_student_face_embedding(form_data["student_id"], cfg=st.session_state.config)
						if stored_face_check is not None:
							st.markdown("""
							<div style="background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); 
							            padding: 1.5rem; 
							            border-radius: 8px; 
							            border-left: 5px solid #2ca02c;
							            margin: 1rem 0;">
								<h4 style="color: #155724; margin-top: 0;">✅ Face Data Stored Successfully</h4>
								<p style="color: #155724; margin-bottom: 0.5rem;">
									The student's face data has been securely stored in the database. The student can now log in using face verification.
								</p>
								<p style="color: #155724; margin-bottom: 0; font-weight: 600;">
									👉 <strong>Next Step:</strong> The student can go to the Student Login page to verify their identity.
								</p>
							</div>
							""", unsafe_allow_html=True)
						else:
							st.error("⚠️ **Warning:** Face data verification failed. The embedding may not have been stored properly.")
							st.warning("Please try registering the face again.")
						
						# Clear form data and generated ID
						if "admin_add_student_form_data" in st.session_state:
							del st.session_state["admin_add_student_form_data"]
						if hasattr(st.session_state, 'generated_student_id'):
							del st.session_state["generated_student_id"]
						if camera_key_admin in st.session_state:
							del st.session_state[camera_key_admin]
						st.rerun()
					else:
						st.error(f"❌ Failed to store face data: {face_result.message}")
			else:
				if form_data:
					st.warning("⚠️ Please capture a face photo to complete registration.")
	
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
		st.subheader("⚙️ System Configuration")
		st.markdown("#### Complete System Settings")

		cfg = st.session_state.config

		# Load current settings from DB
		from src.db import get_policy_settings
		db_settings = get_policy_settings(cfg)

		# Form for editing all settings
		with st.form("system_settings_form"):
			# General Settings Section
			st.markdown("### ⚙️ **General Settings**")

			col1, col2 = st.columns(2)
			with col1:
				policy_profile = st.selectbox(
					"Policy Profile",
					["regular", "sports", "lab"],
					index=["regular", "sports", "lab"].index(cfg.policy_profile),
					help="Adjusts expected attire and safety items based on activity type"
				)
			with col2:
				hist_bins = st.slider(
					"Histogram Bins",
					8, 64, cfg.hist_bins, 4,
					help="Number of bins for color histogram analysis"
				)

			col3, col4 = st.columns(2)
			with col3:
				expected_top = st.text_input(
					"Expected Top Color",
					value=cfg.expected_top,
					help="Color keyword for top attire (e.g., white, blue)"
				)
			with col4:
				expected_bottom = st.text_input(
					"Expected Bottom Color",
					value=cfg.expected_bottom,
					help="Color keyword for bottom attire (e.g., black, navy)"
				)

			# Model Settings Section
			st.markdown("---")
			st.markdown("### 🤖 **Model Settings**")

			col5, col6 = st.columns(2)
			with col5:
				confidence_threshold = st.slider(
					"Decision Threshold",
					0.5, 0.95, float(cfg.confidence_threshold), 0.01,
					help="Minimum confidence score for classification"
				)
			with col6:
				max_video_fps = st.slider(
					"Max Video FPS",
					5, 30, cfg.max_video_fps, 1,
					help="Maximum frames per second for video processing"
				)

			st.markdown("**Model Controls**")
			col7, col8 = st.columns(2)
			with col7:
				enable_rules = st.checkbox(
					"Enable Rule-based Checks",
					value=cfg.enable_rules,
					help="Use predefined rules for attire verification"
				)
			with col8:
				enable_model = st.checkbox(
					"Enable ML Model",
					value=cfg.enable_model,
					help="Use machine learning model for classification"
				)

			save_frames = st.checkbox(
				"Save Frames to Dataset",
				value=False,
				help="Automatically save processed frames for training"
			)
			current_label = st.text_input(
				"Dataset Label",
				value=cfg.current_label,
				help="Label for saved training samples"
			)

			# ID Card Detection Section
			st.markdown("---")
			st.markdown("### 🆔 **ID Card Detection**")

			col9, col10 = st.columns(2)
			with col9:
				enable_id_card_detection = st.checkbox(
					"Enable Detection",
					value=cfg.enable_id_card_detection,
					help="Activate automatic ID card detection"
				)
			with col10:
				id_card_required = st.checkbox(
					"ID Card Required",
					value=cfg.id_card_required,
					help="Require visible student ID card"
				)

			id_card_confidence_threshold = st.slider(
				"Confidence Threshold",
				0.1, 0.9, float(cfg.id_card_confidence_threshold), 0.05,
				help="Minimum confidence for ID card detection"
			)

			# Uniform Policy Section
			st.markdown("---")
			st.markdown("### 👔 **Uniform Policy**")

			policy_gender = st.selectbox(
				"Policy Gender Focus",
				["male", "female"],
				index=0 if cfg.policy_gender == "male" else 1,
				help="Gender-specific uniform requirements to configure"
			)

			st.markdown("**Male Uniform Requirements**")
			col11, col12 = st.columns(2)
			with col11:
				require_shirt_for_male = st.checkbox(
					"Require Formal Shirt",
					value=getattr(cfg, "require_shirt_for_male", True),
					help="Require formal shirt for male students"
				)
				require_black_shoes_male = st.checkbox(
					"Require Black Shoes",
					value=getattr(cfg, "require_black_shoes_male", True),
					help="Require black formal shoes for males"
				)
			with col12:
				allow_any_color_pants_male = st.checkbox(
					"Allow Any Color Pants",
					value=getattr(cfg, "allow_any_color_pants_male", True),
					help="Allow pants in any color for males"
				)
				require_footwear_male = st.checkbox(
					"Require Footwear",
					value=cfg.require_footwear_male,
					help="Require any type of footwear for males"
				)

			# Days & Policy Configuration Section
			st.markdown("---")
			st.markdown("### 📅 **Days & Policy Configuration**")

			col13, col14 = st.columns(2)

			with col13:
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

			with col14:
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

			col15, col16 = st.columns(2)

			with col15:
				st.markdown("**Male Uniform Reference**")
				male_uniform = st.file_uploader("Upload male uniform image", type=["jpg", "jpeg", "png"], key="male_uniform")
				if male_uniform:
					st.image(male_uniform, caption="Male Uniform", width=200)

				st.markdown("**Male Casual Reference**")
				male_casual = st.file_uploader("Upload male casual image", type=["jpg", "jpeg", "png"], key="male_casual")
				if male_casual:
					st.image(male_casual, caption="Male Casual", width=200)

			with col16:
				st.markdown("**Female Uniform Reference**")
				female_uniform = st.file_uploader("Upload female uniform image", type=["jpg", "jpeg", "png"], key="female_uniform")
				if female_uniform:
					st.image(female_uniform, caption="Female Uniform", width=200)

				st.markdown("**Female Casual Reference**")
				female_casual = st.file_uploader("Upload female casual image", type=["jpg", "jpeg", "png"], key="female_casual")
				if female_casual:
					st.image(female_casual, caption="Female Casual", width=200)

			st.markdown("---")

			if st.form_submit_button("💾 Save All Settings"):
				# Update config object with form values
				cfg.policy_profile = policy_profile
				cfg.hist_bins = hist_bins
				cfg.expected_top = expected_top
				cfg.expected_bottom = expected_bottom
				cfg.confidence_threshold = confidence_threshold
				cfg.max_video_fps = max_video_fps
				cfg.enable_rules = enable_rules
				cfg.enable_model = enable_model
				cfg.save_frames = save_frames
				cfg.current_label = current_label
				cfg.enable_id_card_detection = enable_id_card_detection
				cfg.id_card_required = id_card_required
				cfg.id_card_confidence_threshold = id_card_confidence_threshold
				cfg.policy_gender = policy_gender
				cfg.require_shirt_for_male = require_shirt_for_male
				cfg.require_black_shoes_male = require_black_shoes_male
				cfg.allow_any_color_pants_male = allow_any_color_pants_male
				cfg.require_footwear_male = require_footwear_male

				# Save config
				cfg.save()

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

				st.success("✅ All system settings updated successfully!")
				st.balloons()

		st.markdown("---")
		st.markdown("#### Current Settings Overview")

		current_settings = {
			"Policy Profile": cfg.policy_profile,
			"Histogram Bins": cfg.hist_bins,
			"Expected Top Color": cfg.expected_top,
			"Expected Bottom Color": cfg.expected_bottom,
			"Confidence Threshold": cfg.confidence_threshold,
			"Max Video FPS": cfg.max_video_fps,
			"Enable Rules": cfg.enable_rules,
			"Enable Model": cfg.enable_model,
			"ID Card Detection": cfg.enable_id_card_detection,
			"ID Card Required": cfg.id_card_required,
			"ID Card Confidence": cfg.id_card_confidence_threshold,
			"Policy Gender": cfg.policy_gender,
			"Uniform Day": db_settings.get("policy_uniform_day", "wednesday"),
			"Casual Day": db_settings.get("policy_casual_day", "friday"),
			"Entry Start": cfg.normal_entry_start_time,
			"Entry End": cfg.normal_entry_end_time,
		}

		settings_df = pd.DataFrame([current_settings])
		st.dataframe(settings_df, width='stretch')

		# Model Actions Section
		st.markdown("---")
		st.markdown("### 🔧 **Model Actions**")

		col17, col18 = st.columns(2)
		with col17:
			if st.button("📥 Load Model", use_container_width=True):
				try:
					st.session_state.classifier.load(cfg.model_path)
					st.success("✅ Model loaded successfully")
				except Exception as e:
					st.error(f"❌ Failed to load model: {e}")

		with col18:
			if st.button("🗑️ Clear Session", use_container_width=True):
				for k in list(st.session_state.keys()):
					if k not in ["config", "admin_authenticated", "admin_username", "admin_password", "show_admin_settings", "nav_selection"]:
						del st.session_state[k]
				st.success("✅ Session cleared")
				st.rerun()

	with tab7:
		st.subheader("📚 Student History")
		st.markdown("#### Verification Event History")

		# Get all events
		all_events = list_events(limit=5000)  # Get more events for history

		if all_events:
			# Convert to DataFrame for easier manipulation
			events_df = pd.DataFrame(all_events)

			# Search/Filter functionality
			col1, col2 = st.columns([2, 1])
			with col1:
				search_student_id = st.text_input("🔍 Search by Student ID", key="history_search_student_id", placeholder="Enter student ID to filter...")
			with col2:
				date_filter = st.selectbox("Filter by Date", ["All", "Today", "Last 7 days", "Last 30 days"], key="history_date_filter")

			# Apply filters
			filtered_events = events_df.copy()

			# Student ID filter
			if search_student_id:
				filtered_events = filtered_events[filtered_events['student_id'].str.contains(search_student_id, case=False, na=False)]
				st.info(f"Found {len(filtered_events)} events for student ID containing '{search_student_id}'")

			# Date filter
			if date_filter != "All":
				now = pd.Timestamp.now()
				if date_filter == "Today":
					start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
				elif date_filter == "Last 7 days":
					start_date = now - pd.Timedelta(days=7)
				elif date_filter == "Last 30 days":
					start_date = now - pd.Timedelta(days=30)

				# Convert timestamp to datetime if it's not already
				if 'timestamp' in filtered_events.columns:
					filtered_events['timestamp'] = pd.to_datetime(filtered_events['timestamp'], errors='coerce')
					filtered_events = filtered_events[filtered_events['timestamp'] >= start_date]

			# Display results
			if not filtered_events.empty:
				st.markdown(f"**Showing {len(filtered_events)} events**")

				# Summary statistics
				col1, col2, col3, col4 = st.columns(4)
				with col1:
					pass_count = len(filtered_events[filtered_events['status'] == 'PASS'])
					st.metric("PASS Events", pass_count)
				with col2:
					fail_count = len(filtered_events[filtered_events['status'] == 'FAIL'])
					st.metric("FAIL Events", fail_count)
				with col3:
					warning_count = len(filtered_events[filtered_events['status'] == 'WARNING'])
					st.metric("WARNING Events", warning_count)
				with col4:
					unique_students = filtered_events['student_id'].nunique()
					st.metric("Unique Students", unique_students)

				st.markdown("---")

				# Display events table
				# Reorder columns for better readability
				display_cols = ['timestamp', 'student_id', 'zone', 'status', 'score', 'label', 'details']
				available_cols = [col for col in display_cols if col in filtered_events.columns]
				st.dataframe(filtered_events[available_cols], width='stretch')

				# Export functionality
				st.markdown("---")
				st.markdown("#### 📥 Export History")

				col1, col2 = st.columns(2)
				with col1:
					if not filtered_events.empty:
						csv_data = filtered_events.to_csv(index=False).encode('utf-8')
						st.download_button(
							label="📄 Download Filtered History (CSV)",
							data=csv_data,
							file_name=f"student_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
							mime="text/csv",
							key="download_history_csv"
						)

				with col2:
					if not filtered_events.empty:
						# Create a summary report
						summary = {
							"Total Events": len(filtered_events),
							"PASS Events": pass_count,
							"FAIL Events": fail_count,
							"WARNING Events": warning_count,
							"Unique Students": unique_students,
							"Date Range": f"{filtered_events['timestamp'].min()} to {filtered_events['timestamp'].max()}" if 'timestamp' in filtered_events.columns else "N/A",
							"Generated At": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
						}

						summary_text = "\n".join([f"{k}: {v}" for k, v in summary.items()])
						st.download_button(
							label="📊 Download Summary Report (TXT)",
							data=summary_text,
							file_name=f"history_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
							mime="text/plain",
							key="download_history_summary"
						)

				# Detailed view for specific events
				st.markdown("---")
				st.markdown("#### 🔍 Detailed Event View")

				if len(filtered_events) <= 100:  # Only show if not too many events
					selected_event_index = st.selectbox(
						"Select an event to view details",
						options=range(len(filtered_events)),
						format_func=lambda x: f"{filtered_events.iloc[x]['timestamp']} - {filtered_events.iloc[x]['student_id']} - {filtered_events.iloc[x]['status']}",
						key="selected_event_details"
					)

					if selected_event_index is not None:
						selected_event = filtered_events.iloc[selected_event_index]

						st.markdown("**Event Details:**")
						detail_cols = st.columns(2)

						with detail_cols[0]:
							st.write(f"**Student ID:** {selected_event.get('student_id', 'N/A')}")
							st.write(f"**Zone:** {selected_event.get('zone', 'N/A')}")
							st.write(f"**Status:** {selected_event.get('status', 'N/A')}")
							st.write(f"**Score:** {selected_event.get('score', 'N/A')}")

						with detail_cols[1]:
							st.write(f"**Timestamp:** {selected_event.get('timestamp', 'N/A')}")
							st.write(f"**Label:** {selected_event.get('label', 'N/A')}")
							st.write(f"**Event ID:** {selected_event.get('id', 'N/A')}")

						if 'details' in selected_event and selected_event['details']:
							st.markdown("**Additional Details:**")
							try:
								# Try to parse as JSON for better display
								import json
								details_dict = json.loads(selected_event['details'])
								st.json(details_dict)
							except:
								st.text(selected_event['details'])
				else:
					st.info("Too many events to show detailed view. Please filter the results to fewer than 100 events.")

			else:
				if search_student_id or date_filter != "All":
					st.warning("No events found matching the current filters.")
				else:
					st.info("No events found in the system.")
		else:
			st.info("No events found in the system yet. Events will appear here after students complete verification processes.")


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


def apply_custom_css():
	"""Apply professional custom CSS styling"""
	st.markdown("""
	<style>
		/* Professional Color Scheme */
		:root {
			--primary-color: #1f77b4;
			--secondary-color: #2ca02c;
			--accent-color: #ff7f0e;
			--danger-color: #d62728;
			--success-color: #2ca02c;
			--warning-color: #ff7f0e;
			--info-color: #17a2b8;
			--dark-bg: #0e1117;
			--light-bg: #ffffff;
		}
		
		/* Main Container Styling */
		.main .block-container {
			padding-top: 2rem;
			padding-bottom: 2rem;
			max-width: 1200px;
		}
		
		/* Professional Header Styling */
		.stApp > header {
			background-color: #1f77b4;
		}
		
		/* Professional Title Styling */
		h1 {
			color: #1f77b4;
			font-weight: 700;
			border-bottom: 3px solid #1f77b4;
			padding-bottom: 0.5rem;
			margin-bottom: 1.5rem;
		}
		
		h2 {
			color: #2c3e50;
			font-weight: 600;
			margin-top: 2rem;
			margin-bottom: 1rem;
		}
		
		h3 {
			color: #34495e;
			font-weight: 600;
		}
		
		/* Professional Button Styling */
		.stButton > button {
			border-radius: 8px;
			font-weight: 600;
			transition: all 0.3s ease;
			border: none;
			box-shadow: 0 2px 4px rgba(0,0,0,0.1);
		}
		
		.stButton > button:hover {
			transform: translateY(-2px);
			box-shadow: 0 4px 8px rgba(0,0,0,0.2);
		}
		
		/* Professional Card Styling */
		.stAlert {
			border-radius: 8px;
			border-left: 4px solid;
			padding: 1rem;
		}
		
		/* Professional Form Styling */
		.stTextInput > div > div > input,
		.stSelectbox > div > div > select {
			border-radius: 6px;
			border: 2px solid #e0e0e0;
			transition: border-color 0.3s ease;
		}
		
		.stTextInput > div > div > input:focus,
		.stSelectbox > div > div > select:focus {
			border-color: #1f77b4;
			box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
		}
		
		/* Professional Metric Cards */
		[data-testid="stMetricValue"] {
			font-size: 2rem;
			font-weight: 700;
			color: #1f77b4;
		}
		
		[data-testid="stMetricLabel"] {
			font-size: 0.9rem;
			color: #7f8c8d;
			text-transform: uppercase;
			letter-spacing: 0.5px;
		}
		
		/* Professional Sidebar */
		.css-1d391kg {
			background-color: #f8f9fa;
		}
		
		/* Professional Success Messages */
		.stSuccess {
			background-color: #d4edda;
			border-left-color: #2ca02c;
			color: #155724;
		}
		
		/* Professional Error Messages */
		.stError {
			background-color: #f8d7da;
			border-left-color: #d62728;
			color: #721c24;
		}
		
		/* Professional Warning Messages */
		.stWarning {
			background-color: #fff3cd;
			border-left-color: #ff7f0e;
			color: #856404;
		}
		
		/* Professional Info Messages */
		.stInfo {
			background-color: #d1ecf1;
			border-left-color: #17a2b8;
			color: #0c5460;
		}
		
		/* Professional Table Styling */
		.dataframe {
			border-radius: 8px;
			overflow: hidden;
			box-shadow: 0 2px 4px rgba(0,0,0,0.1);
		}
		
		/* Professional Divider */
		hr {
			border: none;
			border-top: 2px solid #e0e0e0;
			margin: 2rem 0;
		}
		
		/* Professional Footer */
		.footer {
			position: fixed;
			bottom: 0;
			left: 0;
			right: 0;
			background-color: #2c3e50;
			color: white;
			text-align: center;
			padding: 1rem;
			font-size: 0.85rem;
			z-index: 999;
		}
		
		/* Professional Loading Spinner */
		.stSpinner > div {
			border-top-color: #1f77b4;
		}
		
		/* Professional Badge Styling */
		.badge {
			display: inline-block;
			padding: 0.25rem 0.75rem;
			border-radius: 12px;
			font-size: 0.875rem;
			font-weight: 600;
			background-color: #1f77b4;
			color: white;
		}
		
		/* Professional Code Blocks */
		.stCodeBlock {
			border-radius: 8px;
			background-color: #f8f9fa;
		}
		
		/* Hide Streamlit Branding */
		#MainMenu {visibility: hidden;}
		footer {visibility: hidden;}
		header {visibility: visible;}
		
		/* Professional Tabs */
		.stTabs [data-baseweb="tab-list"] {
			gap: 8px;
		}
		
		.stTabs [data-baseweb="tab"] {
			border-radius: 8px 8px 0 0;
			padding: 0.75rem 1.5rem;
			font-weight: 600;
		}
		
		/* Professional Expander */
		.streamlit-expanderHeader {
			font-weight: 600;
			background-color: #f8f9fa;
			border-radius: 6px;
		}
	</style>
	""", unsafe_allow_html=True)


def render_professional_header():
	"""Render professional header with branding"""
	st.markdown("""
	<div style="background: linear-gradient(135deg, #1f77b4 0%, #2ca02c 100%); 
	            padding: 2rem; 
	            border-radius: 10px; 
	            margin-bottom: 2rem;
	            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
		<h1 style="color: white; margin: 0; font-size: 2.5rem; text-align: center;">
			🎓 Student Attire & Safety Verification System
		</h1>
		<p style="color: rgba(255,255,255,0.9); text-align: center; margin-top: 0.5rem; font-size: 1.1rem;">
			AI-Powered Compliance Monitoring & Identity Verification Platform
		</p>
	</div>
	""", unsafe_allow_html=True)


def render_professional_footer():
	"""Render professional footer"""
	st.markdown("""
	<div style="margin-top: 4rem; padding: 2rem; background-color: #f8f9fa; border-radius: 8px; text-align: center;">
		<p style="color: #6c757d; margin: 0.5rem 0;">
			<strong>Student Attire & Safety Verification System</strong> | 
			Powered by Streamlit, MediaPipe & Machine Learning
		</p>
		<p style="color: #adb5bd; font-size: 0.85rem; margin: 0.5rem 0;">
			© 2024 All Rights Reserved | Secure • Reliable • Professional
		</p>
	</div>
	""", unsafe_allow_html=True)


def main():
	st.set_page_config(
		page_title="Student Attire Verification System",
		page_icon="🎓",
		layout="wide",
		initial_sidebar_state="expanded",
		menu_items={
			'Get Help': None,
			'Report a bug': None,
			'About': "Student Attire & Safety Verification System - AI-Powered Compliance Monitoring Platform"
		}
	)
	
	# Apply professional styling
	apply_custom_css()
	
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
	nav_options = ["Home", "Student Verification", "Student Registration", "Admin"]
	nav = st.sidebar.radio("Go to", nav_options, index=nav_options.index(st.session_state.nav_selection) if st.session_state.nav_selection in nav_options else 0)

	# Update navigation state when sidebar changes
	st.session_state.nav_selection = nav

	if nav == "Admin":
		sidebar_settings()
	elif nav == "Home":
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
	else:
		st.error("Invalid navigation option selected. Redirecting to Home.")
		st.session_state.nav_selection = "Home"
		st.rerun()


if __name__ == "__main__":
	main()
