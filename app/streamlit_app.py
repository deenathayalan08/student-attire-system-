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
from src.rbac import (
	require_permission, require_admin, require_student_or_admin, require_login,
	Permission, Role, has_permission, is_admin, is_student, is_logged_in,
	filter_navigation_by_role, show_permission_denied_message, log_access_attempt,
	can_manage_students, can_manage_departments, can_view_all_reports, 
	can_modify_system_settings, can_delete_data, can_export_data,
	check_data_access_permission, get_current_user, get_user_role
)


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


# Note: is_logged_in(), is_admin(), and is_student() functions are now imported from src.rbac module


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
	"""System settings - Admin only with RBAC protection"""
	# RBAC: Only admins can access system settings
	if not can_modify_system_settings():
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
	# Hero section with modern design
	st.markdown("""
	<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem 2rem; margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px; text-align: center;">
		<h1 style="color: white; font-size: 2.5rem; margin-bottom: 0.5rem; font-weight: 700;">🏫 SAVS Enterprise</h1>
		<h3 style="color: rgba(255,255,255,0.9); margin-bottom: 1rem; font-weight: 400;">Student Attire Verification System</h3>
		<p style="color: rgba(255,255,255,0.8); font-size: 1.1rem; max-width: 600px; margin: 0 auto;">
			AI-powered dress code compliance monitoring with biometric authentication and real-time analytics
		</p>
	</div>
	""", unsafe_allow_html=True)

	# System status dashboard
	col1, col2, col3, col4 = st.columns(4)
	
	with col1:
		st.markdown("""
		<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; border-left: 4px solid #28a745;">
			<div style="font-size: 2rem; color: #28a745; margin-bottom: 0.5rem;">✅</div>
			<div style="font-weight: 600; color: #2c3e50;">System Online</div>
			<div style="color: #6c757d; font-size: 0.9rem;">All services active</div>
		</div>
		""", unsafe_allow_html=True)
	
	with col2:
		# Get real stats
		try:
			from .db import get_compliance_stats
			stats = get_compliance_stats(cfg=st.session_state.config)
			total_students = stats.get('total_students', 0)
		except:
			total_students = 0
			
		st.markdown(f"""
		<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; border-left: 4px solid #007bff;">
			<div style="font-size: 2rem; color: #007bff; margin-bottom: 0.5rem;">👥</div>
			<div style="font-weight: 600; color: #2c3e50;">{total_students} Students</div>
			<div style="color: #6c757d; font-size: 0.9rem;">Registered users</div>
		</div>
		""", unsafe_allow_html=True)
	
	with col3:
		try:
			verified_students = stats.get('verified_students', 0)
		except:
			verified_students = 0
			
		st.markdown(f"""
		<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; border-left: 4px solid #ffc107;">
			<div style="font-size: 2rem; color: #ffc107; margin-bottom: 0.5rem;">🔐</div>
			<div style="font-weight: 600; color: #2c3e50;">{verified_students} Verified</div>
			<div style="color: #6c757d; font-size: 0.9rem;">Face authenticated</div>
		</div>
		""", unsafe_allow_html=True)
	
	with col4:
		try:
			compliance_rate = stats.get('compliance_percentage', 0)
		except:
			compliance_rate = 0
			
		st.markdown(f"""
		<div style="background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); text-align: center; border-left: 4px solid #17a2b8;">
			<div style="font-size: 2rem; color: #17a2b8; margin-bottom: 0.5rem;">📊</div>
			<div style="font-weight: 600; color: #2c3e50;">{compliance_rate:.1f}%</div>
			<div style="color: #6c757d; font-size: 0.9rem;">Compliance rate</div>
		</div>
		""", unsafe_allow_html=True)
	
	st.markdown("<br>", unsafe_allow_html=True)
	
	# Quick access with modern cards
	st.markdown("""
	<div style="margin: 2rem 0 1rem 0;">
		<h3 style="color: #2c3e50; font-weight: 600; margin-bottom: 1rem;">🚀 Quick Access</h3>
	</div>
	""", unsafe_allow_html=True)
	
	col1, col2, col3 = st.columns(3)
	
	with col1:
		st.markdown("""
		<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
			<div style="font-size: 3rem; margin-bottom: 1rem;">🎓</div>
			<h4 style="margin-bottom: 1rem; font-weight: 600;">Student Portal</h4>
			<p style="margin-bottom: 1.5rem; opacity: 0.9;">Login, register, and verify your attire compliance with AI-powered analysis</p>
		</div>
		""", unsafe_allow_html=True)
		
		if st.button("🎓 Enter Student Portal", use_container_width=True, type="primary", key="home_student"):
			navigate_to("student_portal")
	
	with col2:
		st.markdown("""
		<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
			<div style="font-size: 3rem; margin-bottom: 1rem;">👨‍💼</div>
			<h4 style="margin-bottom: 1rem; font-weight: 600;">Admin Portal</h4>
			<p style="margin-bottom: 1.5rem; opacity: 0.9;">Manage students, departments, and view comprehensive analytics reports</p>
		</div>
		""", unsafe_allow_html=True)
		
		if st.button("👨‍💼 Admin Access", use_container_width=True, type="primary", key="home_admin"):
			navigate_to("admin_login")
	
	with col3:
		st.markdown("""
		<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin-bottom: 1rem;">
			<div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
			<h4 style="margin-bottom: 1rem; font-weight: 600;">Analytics Hub</h4>
			<p style="margin-bottom: 1.5rem; opacity: 0.9;">Real-time compliance monitoring and advanced reporting dashboard</p>
		</div>
		""", unsafe_allow_html=True)
		
		if st.button("📊 View Analytics", use_container_width=True, type="primary", key="home_analytics"):
			if is_logged_in():
				if has_permission(Permission.SYSTEM_SETTINGS):
					navigate_to("admin_dashboard")
				elif has_permission(Permission.VIEW_OWN_REPORTS):
					navigate_to("student_dashboard")
				else:
					st.error("🚫 Access denied: Insufficient permissions for analytics")
					st.info("Contact administrator for access")
			else:
				st.info("🔐 Please login to access analytics")
	
	# Feature showcase
	st.markdown("<br><br>", unsafe_allow_html=True)
	st.markdown("""
	<div style="margin: 2rem 0 1rem 0;">
		<h3 style="color: #2c3e50; font-weight: 600; margin-bottom: 1rem;">✨ Enterprise Features</h3>
	</div>
	""", unsafe_allow_html=True)
	
	col1, col2 = st.columns(2)
	
	with col1:
		st.markdown("""
		<div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
			<h4 style="color: #007bff; margin-bottom: 1rem;">🔐 Advanced Authentication</h4>
			<ul style="color: #495057; line-height: 1.8;">
				<li><strong>Multi-method Face Recognition</strong> - 30% confidence threshold with histogram correlation</li>
				<li><strong>Emergency Login System</strong> - Backup username/password authentication</li>
				<li><strong>Biometric Security</strong> - Encrypted face storage with unique hashing</li>
				<li><strong>Real-time Verification</strong> - Instant face matching with quality analysis</li>
			</ul>
		</div>
		""", unsafe_allow_html=True)
	
	with col2:
		st.markdown("""
		<div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
			<h4 style="color: #28a745; margin-bottom: 1rem;">🎯 AI-Powered Analysis</h4>
			<ul style="color: #495057; line-height: 1.8;">
				<li><strong>Computer Vision Detection</strong> - HSV color analysis for uniform compliance</li>
				<li><strong>Object Recognition</strong> - ID card and footwear detection</li>
				<li><strong>Pose Estimation</strong> - MediaPipe integration for body region analysis</li>
				<li><strong>Violation Tracking</strong> - Detailed compliance scoring and reporting</li>
			</ul>
		</div>
		""", unsafe_allow_html=True)
	
	# Technology stack
	st.markdown("<br><br>", unsafe_allow_html=True)
	st.markdown("""
	<div style="background: #f8f9fa; padding: 2rem; border-radius: 15px; text-align: center;">
		<h4 style="color: #2c3e50; margin-bottom: 1rem;">🛠️ Powered by Enterprise Technology</h4>
		<div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 1rem; margin-top: 1rem;">
			<span style="background: white; padding: 0.5rem 1rem; border-radius: 20px; color: #495057; font-weight: 500;">Python 3.11</span>
			<span style="background: white; padding: 0.5rem 1rem; border-radius: 20px; color: #495057; font-weight: 500;">Streamlit</span>
			<span style="background: white; padding: 0.5rem 1rem; border-radius: 20px; color: #495057; font-weight: 500;">OpenCV</span>
			<span style="background: white; padding: 0.5rem 1rem; border-radius: 20px; color: #495057; font-weight: 500;">MediaPipe</span>
			<span style="background: white; padding: 0.5rem 1rem; border-radius: 20px; color: #495057; font-weight: 500;">SQLite</span>
			<span style="background: white; padding: 0.5rem 1rem; border-radius: 20px; color: #495057; font-weight: 500;">scikit-learn</span>
		</div>
	</div>
	""", unsafe_allow_html=True)


def render_student_portal():
	"""Enhanced student portal with modern enterprise UI"""
	
	# Modern header
	st.markdown("""
	<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px; text-align: center;">
		<h1 style="color: white; font-size: 2.2rem; margin-bottom: 0.5rem; font-weight: 600;">🎓 Student Portal</h1>
		<p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0;">Your gateway to attire verification and compliance tracking</p>
	</div>
	""", unsafe_allow_html=True)

	# Check if user is already logged in
	if is_logged_in():
		user = st.session_state.get('user')
		
		# Welcome back section
		st.markdown(f"""
		<div style="background: linear-gradient(135deg, #28a745, #20c997); padding: 1.5rem; border-radius: 15px; color: white; margin-bottom: 2rem; text-align: center;">
			<h3 style="margin-bottom: 0.5rem;">✅ Welcome back, {user.get('full_name', 'User')[:25]}!</h3>
			<p style="margin: 0; opacity: 0.9;">Logged in via {user.get('auth_method', 'Unknown').title()} Authentication</p>
		</div>
		""", unsafe_allow_html=True)
		
		# Quick actions dashboard
		st.markdown("""
		<div style="margin: 2rem 0 1rem 0;">
			<h3 style="color: #2c3e50; font-weight: 600;">⚡ Quick Actions</h3>
		</div>
		""", unsafe_allow_html=True)
		
		col1, col2, col3 = st.columns(3)
		
		with col1:
			st.markdown("""
			<div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; border-top: 4px solid #007bff;">
				<div style="font-size: 2.5rem; color: #007bff; margin-bottom: 1rem;">📋</div>
				<h4 style="color: #2c3e50; margin-bottom: 1rem;">My Dashboard</h4>
				<p style="color: #6c757d; margin-bottom: 1.5rem;">View your compliance history, statistics, and verification reports</p>
			</div>
			""", unsafe_allow_html=True)
			
			if st.button("📋 Open Dashboard", use_container_width=True, type="primary", key="dashboard_btn"):
				navigate_to("student_dashboard")
		
		with col2:
			st.markdown("""
			<div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; border-top: 4px solid #28a745;">
				<div style="font-size: 2.5rem; color: #28a745; margin-bottom: 1rem;">🎓</div>
				<h4 style="color: #2c3e50; margin-bottom: 1rem;">Verify Attire</h4>
				<p style="color: #6c757d; margin-bottom: 1.5rem;">Real-time attire verification using AI-powered analysis</p>
			</div>
			""", unsafe_allow_html=True)
			
			if st.button("🎓 Start Verification", use_container_width=True, type="primary", key="verify_btn"):
				navigate_to("verification")
		
		with col3:
			st.markdown("""
			<div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); text-align: center; border-top: 4px solid #ffc107;">
				<div style="font-size: 2.5rem; color: #ffc107; margin-bottom: 1rem;">👤</div>
				<h4 style="color: #2c3e50; margin-bottom: 1rem;">My Profile</h4>
				<p style="color: #6c757d; margin-bottom: 1.5rem;">Manage your personal information and account settings</p>
			</div>
			""", unsafe_allow_html=True)
			
			if st.button("👤 View Profile", use_container_width=True, type="primary", key="profile_btn_portal"):
				navigate_to("profile")
		
		# Additional actions
		st.markdown("<br>", unsafe_allow_html=True)
		col1, col2 = st.columns([3, 1])
		with col2:
			if st.button("🚪 Logout", use_container_width=True, type="secondary", key="logout_btn_portal"):
				logout_and_redirect()
	
	else:
		# Login/Register section for non-authenticated users
		st.markdown("""
		<div style="text-align: center; margin: 2rem 0;">
			<h3 style="color: #2c3e50; font-weight: 600;">Welcome to Student Portal!</h3>
			<p style="color: #6c757d; font-size: 1.1rem;">Choose your path to get started</p>
		</div>
		""", unsafe_allow_html=True)

		col1, col2 = st.columns(2)

		with col1:
			st.markdown("""
			<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2.5rem; border-radius: 20px; text-align: center; color: white; margin-bottom: 1rem;">
				<div style="font-size: 4rem; margin-bottom: 1rem;">📝</div>
				<h3 style="margin-bottom: 1rem; font-weight: 600;">New Student</h3>
				<p style="margin-bottom: 1.5rem; opacity: 0.9; line-height: 1.6;">
					Create your account with our 4-stage registration process including biometric face authentication
				</p>
				<div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
					<div style="font-size: 0.9rem; font-weight: 500;">✨ What you'll get:</div>
					<div style="font-size: 0.8rem; opacity: 0.9; margin-top: 0.5rem;">
						• Secure face authentication<br>
						• Personal compliance dashboard<br>
						• Real-time verification system
					</div>
				</div>
			</div>
			""", unsafe_allow_html=True)
			
			if st.button("📝 Start Registration", use_container_width=True, type="primary", key="portal_register"):
				navigate_to('register')

		with col2:
			st.markdown("""
			<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 2.5rem; border-radius: 20px; text-align: center; color: white; margin-bottom: 1rem;">
				<div style="font-size: 4rem; margin-bottom: 1rem;">🔐</div>
				<h3 style="margin-bottom: 1rem; font-weight: 600;">Existing Student</h3>
				<p style="margin-bottom: 1.5rem; opacity: 0.9; line-height: 1.6;">
					Login using advanced face authentication or emergency credentials for instant access
				</p>
				<div style="background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
					<div style="font-size: 0.9rem; font-weight: 500;">🚀 Login options:</div>
					<div style="font-size: 0.8rem; opacity: 0.9; margin-top: 0.5rem;">
						• Face biometric authentication<br>
						• Emergency login (ID + Password)<br>
						• Quality analysis & bypass options
					</div>
				</div>
			</div>
			""", unsafe_allow_html=True)
			
			if st.button("🔐 Login Now", use_container_width=True, type="primary", key="portal_login"):
				navigate_to('face_auth')

		# Process flow
		st.markdown("<br><br>", unsafe_allow_html=True)
		st.markdown("""
		<div style="background: #f8f9fa; padding: 2rem; border-radius: 15px;">
			<h4 style="color: #2c3e50; text-align: center; margin-bottom: 2rem;">📋 How It Works</h4>
			<div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
				<div style="text-align: center; flex: 1; min-width: 200px;">
					<div style="background: #007bff; color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold;">1</div>
					<h5 style="color: #2c3e50; margin-bottom: 0.5rem;">Register/Login</h5>
					<p style="color: #6c757d; font-size: 0.9rem;">Create account or authenticate with face/credentials</p>
				</div>
				<div style="text-align: center; flex: 1; min-width: 200px;">
					<div style="background: #28a745; color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold;">2</div>
					<h5 style="color: #2c3e50; margin-bottom: 0.5rem;">Verify Attire</h5>
					<p style="color: #6c757d; font-size: 0.9rem;">Upload photo or use camera for AI analysis</p>
				</div>
				<div style="text-align: center; flex: 1; min-width: 200px;">
					<div style="background: #ffc107; color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold;">3</div>
					<h5 style="color: #2c3e50; margin-bottom: 0.5rem;">Get Results</h5>
					<p style="color: #6c757d; font-size: 0.9rem;">Instant compliance report with detailed analysis</p>
				</div>
				<div style="text-align: center; flex: 1; min-width: 200px;">
					<div style="background: #17a2b8; color: white; width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem; font-weight: bold;">4</div>
					<h5 style="color: #2c3e50; margin-bottom: 0.5rem;">Track Progress</h5>
					<p style="color: #6c757d; font-size: 0.9rem;">Monitor compliance history and statistics</p>
				</div>
			</div>
		</div>
		""", unsafe_allow_html=True)
		
		# Tips section
		with st.expander("💡 Pro Tips for Best Experience", expanded=False):
			col1, col2 = st.columns(2)
			with col1:
				st.markdown("""
				**🔐 Face Authentication Tips:**
				- Ensure good lighting (natural light preferred)
				- Position face in center of camera
				- Remove sunglasses and hats
				- Hold device steady during capture
				- Use emergency login if face auth fails
				""")
			with col2:
				st.markdown("""
				**📸 Attire Verification Tips:**
				- Capture full-body image when possible
				- Stand against plain background
				- Ensure uniform is clearly visible
				- Good lighting improves accuracy
				- Multiple angles can help verification
				""")


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
	
	try:
		bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
		pose = extract_pose(bgr)  # This will return None (MediaPipe disabled)
		
		# Use universal formal vs casual analysis
		st.info("🔍 Using universal formal/casual analysis")
		from src.verify import verify_attire_universal
		result = verify_attire_universal(bgr)
		
		# Create basic features with simple ID card detection
		features = extract_features_from_image(bgr, pose_landmarks=None, bins=cfg.hist_bins)
		
		# If feature extraction fails, create minimal features
		if not features:
			features = {
				"id_card_detected": 0.0,
				"id_card_confidence": 0.0,
				"image_height": bgr.shape[0],
				"image_width": bgr.shape[1]
			}
		
		# Create simple annotated image (no pose annotations since pose is None)
		annotated = bgr.copy()
		
		# Add violation indicators (works without pose)
		violations = result.get("violations", {}).get("violations", [])
		annotated = draw_violation_indicators(annotated, None, violations)
		
		# Add detailed badge
		annotated = overlay_detailed_badge(annotated, result)
			
	except Exception as e:
		st.error(f"❌ Error processing image: {str(e)}")
		st.info("💡 Using basic fallback...")
		try:
			# Try simple analysis as last resort
			bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
			from src.verify import verify_attire_universal
			result = verify_attire_universal(bgr)
			annotated = bgr.copy()  # Simple copy without annotations
			
			# Create basic features for ID card detection
			try:
				features = extract_features_from_image(bgr, pose_landmarks=None, bins=cfg.hist_bins)
			except:
				features = {
					"id_card_detected": 0.0,
					"id_card_confidence": 0.0,
					"image_height": bgr.shape[0],
					"image_width": bgr.shape[1]
				}
		except Exception as e2:
			st.error(f"❌ Could not analyze image: {str(e2)}")
			return {
				"status": "ERROR",
				"success_score": 0.0,
				"violations": {"violations": []},
				"event_id": None,
			}

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


@require_student_or_admin
def render_image_tab():
	"""Image verification tab - RBAC Protected"""
	st.subheader("Single Image")
	
	# RBAC: Check permissions for verification
	if not has_permission(Permission.SELF_VERIFICATION):
		show_permission_denied_message(required_permission="self_verification")
		return
	
	# Pre-fill student id when user is logged in (roll_no or id)
	_user = st.session_state.get('user') or {}
	prefill_id = _user.get('roll_no') or _user.get('student_id') or _user.get('id') or ""
	
	# RBAC: Students can only verify themselves, admins can verify anyone
	if is_student():
		student_id = prefill_id  # Force student to use their own ID
		st.info(f"🔒 **Student Mode:** Verifying as {prefill_id}")
		st.text_input("Your Student ID", value=prefill_id, disabled=True, key="student_id_image_display")
	else:
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


@require_student_or_admin
def render_webcam_tab():
	"""Webcam verification tab - RBAC Protected"""
	st.subheader("Webcam")
	
	# RBAC: Check permissions for verification
	if not has_permission(Permission.SELF_VERIFICATION):
		show_permission_denied_message(required_permission="self_verification")
		return
	
	_user = st.session_state.get('user') or {}
	prefill_id = _user.get('roll_no') or _user.get('student_id') or _user.get('id') or ""
	
	# RBAC: Students can only verify themselves, admins can verify anyone
	if is_student():
		student_id = prefill_id  # Force student to use their own ID
		st.info(f"🔒 **Student Mode:** Verifying as {prefill_id}")
		st.text_input("Your Student ID", value=prefill_id, disabled=True, key="student_id_webcam_display")
	else:
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


@require_student_or_admin
def render_video_tab():
	"""Video verification tab - RBAC Protected"""
	st.subheader("Video")
	
	# RBAC: Check permissions for verification
	if not has_permission(Permission.SELF_VERIFICATION):
		show_permission_denied_message(required_permission="self_verification")
		return
	
	_user = st.session_state.get('user') or {}
	prefill_id = _user.get('roll_no') or _user.get('student_id') or _user.get('id') or ""
	
	# RBAC: Students can only verify themselves, admins can verify anyone
	if is_student():
		student_id = prefill_id  # Force student to use their own ID
		st.info(f"🔒 **Student Mode:** Verifying as {prefill_id}")
		st.text_input("Your Student ID", value=prefill_id, disabled=True, key="student_id_video_display")
	else:
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
		# Use new clothing-focused verification
		from src.verify import verify_attire_and_safety_new
		result = verify_attire_and_safety_new(features, cfg, st.session_state.classifier)
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




@require_student_or_admin
def render_student_verification():
	"""Student verification hub - RBAC Protected"""
	# RBAC: Check permissions for verification
	if not has_permission(Permission.SELF_VERIFICATION):
		show_permission_denied_message(required_permission="self_verification")
		return
	
	# Modern header with role indication
	user = st.session_state.get('user', {})
	role_indicator = "👨‍💼 Admin Mode" if is_admin() else "🎓 Student Mode"
	
	st.markdown(f"""
	<div style="background: linear-gradient(135deg, #28a745 0%, #20c997 100%); padding: 2rem; margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px; text-align: center;">
		<h1 style="color: white; font-size: 2.2rem; margin-bottom: 0.5rem; font-weight: 600;">🎓 Attire Verification</h1>
		<p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0;">{role_indicator} - AI-Powered Compliance Analysis</p>
	</div>
	""", unsafe_allow_html=True)
	
	tabs = st.tabs(["📤 Upload Image", "📷 Live Camera", "🎥 Video Analysis"])
	with tabs[0]:
		render_image_tab()
	with tabs[1]:
		render_webcam_tab()
	with tabs[2]:
		render_video_tab()


@require_admin
def render_admin_tab():
	"""Admin Dashboard - Protected by RBAC"""
	log_access_attempt("admin_dashboard", "admin_role", True)
	
	st.markdown("""
	<div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 2rem; margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px; text-align: center;">
		<h1 style="color: white; font-size: 2.2rem; margin-bottom: 0.5rem; font-weight: 600;">👨‍💼 Admin Dashboard</h1>
		<p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0;">🔒 Secure Administrative Control Panel</p>
	</div>
	""", unsafe_allow_html=True)
	
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
		# RBAC: Check if user can manage students
		if not can_manage_students():
			show_permission_denied_message(required_permission="manage_students")
			return
		
		students = get_all_students(cfg=st.session_state.config)
		st.subheader("All Students")
		st.info("🔒 **Admin Only** - Student management requires administrative privileges")
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
			
			# Delete student section - Extra RBAC check
			if can_delete_data():
				st.markdown("#### 🗑️ Delete Student")
				st.error("⚠️ **DANGER ZONE:** Deleting a student will permanently remove all their data including events, face images, and login credentials.")
				st.warning("🔒 **Admin Only** - Data deletion requires highest level administrative privileges")
			
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
						# Final RBAC check before deletion
						if not can_delete_data():
							st.error("🚫 Access denied: You don't have permission to delete student data")
							return
						
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
				st.info("🔒 **Delete functionality restricted** - You don't have permission to delete student data")
		else:
			st.info("No students in database")
	
	with tab2:
		# RBAC: Check if user can view all reports
		if not can_view_all_reports():
			show_permission_denied_message(required_permission="view_all_reports")
			return
		
		st.subheader("Daily Compliance Report")
		st.info("🔒 **Admin Only** - System-wide reports require administrative privileges")
		compliance_df = pd.DataFrame([stats])
		st.dataframe(compliance_df)
		
		# Download button
		csv = compliance_df.to_csv(index=False).encode('utf-8')
		st.download_button("Download Compliance Report", csv, "compliance_report.csv", "text/csv")
	
	with tab3:
		# RBAC: Check if user can manage students
		if not can_manage_students():
			show_permission_denied_message(required_permission="manage_students")
			return
		
		st.subheader("Add/Update Student")
		st.info("🔒 **Admin Only** - Adding students requires administrative privileges")

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


@require_permission(Permission.MANAGE_DEPARTMENTS)
def render_add_department_tab():
	"""Render the Add Department form - Admin only"""
	st.subheader("➕ Create New Department")
	
	# RBAC Security Notice
	st.info("🔒 **Admin Only Feature** - Department management requires administrative privileges")

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


@require_permission(Permission.MANAGE_DEPARTMENTS)
def render_departments_tab():
	"""Render the Departments management view - Admin only"""
	st.subheader("📊 Departments Management")
	
	# RBAC Security Notice
	st.info("🔒 **Admin Only Feature** - Department management requires administrative privileges")
	
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
	"""Enterprise-grade sidebar navigation with advanced UI"""
	# Modern header with branding
	st.sidebar.markdown("""
	<div style="text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 1rem -1rem; border-radius: 0 0 10px 10px;">
		<h2 style="color: white; margin: 0; font-weight: 600;">🏫 SAVS</h2>
		<p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.8rem;">Student Attire Verification</p>
	</div>
	""", unsafe_allow_html=True)
	
	user = st.session_state.get('user')
	current_page = get_current_page()
	
	# User profile card
	if user:
		st.sidebar.markdown("""
		<div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #007bff;">
			<div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
				<div style="width: 40px; height: 40px; background: linear-gradient(135deg, #007bff, #0056b3); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 0.75rem;">
					<span style="color: white; font-weight: bold; font-size: 1.2rem;">👤</span>
				</div>
				<div>
					<div style="font-weight: 600; color: #2c3e50; font-size: 0.9rem;">""" + user.get('full_name', 'User')[:20] + """</div>
					<div style="color: #6c757d; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">""" + user.get('role', 'N/A').title() + """</div>
				</div>
			</div>
		</div>
		""", unsafe_allow_html=True)
		
		# Status indicators
		auth_method = user.get('auth_method', 'unknown')
		auth_color = "#28a745" if auth_method == "face" else "#ffc107" if auth_method == "emergency" else "#6c757d"
		st.sidebar.markdown(f"""
		<div style="display: flex; justify-content: space-between; margin-bottom: 1rem; font-size: 0.75rem;">
			<span style="color: {auth_color};">● {auth_method.title()} Auth</span>
			<span style="color: #6c757d;">Online</span>
		</div>
		""", unsafe_allow_html=True)
	
	# Navigation sections with RBAC-based filtering
	nav_sections = filter_navigation_by_role()
	
	# Render navigation sections
	for section_name, items in nav_sections:
		st.sidebar.markdown(f"""
		<div style="margin: 1.5rem 0 0.5rem 0; font-weight: 600; color: #495057; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">
			{section_name}
		</div>
		""", unsafe_allow_html=True)
		
		for label, page_id, icon in items:
			is_active = current_page == page_id
			button_style = """
			background: linear-gradient(135deg, #007bff, #0056b3); 
			color: white; 
			border: none; 
			box-shadow: 0 2px 4px rgba(0,123,255,0.3);
			""" if is_active else """
			background: white; 
			color: #495057; 
			border: 1px solid #dee2e6;
			"""
			
			hover_style = "transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0,0,0,0.1);" if not is_active else ""
			
			if st.sidebar.button(
				label, 
				key=f"nav_{page_id}", 
				use_container_width=True,
				help=f"Navigate to {label}"
			):
				navigate_to(page_id)
	
	# Action buttons
	if user:
		st.sidebar.markdown("---")
		st.sidebar.markdown("""
		<div style="margin: 1rem 0 0.5rem 0; font-weight: 600; color: #495057; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px;">
			⚡ Quick Actions
		</div>
		""", unsafe_allow_html=True)
		
		col1, col2 = st.sidebar.columns(2)
		with col1:
			if st.button("🔄 Refresh", use_container_width=True, help="Refresh current page"):
				st.rerun()
		with col2:
			if st.button("🚪 Logout", use_container_width=True, type="secondary", help="Logout securely"):
				logout_and_redirect()
	
	# Settings for admin (collapsible)
	if is_admin():
		with st.sidebar.expander("⚙️ System Settings", expanded=False):
			sidebar_settings()
	
	# Footer with system info and security status
	st.sidebar.markdown("---")
	
	# Security status indicator
	user_role = get_user_role().value if get_current_user() else "guest"
	role_color = {
		"admin": "#dc3545",
		"teacher": "#007bff", 
		"security_staff": "#ffc107",
		"student": "#28a745",
		"guest": "#6c757d"
	}.get(user_role, "#6c757d")
	
	st.sidebar.markdown(f"""
	<div style="text-align: center; color: #6c757d; font-size: 0.7rem; margin-top: 2rem;">
		<div>SAVS v2.0 Enterprise</div>
		<div>© 2024 RBAC Secured</div>
		<div style="margin-top: 0.5rem;">
			<span style="color: #28a745;">●</span> System Online
		</div>
		<div style="margin-top: 0.5rem; padding: 0.25rem; background: rgba(0,0,0,0.05); border-radius: 4px;">
			<span style="color: {role_color};">🔒</span> {user_role.title()} Mode
		</div>
	</div>
	""", unsafe_allow_html=True)


def main():
	st.set_page_config(page_title="Attire & Safety Verification", layout="wide")
	ensure_dirs()
	init_session_state()
	init_db()
	
	# Security: Initialize RBAC logging
	import logging
	logging.basicConfig(level=logging.INFO)
	rbac_logger = logging.getLogger('rbac_audit')
	rbac_logger.info("SAVS System started with RBAC enabled")

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
		
		# Show face authentication UI
		# The UI handles its own navigation after successful login
		show_face_authentication(st.session_state.config)
	
	elif current_page == "register":
		user = show_registration_form(st.session_state.config)
		# Registration redirects to face_auth automatically
	
	elif current_page == "student_dashboard":
		# RBAC: Student dashboard requires login and self-verification permission
		if not has_permission(Permission.VIEW_OWN_REPORTS):
			log_access_attempt("student_dashboard", "student_access", False)
			show_permission_denied_message(required_permission="view_own_reports")
		else:
			show_student_dashboard(st.session_state.config)
	
	elif current_page == "verification":
		# RBAC: Verification requires self-verification permission
		if not has_permission(Permission.SELF_VERIFICATION):
			log_access_attempt("verification", "self_verification", False)
			show_permission_denied_message(required_permission="self_verification")
		else:
			render_student_verification()
	
	elif current_page == "admin_login":
		st.markdown("""
		<div style="background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); padding: 2rem; margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 20px 20px; text-align: center;">
			<h1 style="color: white; font-size: 2.2rem; margin-bottom: 0.5rem; font-weight: 600;">👨‍💼 Admin Access</h1>
			<p style="color: rgba(255,255,255,0.9); font-size: 1.1rem; margin: 0;">🔒 Secure Administrative Authentication</p>
		</div>
		""", unsafe_allow_html=True)
		
		st.error("🔒 **RESTRICTED AREA** - Administrator Access Only")
		st.warning("⚠️ This area contains sensitive system management functions.")
		st.info("💡 Please enter valid administrator credentials to continue.")
		
		# Security notice
		with st.expander("🛡️ Security Information", expanded=False):
			st.markdown("""
			**Admin Access Includes:**
			- 👥 Complete student management (add, edit, delete)
			- 🏢 Department and class administration  
			- 📊 System-wide analytics and reporting
			- ⚙️ Configuration and policy settings
			- 🗑️ Data deletion capabilities
			- 📥 Export and backup functions
			
			**Security Features:**
			- 🔐 Role-based access control (RBAC)
			- 📝 Audit logging of all actions
			- 🚫 Automatic permission validation
			- 🔄 Session timeout protection
			""")
		
		st.markdown("---")

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
		# RBAC: Admin dashboard requires admin permissions
		if not has_permission(Permission.SYSTEM_SETTINGS):
			log_access_attempt("admin_dashboard", "admin_role", False)
			show_permission_denied_message(required_role="admin")
		else:
			render_admin_tab()
	
	elif current_page == "profile":
		# RBAC: Profile requires login and view own profile permission
		if not has_permission(Permission.VIEW_OWN_PROFILE):
			log_access_attempt("profile", "view_own_profile", False)
			show_permission_denied_message(required_permission="view_own_profile")
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
