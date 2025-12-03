import hashlib
import secrets
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

from .config import AppConfig
from .db import get_conn


def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        if not hashed_password or ':' not in hashed_password:
            return False
        
        parts = hashed_password.split(":", 1)
        if len(parts) != 2:
            return False
        
        salt, stored_hash = parts
        computed_hash = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
        return computed_hash == stored_hash
    except (ValueError, TypeError) as e:
        logger.error(f"Password verification error: {e}")
        return False


def authenticate_user(username: str, password: str, cfg: AppConfig | None = None) -> Optional[Dict]:
    """Authenticate user and return user info if successful"""
    from .validation import validate_username, sanitize_string
    import logging
    logger = logging.getLogger(__name__)
    
    # Validate and sanitize inputs
    if not username or not password:
        logger.warning("Empty username or password")
        return None
    
    username = sanitize_string(username, 30)
    
    if not validate_username(username):
        logger.warning(f"Invalid username format: {username}")
        return None
    
    # Check for hardcoded admin credentials
    if username == "admin" and password == "admin123":
        return {
            'username': 'admin',
            'role': 'admin',
            'full_name': 'System Administrator',
            'email': 'admin@system.com'
        }

    conn = get_conn(cfg)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row and verify_password(password, row['password']):
            user = dict(row)
            # Remove password from returned data
            user.pop('password', None)
            return user
    finally:
        conn.close()

    return None


def register_student(student_data: Dict, cfg: AppConfig | None = None) -> bool:
	"""Register a new student user with face biometric data"""
	import logging
	from .validation import (validate_email, validate_student_id, validate_username, 
	                         validate_password, validate_name, sanitize_string)
	logger = logging.getLogger(__name__)
	
	required_fields = ['full_name', 'email', 'student_id']
	for field in required_fields:
		if not student_data.get(field):
			logger.error(f"Missing required field: {field}")
			return False

	# Validate inputs
	if not validate_name(student_data['full_name']):
		logger.error("Invalid full name format")
		return False
	
	if not validate_email(student_data['email']):
		logger.error("Invalid email format")
		return False
	
	if not validate_student_id(student_data['student_id']):
		logger.error("Invalid student ID format")
		return False
	
	if not validate_username(student_data.get('username', '')):
		logger.error("Invalid username format")
		return False

	# Check if password exists and validate
	if not student_data.get('password'):
		logger.error("Missing password field")
		return False
	
	is_valid, error_msg = validate_password(student_data['password'])
	if not is_valid:
		logger.error(f"Invalid password: {error_msg}")
		return False

	# Sanitize string inputs
	student_data['full_name'] = sanitize_string(student_data['full_name'], 100)
	student_data['email'] = sanitize_string(student_data['email'], 255)
	student_data['student_id'] = sanitize_string(student_data['student_id'], 20)

	# Hash password
	hashed_password = hash_password(student_data['password'])

	conn = get_conn(cfg)
	try:
		with conn:
			# Add to users table
			logger.info(f"Registering user: {student_data['username']}")
			conn.execute(
				"INSERT INTO users (username, password, role, full_name, email, assigned_class) VALUES (?,?,?,?,?,?)",
				(
					student_data['username'],
					hashed_password,
					'student',
					student_data['full_name'],
					student_data['email'],
					student_data.get('class', '')
				)
			)

			# Add to students table - start with basic columns that definitely exist
			conn.execute(
				"""INSERT INTO students
				   (id, name, class, department, email, phone, contact_info, verified)
				   VALUES (?,?,?,?,?,?,?,?)""",
				(
					student_data['student_id'],
					student_data['full_name'],
					student_data.get('class', ''),
					student_data.get('department', ''),
					student_data['email'],
					student_data.get('phone', ''),
					student_data.get('contact_info', ''),
					0  # Initially unverified, set to 1 after face capture
				)
			)

			# Update additional columns that may have been added via migrations
			# These updates are safe even if columns don't exist (they'll be ignored)
			try:
				if student_data.get('gender'):
					conn.execute("UPDATE students SET gender = ? WHERE id = ?",
							   (student_data.get('gender', 'U'), student_data['student_id']))
			except sqlite3.OperationalError:
				pass  # Column might not exist yet

			try:
				if student_data.get('roll_no'):
					conn.execute("UPDATE students SET roll_no = ? WHERE id = ?",
							   (student_data.get('roll_no', student_data['student_id']), student_data['student_id']))
			except sqlite3.OperationalError:
				pass  # Column might not exist yet

			try:
				if student_data.get('face_hash'):
					conn.execute("UPDATE students SET face_hash = ? WHERE id = ?",
							   (student_data.get('face_hash', ''), student_data['student_id']))
			except sqlite3.OperationalError:
				pass  # Column might not exist yet

			try:
				if student_data.get('face_image_path'):
					conn.execute("UPDATE students SET face_image_path = ? WHERE id = ?",
							   (student_data.get('face_image_path', ''), student_data['student_id']))
			except sqlite3.OperationalError:
				pass  # Column might not exist yet

		logger.info(f"Registration successful for student: {student_data['student_id']}")
		return True
	except sqlite3.IntegrityError as e:
		# Username or student_id already exists
		logger.error(f"Registration IntegrityError: {e}")
		return False
	except sqlite3.Error as e:
		# Database errors
		logger.error(f"Registration database error: {e}")
		return False
	finally:
		conn.close()


def check_username_exists(username: str, cfg: AppConfig | None = None) -> bool:
    """Check if username already exists"""
    from .validation import sanitize_string
    
    if not username:
        return False
    
    username = sanitize_string(username, 30)
    
    conn = get_conn(cfg)
    try:
        row = conn.execute("SELECT username FROM users WHERE username=?", (username,)).fetchone()
        return row is not None
    finally:
        conn.close()


def check_student_id_exists(student_id: str, cfg: AppConfig | None = None) -> bool:
    """Check if student ID already exists"""
    from .validation import sanitize_string
    
    if not student_id:
        return False
    
    student_id = sanitize_string(student_id, 20)
    
    conn = get_conn(cfg)
    try:
        row = conn.execute("SELECT id FROM students WHERE id=?", (student_id,)).fetchone()
        return row is not None
    finally:
        conn.close()


def get_current_user(session_state) -> Optional[Dict]:
    """Get current logged in user from session"""
    if 'user' in session_state and session_state['user']:
        # Check if session is still valid (optional: add expiration check)
        return session_state['user']
    return None


def login_user(user_data: Dict, session_state) -> None:
    """Log in user by storing in session"""
    session_state['user'] = user_data
    session_state['login_time'] = datetime.now()


def logout_user(session_state) -> None:
    """Log out user by clearing session"""
    if 'user' in session_state:
        del session_state['user']
    if 'login_time' in session_state:
        del session_state['login_time']


def require_auth(session_state, allowed_roles: list = None) -> bool:
    """Check if user is authenticated and has required role"""
    user = get_current_user(session_state)
    if not user:
        return False

    if allowed_roles and user.get('role') not in allowed_roles:
        return False

    return True


def is_admin(session_state) -> bool:
    """Check if current user is admin"""
    user = get_current_user(session_state)
    return user and user.get('role') == 'admin'


def is_student(session_state) -> bool:
    """Check if current user is student"""
    user = get_current_user(session_state)
    return user and user.get('role') == 'student'


def is_teacher(session_state) -> bool:
    """Check if current user is teacher"""
    user = get_current_user(session_state)
    return user and user.get('role') == 'teacher'


def is_security_staff(session_state) -> bool:
    """Check if current user is security staff"""
    user = get_current_user(session_state)
    return user and user.get('role') == 'security_staff'
