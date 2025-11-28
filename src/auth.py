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
    try:
        salt, stored_hash = hashed_password.split(":")
        computed_hash = hashlib.sha256(f"{password}{salt}".encode()).hexdigest()
        return computed_hash == stored_hash
    except:
        return False


def authenticate_user(username: str, password: str, cfg: AppConfig | None = None) -> Optional[Dict]:
    """Authenticate user and return user info if successful"""
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
    """Register a new student user"""
    required_fields = ['username', 'password', 'full_name', 'email', 'student_id', 'class_name']
    for field in required_fields:
        if not student_data.get(field):
            return False

    # Hash password
    hashed_password = hash_password(student_data['password'])

    conn = get_conn(cfg)
    try:
        with conn:
            # Add to users table
            conn.execute(
                "INSERT INTO users (username, password, role, full_name, email, assigned_class) VALUES (?,?,?,?,?,?)",
                (
                    student_data['username'],
                    hashed_password,
                    'student',
                    student_data['full_name'],
                    student_data['email'],
                    student_data['class_name']
                )
            )

            # Add to students table
            conn.execute(
                "INSERT INTO students (id, name, class, department, email, phone) VALUES (?,?,?,?,?,?)",
                (
                    student_data['student_id'],
                    student_data['full_name'],
                    student_data['class_name'],
                    student_data.get('department', ''),
                    student_data['email'],
                    student_data.get('phone', '')
                )
            )
        return True
    except sqlite3.IntegrityError:
        # Username or student_id already exists
        return False
    finally:
        conn.close()


def check_username_exists(username: str, cfg: AppConfig | None = None) -> bool:
    """Check if username already exists"""
    conn = get_conn(cfg)
    try:
        row = conn.execute("SELECT username FROM users WHERE username=?", (username,)).fetchone()
        return row is not None
    finally:
        conn.close()


def check_student_id_exists(student_id: str, cfg: AppConfig | None = None) -> bool:
    """Check if student ID already exists"""
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
