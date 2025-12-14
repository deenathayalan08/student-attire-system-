"""
Security Validation Module
Validates RBAC implementation and security boundaries
"""

import streamlit as st
from typing import Dict, List, Tuple
from .rbac import (
    Role, Permission, get_user_role, has_permission, 
    is_admin, is_student, get_current_user
)


def validate_rbac_implementation() -> Dict[str, bool]:
    """Validate that RBAC is properly implemented"""
    results = {}
    
    # Test 1: Role detection
    current_role = get_user_role()
    results['role_detection'] = current_role is not None
    
    # Test 2: Permission system
    try:
        # Test basic permission check
        test_perm = has_permission(Permission.LOGIN)
        results['permission_system'] = True
    except Exception:
        results['permission_system'] = False
    
    # Test 3: Admin detection
    try:
        admin_status = is_admin()
        results['admin_detection'] = isinstance(admin_status, bool)
    except Exception:
        results['admin_detection'] = False
    
    # Test 4: Student detection  
    try:
        student_status = is_student()
        results['student_detection'] = isinstance(student_status, bool)
    except Exception:
        results['student_detection'] = False
    
    return results


def get_security_status() -> Dict[str, any]:
    """Get comprehensive security status"""
    user = get_current_user()
    role = get_user_role()
    
    return {
        'authenticated': user is not None,
        'user_id': user.get('id', 'anonymous') if user else 'anonymous',
        'role': role.value,
        'permissions': get_user_permissions(),
        'session_valid': validate_session(),
        'rbac_status': validate_rbac_implementation()
    }


def get_user_permissions() -> List[str]:
    """Get list of current user's permissions"""
    from .rbac import ROLE_PERMISSIONS
    role = get_user_role()
    permissions = ROLE_PERMISSIONS.get(role, [])
    return [perm.value for perm in permissions]


def validate_session() -> bool:
    """Validate current session security"""
    user = get_current_user()
    if not user:
        return True  # Guest sessions are valid
    
    # Check required fields
    required_fields = ['role']
    for field in required_fields:
        if field not in user:
            return False
    
    # Validate role
    try:
        Role(user['role'])
        return True
    except ValueError:
        return False


def check_page_access(page: str) -> Tuple[bool, str]:
    """Check if current user can access a specific page"""
    role = get_user_role()
    
    # Define page access rules
    page_permissions = {
        'home': [Permission.LOGIN],  # Everyone can access home
        'student_portal': [Permission.LOGIN],  # Everyone can access portal
        'admin_dashboard': [Permission.SYSTEM_SETTINGS],
        'student_dashboard': [Permission.VIEW_OWN_REPORTS],
        'verification': [Permission.SELF_VERIFICATION],
        'profile': [Permission.VIEW_OWN_PROFILE],
        'admin_login': [],  # No specific permission needed to view login
        'register': [Permission.REGISTER],
        'face_auth': [Permission.LOGIN]
    }
    
    required_perms = page_permissions.get(page, [])
    
    # Guest can access registration and login pages
    if role == Role.GUEST and page in ['register', 'face_auth', 'admin_login', 'home', 'student_portal']:
        return True, "Guest access allowed"
    
    # Check if user has any required permission
    if not required_perms:
        return True, "No permissions required"
    
    for perm in required_perms:
        if has_permission(perm):
            return True, f"Access granted via {perm.value}"
    
    return False, f"Missing required permissions: {[p.value for p in required_perms]}"


def show_security_dashboard():
    """Display security status dashboard for admins"""
    if not is_admin():
        st.error("🚫 Admin access required for security dashboard")
        return
    
    st.subheader("🛡️ Security Status Dashboard")
    
    # Overall security status
    status = get_security_status()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        auth_status = "✅ Authenticated" if status['authenticated'] else "❌ Not Authenticated"
        st.metric("Authentication", auth_status)
    
    with col2:
        st.metric("Current Role", status['role'].title())
    
    with col3:
        perm_count = len(status['permissions'])
        st.metric("Permissions", f"{perm_count} granted")
    
    with col4:
        session_status = "✅ Valid" if status['session_valid'] else "❌ Invalid"
        st.metric("Session", session_status)
    
    # RBAC validation results
    st.markdown("---")
    st.markdown("#### 🔍 RBAC System Validation")
    
    rbac_status = status['rbac_status']
    for test_name, result in rbac_status.items():
        status_icon = "✅" if result else "❌"
        st.write(f"{status_icon} **{test_name.replace('_', ' ').title()}**: {'Passed' if result else 'Failed'}")
    
    # Current user permissions
    st.markdown("---")
    st.markdown("#### 🔑 Current User Permissions")
    
    permissions = status['permissions']
    if permissions:
        cols = st.columns(3)
        for i, perm in enumerate(permissions):
            cols[i % 3].write(f"✓ {perm.replace('_', ' ').title()}")
    else:
        st.info("No permissions granted")
    
    # Page access validation
    st.markdown("---")
    st.markdown("#### 📄 Page Access Validation")
    
    test_pages = ['home', 'admin_dashboard', 'student_dashboard', 'verification', 'profile']
    
    for page in test_pages:
        can_access, reason = check_page_access(page)
        status_icon = "✅" if can_access else "❌"
        st.write(f"{status_icon} **{page.replace('_', ' ').title()}**: {reason}")


def log_security_event(event_type: str, details: str, severity: str = "INFO"):
    """Log security-related events"""
    import logging
    
    logger = logging.getLogger('security_audit')
    user = get_current_user()
    user_id = user.get('id', 'anonymous') if user else 'anonymous'
    role = get_user_role().value
    
    log_message = f"SECURITY [{severity}] {event_type}: {details} | User: {user_id} ({role})"
    
    if severity == "ERROR":
        logger.error(log_message)
    elif severity == "WARNING":
        logger.warning(log_message)
    else:
        logger.info(log_message)


def validate_data_access(requested_student_id: str, requesting_user_id: str) -> bool:
    """Validate if a user can access specific student data"""
    user = get_current_user()
    if not user:
        log_security_event("DATA_ACCESS_DENIED", f"Unauthenticated access attempt for student {requested_student_id}", "WARNING")
        return False
    
    # Admin can access all data
    if is_admin():
        log_security_event("DATA_ACCESS_GRANTED", f"Admin access to student {requested_student_id}", "INFO")
        return True
    
    # Students can only access their own data
    if is_student():
        user_student_id = user.get('student_id') or user.get('id') or user.get('username')
        if str(user_student_id) == str(requested_student_id):
            log_security_event("DATA_ACCESS_GRANTED", f"Student self-access to {requested_student_id}", "INFO")
            return True
        else:
            log_security_event("DATA_ACCESS_DENIED", f"Student {user_student_id} attempted to access {requested_student_id}", "WARNING")
            return False
    
    log_security_event("DATA_ACCESS_DENIED", f"Unknown role attempted to access student {requested_student_id}", "ERROR")
    return False


def security_middleware():
    """Security middleware to run on every page load"""
    # Validate session
    if not validate_session():
        st.error("🚫 Invalid session detected. Please login again.")
        st.session_state.clear()
        st.rerun()
    
    # Log page access
    current_page = st.session_state.get('current_page', 'unknown')
    user = get_current_user()
    user_id = user.get('id', 'anonymous') if user else 'anonymous'
    
    log_security_event("PAGE_ACCESS", f"User {user_id} accessed {current_page}", "INFO")


# Security constants
SECURITY_HEADERS = {
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
}

SENSITIVE_OPERATIONS = [
    'delete_student',
    'add_department', 
    'update_department',
    'delete_department',
    'export_data',
    'system_settings'
]