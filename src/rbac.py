"""
Role-Based Access Control (RBAC) Module
Provides enterprise-grade access control and permission management
"""

import functools
import streamlit as st
from typing import List, Optional, Callable, Any
from enum import Enum


class Role(Enum):
    """User roles with hierarchical permissions"""
    ADMIN = "admin"
    TEACHER = "teacher"
    SECURITY_STAFF = "security_staff"
    STUDENT = "student"
    GUEST = "guest"


class Permission(Enum):
    """System permissions"""
    # Admin permissions
    MANAGE_STUDENTS = "manage_students"
    MANAGE_DEPARTMENTS = "manage_departments"
    MANAGE_USERS = "manage_users"
    VIEW_ALL_REPORTS = "view_all_reports"
    SYSTEM_SETTINGS = "system_settings"
    DELETE_DATA = "delete_data"
    EXPORT_DATA = "export_data"
    
    # Teacher permissions
    VIEW_CLASS_REPORTS = "view_class_reports"
    MANAGE_CLASS_STUDENTS = "manage_class_students"
    
    # Security staff permissions
    MONITOR_COMPLIANCE = "monitor_compliance"
    VIEW_SECURITY_ALERTS = "view_security_alerts"
    
    # Student permissions
    SELF_VERIFICATION = "self_verification"
    VIEW_OWN_PROFILE = "view_own_profile"
    VIEW_OWN_REPORTS = "view_own_reports"
    
    # General permissions
    LOGIN = "login"
    REGISTER = "register"


# Role-Permission mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [
        Permission.MANAGE_STUDENTS,
        Permission.MANAGE_DEPARTMENTS,
        Permission.MANAGE_USERS,
        Permission.VIEW_ALL_REPORTS,
        Permission.SYSTEM_SETTINGS,
        Permission.DELETE_DATA,
        Permission.EXPORT_DATA,
        Permission.VIEW_CLASS_REPORTS,
        Permission.MANAGE_CLASS_STUDENTS,
        Permission.MONITOR_COMPLIANCE,
        Permission.VIEW_SECURITY_ALERTS,
        Permission.SELF_VERIFICATION,
        Permission.VIEW_OWN_PROFILE,
        Permission.VIEW_OWN_REPORTS,
        Permission.LOGIN,
    ],
    Role.TEACHER: [
        Permission.VIEW_CLASS_REPORTS,
        Permission.MANAGE_CLASS_STUDENTS,
        Permission.MONITOR_COMPLIANCE,
        Permission.SELF_VERIFICATION,
        Permission.VIEW_OWN_PROFILE,
        Permission.VIEW_OWN_REPORTS,
        Permission.LOGIN,
    ],
    Role.SECURITY_STAFF: [
        Permission.MONITOR_COMPLIANCE,
        Permission.VIEW_SECURITY_ALERTS,
        Permission.VIEW_CLASS_REPORTS,
        Permission.SELF_VERIFICATION,
        Permission.VIEW_OWN_PROFILE,
        Permission.LOGIN,
    ],
    Role.STUDENT: [
        Permission.SELF_VERIFICATION,
        Permission.VIEW_OWN_PROFILE,
        Permission.VIEW_OWN_REPORTS,
        Permission.LOGIN,
    ],
    Role.GUEST: [
        Permission.REGISTER,
    ]
}


def get_current_user() -> Optional[dict]:
    """Get current user from session state"""
    return st.session_state.get('user')


def get_user_role() -> Role:
    """Get current user's role"""
    user = get_current_user()
    if not user:
        return Role.GUEST
    
    role_str = user.get('role', 'guest').lower()
    try:
        return Role(role_str)
    except ValueError:
        return Role.GUEST


def has_permission(permission: Permission) -> bool:
    """Check if current user has specific permission"""
    user_role = get_user_role()
    return permission in ROLE_PERMISSIONS.get(user_role, [])


def has_any_permission(permissions: List[Permission]) -> bool:
    """Check if current user has any of the specified permissions"""
    return any(has_permission(perm) for perm in permissions)


def has_all_permissions(permissions: List[Permission]) -> bool:
    """Check if current user has all specified permissions"""
    return all(has_permission(perm) for perm in permissions)


def is_admin() -> bool:
    """Check if current user is admin"""
    return get_user_role() == Role.ADMIN


def is_student() -> bool:
    """Check if current user is student"""
    return get_user_role() == Role.STUDENT


def is_teacher() -> bool:
    """Check if current user is teacher"""
    return get_user_role() == Role.TEACHER


def is_security_staff() -> bool:
    """Check if current user is security staff"""
    return get_user_role() == Role.SECURITY_STAFF


def is_logged_in() -> bool:
    """Check if user is logged in"""
    return get_current_user() is not None


def require_login(func: Callable) -> Callable:
    """Decorator to require user login"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not is_logged_in():
            st.error("🔒 **Access Denied**")
            st.warning("You must be logged in to access this feature.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Login", use_container_width=True, type="primary"):
                    st.session_state.current_page = 'face_auth'
                    st.rerun()
            with col2:
                if st.button("🏠 Home", use_container_width=True):
                    st.session_state.current_page = 'home'
                    st.rerun()
            return None
        return func(*args, **kwargs)
    return wrapper


def require_permission(permission: Permission, redirect_page: str = "home"):
    """Decorator to require specific permission"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not has_permission(permission):
                user = get_current_user()
                user_role = get_user_role().value if user else "guest"
                
                st.error("🚫 **Access Denied**")
                st.warning(f"Your role ({user_role.title()}) does not have permission to access this feature.")
                st.info(f"**Required Permission:** {permission.value.replace('_', ' ').title()}")
                
                # Show what they can access instead
                available_permissions = ROLE_PERMISSIONS.get(get_user_role(), [])
                if available_permissions:
                    with st.expander("✅ Your Available Permissions", expanded=False):
                        for perm in available_permissions:
                            st.write(f"• {perm.value.replace('_', ' ').title()}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🏠 Go to Home", use_container_width=True, type="primary"):
                        st.session_state.current_page = 'home'
                        st.rerun()
                with col2:
                    if user_role == "guest":
                        if st.button("🔐 Login", use_container_width=True):
                            st.session_state.current_page = 'face_auth'
                            st.rerun()
                    else:
                        if st.button("📋 My Dashboard", use_container_width=True):
                            if is_student():
                                st.session_state.current_page = 'student_dashboard'
                            elif is_admin():
                                st.session_state.current_page = 'admin_dashboard'
                            else:
                                st.session_state.current_page = 'home'
                            st.rerun()
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_any_permission(permissions: List[Permission], redirect_page: str = "home"):
    """Decorator to require any of the specified permissions"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not has_any_permission(permissions):
                user = get_current_user()
                user_role = get_user_role().value if user else "guest"
                
                st.error("🚫 **Access Denied**")
                st.warning(f"Your role ({user_role.title()}) does not have the required permissions.")
                
                perm_names = [p.value.replace('_', ' ').title() for p in permissions]
                st.info(f"**Required Permissions (any):** {', '.join(perm_names)}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🏠 Go to Home", use_container_width=True, type="primary"):
                        st.session_state.current_page = 'home'
                        st.rerun()
                with col2:
                    if st.button("📋 Dashboard", use_container_width=True):
                        if is_student():
                            st.session_state.current_page = 'student_dashboard'
                        elif is_admin():
                            st.session_state.current_page = 'admin_dashboard'
                        else:
                            st.session_state.current_page = 'home'
                        st.rerun()
                return None
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_admin(func: Callable) -> Callable:
    """Decorator to require admin role"""
    return require_permission(Permission.SYSTEM_SETTINGS)(func)


def require_student_or_admin(func: Callable) -> Callable:
    """Decorator to require student or admin role"""
    return require_any_permission([Permission.SELF_VERIFICATION, Permission.SYSTEM_SETTINGS])(func)


def check_data_access_permission(student_id: str) -> bool:
    """Check if current user can access specific student's data"""
    user = get_current_user()
    if not user:
        return False
    
    # Admin can access all data
    if is_admin():
        return True
    
    # Students can only access their own data
    if is_student():
        user_student_id = (
            user.get('student_id') or 
            user.get('id') or 
            user.get('username') or 
            user.get('roll_no')
        )
        return str(user_student_id) == str(student_id)
    
    # Teachers can access their class students (TODO: implement class-based access)
    if is_teacher():
        return True  # For now, allow teachers to access all student data
    
    # Security staff can access all data for monitoring
    if is_security_staff():
        return True
    
    return False


def filter_navigation_by_role() -> List[tuple]:
    """Filter navigation items based on user role"""
    user_role = get_user_role()
    
    if user_role == Role.GUEST:
        return [
            ("🚀 Quick Start", [
                ("🏠 Home", "home", "🏠"),
                ("🎓 Student Portal", "student_portal", "🎓"),
                ("👨‍💼 Admin Access", "admin_login", "👨‍💼"),
            ])
        ]
    
    elif user_role == Role.ADMIN:
        return [
            ("📊 Administration", [
                ("🏠 Home", "home", "🏠"),
                ("📊 Admin Dashboard", "admin_dashboard", "📊"),
                ("🎓 Verification Hub", "verification", "🎓"),
            ]),
            ("👥 User Management", [
                ("🎓 Student Portal", "student_portal", "🎓"),
                ("👤 My Profile", "profile", "👤"),
            ])
        ]
    
    elif user_role == Role.STUDENT:
        return [
            ("🎓 Student Hub", [
                ("🏠 Home", "home", "🏠"),
                ("🎓 Student Portal", "student_portal", "🎓"),
                ("📋 My Dashboard", "student_dashboard", "📋"),
            ]),
            ("👤 Personal", [
                ("👤 My Profile", "profile", "👤"),
            ])
        ]
    
    elif user_role == Role.TEACHER:
        return [
            ("📚 Teaching", [
                ("🏠 Home", "home", "🏠"),
                ("📊 Class Reports", "class_reports", "📊"),
                ("🎓 Verification Hub", "verification", "🎓"),
            ]),
            ("👤 Personal", [
                ("👤 My Profile", "profile", "👤"),
            ])
        ]
    
    elif user_role == Role.SECURITY_STAFF:
        return [
            ("🛡️ Security", [
                ("🏠 Home", "home", "🏠"),
                ("🚨 Security Alerts", "security_alerts", "🚨"),
                ("📊 Compliance Monitor", "compliance_monitor", "📊"),
            ]),
            ("👤 Personal", [
                ("👤 My Profile", "profile", "👤"),
            ])
        ]
    
    else:
        return [
            ("🚀 Get Started", [
                ("🏠 Home", "home", "🏠"),
                ("🎓 Student Portal", "student_portal", "🎓"),
                ("👨‍💼 Admin Access", "admin_login", "👨‍💼"),
            ])
        ]


def show_permission_denied_message(required_permission: str = None, required_role: str = None):
    """Show a standardized permission denied message"""
    user = get_current_user()
    current_role = get_user_role().value if user else "guest"
    
    st.error("🚫 **Access Denied**")
    
    if required_role:
        st.warning(f"This feature requires **{required_role.title()}** role. Your current role: **{current_role.title()}**")
    elif required_permission:
        st.warning(f"This feature requires **{required_permission.replace('_', ' ').title()}** permission.")
    else:
        st.warning("You don't have permission to access this feature.")
    
    # Show helpful actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Home", use_container_width=True, type="primary"):
            st.session_state.current_page = 'home'
            st.rerun()
    
    with col2:
        if current_role == "guest":
            if st.button("🔐 Login", use_container_width=True):
                st.session_state.current_page = 'face_auth'
                st.rerun()
        else:
            if st.button("📋 Dashboard", use_container_width=True):
                if is_student():
                    st.session_state.current_page = 'student_dashboard'
                elif is_admin():
                    st.session_state.current_page = 'admin_dashboard'
                else:
                    st.session_state.current_page = 'home'
                st.rerun()
    
    with col3:
        if st.button("ℹ️ Help", use_container_width=True):
            with st.expander("🔍 Role & Permission Guide", expanded=True):
                st.markdown("""
                **👨‍💼 Admin:** Full system access, manage students, departments, settings
                
                **👩‍🏫 Teacher:** Manage assigned classes, view class reports
                
                **🛡️ Security:** Monitor compliance, view security alerts
                
                **🎓 Student:** Personal verification, view own reports and profile
                
                **👤 Guest:** Register for new account, view public pages
                """)


def log_access_attempt(page: str, permission: str = None, success: bool = True):
    """Log access attempts for security auditing"""
    user = get_current_user()
    user_id = user.get('id', 'anonymous') if user else 'anonymous'
    user_role = get_user_role().value
    
    # In a production system, this would log to a security audit database
    import logging
    logger = logging.getLogger('rbac_audit')
    
    log_message = f"Access {'GRANTED' if success else 'DENIED'}: User {user_id} ({user_role}) -> {page}"
    if permission:
        log_message += f" [Required: {permission}]"
    
    if success:
        logger.info(log_message)
    else:
        logger.warning(log_message)


# Convenience functions for common permission checks
def can_manage_students() -> bool:
    """Check if user can manage students"""
    return has_permission(Permission.MANAGE_STUDENTS)


def can_manage_departments() -> bool:
    """Check if user can manage departments"""
    return has_permission(Permission.MANAGE_DEPARTMENTS)


def can_view_all_reports() -> bool:
    """Check if user can view all reports"""
    return has_permission(Permission.VIEW_ALL_REPORTS)


def can_modify_system_settings() -> bool:
    """Check if user can modify system settings"""
    return has_permission(Permission.SYSTEM_SETTINGS)


def can_delete_data() -> bool:
    """Check if user can delete data"""
    return has_permission(Permission.DELETE_DATA)


def can_export_data() -> bool:
    """Check if user can export data"""
    return has_permission(Permission.EXPORT_DATA)


def can_self_verify() -> bool:
    """Check if user can perform self verification"""
    return has_permission(Permission.SELF_VERIFICATION)


def can_view_own_profile() -> bool:
    """Check if user can view their own profile"""
    return has_permission(Permission.VIEW_OWN_PROFILE)