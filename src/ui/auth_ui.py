import streamlit as st
from typing import Optional, Dict, Any

from ..auth import authenticate_user, register_student, check_username_exists, check_student_id_exists
from ..config import AppConfig


def show_login_form(cfg: AppConfig) -> Optional[Dict]:
    """Display login form and return user data if successful"""
    st.title("🔐 Login")

    with st.form("login_form"):
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")

        col1, col2 = st.columns([1, 1])
        with col1:
            login_button = st.form_submit_button("Login", use_container_width=True)
        with col2:
            if st.form_submit_button("Forgot Password?", use_container_width=True):
                st.info("Please contact your administrator to reset your password.")

    if login_button:
        if not username or not password:
            st.error("Please enter both username and password.")
            return None

        user = authenticate_user(username, password, cfg)
        if user:
            st.success(f"Welcome back, {user['full_name']}!")
            return user
        else:
            st.error("Invalid username or password.")
            return None

    return None


def show_registration_form(cfg: AppConfig) -> Optional[Dict]:
    """Display student registration form and return user data if successful"""
    st.title("📝 Student Registration")

    with st.form("registration_form"):
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input("Full Name *", key="reg_full_name")
            email = st.text_input("Email *", key="reg_email")
            phone = st.text_input("Phone", key="reg_phone")

        with col2:
            student_id = st.text_input("Student ID *", key="reg_student_id")
            class_name = st.text_input("Class *", key="reg_class")
            department = st.text_input("Department", key="reg_department")

        st.markdown("### Account Information")
        col3, col4 = st.columns(2)

        with col3:
            username = st.text_input("Username *", key="reg_username")
            password = st.text_input("Password *", type="password", key="reg_password")

        with col4:
            confirm_password = st.text_input("Confirm Password *", type="password", key="reg_confirm_password")

        # Terms and conditions
        agree_terms = st.checkbox("I agree to the terms and conditions", key="reg_agree_terms")

        submitted = st.form_submit_button("Register", use_container_width=True, type="primary")

    if submitted:
        # Validation
        if not all([full_name, email, student_id, class_name, username, password]):
            st.error("Please fill in all required fields marked with *.")
            return None

        if password != confirm_password:
            st.error("Passwords do not match.")
            return None

        if len(password) < 6:
            st.error("Password must be at least 6 characters long.")
            return None

        if not agree_terms:
            st.error("Please agree to the terms and conditions.")
            return None

        # Check if username or student ID already exists
        if check_username_exists(username, cfg):
            st.error("Username already exists. Please choose a different username.")
            return None

        if check_student_id_exists(student_id, cfg):
            st.error("Student ID already exists. Please contact administrator if this is an error.")
            return None

        # Attempt registration
        student_data = {
            'username': username,
            'password': password,
            'full_name': full_name,
            'email': email,
            'student_id': student_id,
            'class_name': class_name,
            'department': department,
            'phone': phone
        }

        if register_student(student_data, cfg):
            st.success("Registration successful! You can now login with your credentials.")
            return student_data
        else:
            st.error("Registration failed. Please try again or contact administrator.")
            return None

    return None


def show_welcome_screen():
    """Display the initial welcome screen with user type selection"""
    st.title("🏫 Student Attire Verification System")
    st.markdown("---")

    st.markdown("""
    ### Welcome!

    Are you a new student or an existing user?
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Register (New Student)", use_container_width=True, type="primary"):
            st.session_state['auth_action'] = 'register'
            st.rerun()

    with col2:
        if st.button("Login (Existing User)", use_container_width=True, type="secondary"):
            st.session_state['auth_action'] = 'login'
            st.rerun()

    st.markdown("---")
    st.caption("Select your option above to proceed.")


def show_student_auth_flow(cfg: AppConfig) -> Optional[Dict]:
    """Show student authentication flow (login/register selection)"""
    st.title("👨‍🎓 Student Portal")

    # Check if user is already logged in
    if 'user' in st.session_state and st.session_state['user']:
        user = st.session_state['user']
        st.success(f"Welcome back, {user['full_name']}!")
        return user

    # User type selection
    st.markdown("### Are you a registered student?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("✅ Yes, I have an account", use_container_width=True, type="primary"):
            st.session_state['auth_action'] = 'login'

    with col2:
        if st.button("🆕 No, I'm new here", use_container_width=True, type="secondary"):
            st.session_state['auth_action'] = 'register'

    # Show appropriate form based on selection
    auth_action = st.session_state.get('auth_action')

    if auth_action == 'login':
        st.markdown("---")
        user = show_login_form(cfg)
        if user:
            return user

        # Back button
        if st.button("← Back to selection"):
            del st.session_state['auth_action']

    elif auth_action == 'register':
        st.markdown("---")
        user = show_registration_form(cfg)
        if user:
            # After successful registration, show login form
            st.markdown("---")
            st.info("Please login with your new credentials:")
            login_user = show_login_form(cfg)
            if login_user:
                return login_user

        # Back button
        if st.button("← Back to selection"):
            del st.session_state['auth_action']

    return None


def show_admin_login(cfg: AppConfig) -> Optional[Dict]:
    """Show admin login form"""
    st.title("👨‍💼 Administrator Login")

    # Check if admin is already logged in
    if 'user' in st.session_state and st.session_state['user']:
        user = st.session_state['user']
        if user.get('role') == 'admin':
            st.success(f"Welcome back, Admin {user['full_name']}!")
            return user

    st.markdown("### Admin Access")
    st.info("This area is restricted to system administrators only.")

    user = show_login_form(cfg)
    if user:
        if user.get('role') == 'admin':
            return user
        else:
            st.error("Access denied. Admin privileges required.")
            # Clear the user from session since they don't have admin rights
            if 'user' in st.session_state:
                del st.session_state['user']
            return None

    return None


def show_user_menu():
    """Show user menu in sidebar"""
    if 'user' not in st.session_state or not st.session_state['user']:
        return

    user = st.session_state['user']

    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**Logged in as:** {user['full_name']}")
        st.caption(f"Role: {user['role'].title()}")

        if st.button("Logout", use_container_width=True):
            from ..auth import logout_user
            logout_user(st.session_state)
            st.rerun()

        st.markdown("---")
