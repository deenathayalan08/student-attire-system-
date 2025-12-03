import streamlit as st
from typing import Optional, Dict, Any

from ..auth import authenticate_user, register_student, check_username_exists, check_student_id_exists
from ..config import AppConfig
from ..db import update_student_verification, update_student_face, update_student_roll_no


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
    """Display student registration form with 3 stages including face capture"""
    st.title("📝 Student Registration")
    st.markdown("Complete all stages to finish registration")
    
    # Determine current stage
    current_stage = 1
    if 'reg_stage1' in st.session_state:
        current_stage = 2
    if 'reg_stage2' in st.session_state:
        current_stage = 3
    
    # Progress indicator
    progress_col1, progress_col2, progress_col3 = st.columns(3)
    with progress_col1:
        if current_stage >= 1:
            st.write("#### ✅ Stage 1: ID Generation")
        else:
            st.write("#### Stage 1: ID Generation")
    with progress_col2:
        if current_stage >= 2:
            st.write("#### ✅ Stage 2: Details")
        else:
            st.write("#### Stage 2: Details")
    with progress_col3:
        if current_stage >= 3:
            st.write("#### ✅ Stage 3: Face")
        else:
            st.write("#### Stage 3: Face")

    # STAGE 1: Generate Student ID
    if current_stage == 1:
        st.markdown("---")
        st.markdown("### 📝 Stage 1: Generate Student ID")
        
        from ..db import get_all_departments
        departments = get_all_departments(cfg=cfg)
        dept_options = [""] + [f"{d['name']} ({d['code']})" for d in departments]

        with st.form("stage1_form"):
            col1, col2 = st.columns(2)

            with col1:
                batch_year = st.number_input("Batch Year *", min_value=2000, max_value=2100, value=2024, step=1)
                selected_dept = st.selectbox("Department *", dept_options, index=0)

            with col2:
                section_options = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
                section = st.selectbox("Section *", section_options, index=0)
                student_number = st.number_input("Student Number *", min_value=1, max_value=999, value=1, step=1)

            # Auto-generate Student ID
            auto_student_id = ""
            auto_class = ""
            auto_roll_no = ""
            if selected_dept and batch_year:
                dept_name = selected_dept.split(" (")[0] if " (" in selected_dept else selected_dept
                dept_info = next((d for d in departments if d['name'] == dept_name), None)
                if dept_info:
                    dept_id = f"{dept_info['id']:02d}"
                    batch_yy = str(batch_year)[-2:]
                    section_num = str(section_options.index(section) + 1)
                    student_num = f"{student_number:03d}"
                    auto_student_id = f"{batch_yy}{dept_id}{section_num}{student_num}"
                    auto_class = f"{dept_info['code']}-{section}"
                    auto_roll_no = auto_student_id  # Same as student ID

            stage1_submit = st.form_submit_button("Proceed to Stage 2", use_container_width=True, type="primary")

        # Show generated IDs AFTER clicking Proceed button
        if stage1_submit:
            if not auto_student_id:
                st.error("❌ Please select a department and fill all required fields!")
            else:
                # Display the generated IDs
                st.success("✅ Student ID Generated Successfully!")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Student ID:** `{auto_student_id}`")
                with col2:
                    st.info(f"**Class:** `{auto_class}`")
                
                # Store stage 1 data in session
                st.session_state['reg_stage1'] = {
                    'student_id': auto_student_id,
                    'class': auto_class,
                    'roll_no': auto_roll_no,
                    'department': selected_dept.split(" (")[0] if " (" in selected_dept else selected_dept,
                    'batch_year': batch_year
                }
                st.rerun()

    # STAGE 2: Student Details (show if Stage 1 completed)
    if 'reg_stage1' in st.session_state:
        st.markdown("---")
        st.markdown("### 👤 Stage 2: Student Details")

        stage1_data = st.session_state['reg_stage1']

        with st.form("stage2_form"):
            col1, col2 = st.columns(2)

            with col1:
                full_name = st.text_input("Full Name *", key="reg_full_name")
                email = st.text_input("Email *", key="reg_email")
                phone = st.text_input("Phone", key="reg_phone")

            with col2:
                gender = st.radio("Gender *", ["Male", "Female"], index=0, key="student_gender")
                contact_info = st.text_area("Contact Info", height=80, key="reg_contact")

            # Terms and conditions
            agree_terms = st.checkbox("I agree to the terms and conditions", key="reg_agree_terms")

            col1, col2 = st.columns(2)
            with col1:
                stage2_back = st.form_submit_button("← Back to Stage 1", use_container_width=True)
            with col2:
                stage2_submit = st.form_submit_button("Proceed to Stage 3", use_container_width=True, type="primary")

        if stage2_back:
            del st.session_state['reg_stage1']
            st.rerun()

        if stage2_submit:
            # Validation
            if not all([full_name, email]):
                st.error("Please fill in all required fields marked with *.")
                return None

            if not agree_terms:
                st.error("Please agree to the terms and conditions.")
                return None

            # Store stage 2 data
            st.session_state['reg_stage2'] = {
                'full_name': full_name,
                'email': email,
                'phone': phone,
                'gender': 'M' if gender == 'Male' else 'F',
                'contact_info': contact_info
            }
            st.rerun()

    # STAGE 3: Emergency Login Credentials (show if Stage 2 completed)
    if 'reg_stage2' in st.session_state and 'reg_stage1' in st.session_state and 'reg_stage3' not in st.session_state:
        st.markdown("---")
        st.markdown("### 🔐 Stage 3: Emergency Login Setup")
        st.info("⚠️ **Emergency Login** - Use this as backup when face authentication is not available")
        
        stage1_data = st.session_state['reg_stage1']
        
        st.write("**Username:** Your Student ID will be used as username")
        st.code(stage1_data['student_id'], language=None)
        
        with st.form("stage3_form"):
            st.markdown("#### Create Emergency Password")
            st.caption("This password will be used for emergency login when face authentication fails")
            
            password = st.text_input(
                "Password *",
                type="password",
                help="Minimum 6 characters, must contain letters and numbers",
                key="reg_password"
            )
            
            confirm_password = st.text_input(
                "Confirm Password *",
                type="password",
                key="reg_confirm_password"
            )
            
            # Password strength indicator
            if password:
                from ..validation import validate_password
                is_valid, error_msg = validate_password(password)
                if is_valid:
                    st.success("✅ Password strength: Good")
                else:
                    st.error(f"❌ {error_msg}")
            
            st.markdown("---")
            st.info("💡 **Remember:** Face authentication is the primary login method. This password is only for emergencies.")
            
            col1, col2 = st.columns(2)
            with col1:
                stage3_back = st.form_submit_button("← Back to Stage 2", use_container_width=True)
            with col2:
                stage3_submit = st.form_submit_button("Proceed to Face Capture", use_container_width=True, type="primary")
        
        if stage3_back:
            del st.session_state['reg_stage2']
            st.rerun()
        
        if stage3_submit:
            # Validate password
            if not password or not confirm_password:
                st.error("Please enter and confirm your password.")
                return None
            
            if password != confirm_password:
                st.error("Passwords do not match!")
                return None
            
            from ..validation import validate_password
            is_valid, error_msg = validate_password(password)
            if not is_valid:
                st.error(f"Invalid password: {error_msg}")
                return None
            
            # Store stage 3 data
            st.session_state['reg_stage3'] = {
                'password': password
            }
            st.success("✅ Emergency login credentials set!")
            st.rerun()

    # STAGE 4: Face Capture (show if Stage 3 completed)
    if 'reg_stage3' in st.session_state and 'reg_stage2' in st.session_state and 'reg_stage1' in st.session_state:
        st.markdown("---")
        st.markdown("### 👤 Stage 4: Face Registration (Biometric Verification)")
        
        stage1_data = st.session_state['reg_stage1']
        stage2_data = st.session_state['reg_stage2']
        stage3_data = st.session_state['reg_stage3']
        
        st.info("📸 Capture a clear photo of your face for biometric verification")
        st.write("Make sure: ✓ Face is centered  |  ✓ Good lighting  |  ✓ No obstructions")
        
        # Camera permission prompt
        if 'camera_permission_granted' not in st.session_state:
            st.warning("⚠️ **Camera Access Required**")
            st.write("This application needs access to your camera to capture your face for biometric verification.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ Allow Camera Access", use_container_width=True, type="primary", key="allow_camera"):
                    st.session_state['camera_permission_granted'] = True
                    st.rerun()
            with col2:
                if st.button("❌ Deny Access", use_container_width=True, key="deny_camera"):
                    st.error("❌ Camera access is required to complete registration.")
                    st.info("Please click 'Allow Camera Access' to continue.")
            return None
        
        # Camera capture (only shown after permission granted)
        captured_face = st.camera_input("📷 Capture your face", key="face_register")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Stage 3", use_container_width=True):
                del st.session_state['reg_stage3']
                st.rerun()

        # When a camera capture is provided, store bytes in session so user can confirm or retake
        if captured_face is not None:
            st.session_state['reg_captured_face'] = captured_face.getvalue()

        if 'reg_captured_face' in st.session_state:
            # Show preview and cropping tool
            st.markdown("**📸 Crop Your Face:**")
            st.info("ℹ️ Adjust the box to crop your face. Make sure your face is centered and clearly visible.")
            
            try:
                from PIL import Image
                from streamlit_cropper import st_cropper
                import io as _io
                
                img = Image.open(_io.BytesIO(st.session_state['reg_captured_face']))
                
                # Cropping tool
                cropped_img = st_cropper(
                    img,
                    realtime_update=True,
                    box_color='#00FF00',
                    aspect_ratio=None,  # Free aspect ratio
                    return_type='image'
                )
                
                # Store cropped image
                st.session_state['reg_cropped_face'] = cropped_img
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
                # Fallback: show original
                try:
                    st.image(st.session_state['reg_captured_face'], width=240)
                except Exception:
                    pass

            confirm_col, retake_col = st.columns(2)
            with retake_col:
                if st.button("🔄 Retake Photo", use_container_width=True, key="retake_photo"):
                    if 'reg_captured_face' in st.session_state:
                        del st.session_state['reg_captured_face']
                    if 'reg_cropped_face' in st.session_state:
                        del st.session_state['reg_cropped_face']
                    st.rerun()
            with confirm_col:
                if st.button("✅ Confirm & Register", use_container_width=True, type="primary", key="confirm_photo"):
                    # Process confirmed photo
                    with st.spinner("🔍 Processing face..."):
                        from .face_login_ui import show_face_registration_stage
                        from ..face_authentication import FaceAuthenticator
                        from PIL import Image
                        import io as _io

                        face_auth = FaceAuthenticator(cfg)
                        
                        # Use cropped image if available, otherwise use original
                        if 'reg_cropped_face' in st.session_state:
                            # Convert PIL Image to bytes
                            img_byte_arr = _io.BytesIO()
                            st.session_state['reg_cropped_face'].save(img_byte_arr, format='JPEG')
                            image_bytes = img_byte_arr.getvalue()
                        else:
                            image_bytes = st.session_state['reg_captured_face']

                        success, face_hash, face_image, message = face_auth.capture_face_for_registration(image_bytes)

                        if success:
                            st.success(message)
                            # Show thumbnail confirmation (already shown above)

                            # Save face image
                            face_image_path = ""
                            if face_image is not None:
                                face_image_path = face_auth.save_face_image(
                                    face_image,
                                    stage1_data['student_id'],
                                    stage1_data['roll_no']
                                ) or ""

                            # Create complete student data
                            student_data = {
                                'student_id': stage1_data['student_id'],
                                'roll_no': stage1_data['roll_no'],
                                'full_name': stage2_data['full_name'],
                                'username': stage1_data['student_id'],  # Use student_id as username
                                'password': stage3_data['password'],  # Use emergency password
                                'name': stage2_data['full_name'],
                                'class': stage1_data['class'],
                                'department': stage1_data['department'],
                                'gender': stage2_data['gender'],
                                'email': stage2_data['email'],
                                'phone': stage2_data['phone'],
                                'contact_info': stage2_data['contact_info'],
                                'face_hash': face_hash,
                                'face_image_path': face_image_path
                            }

                            # Register student with all information
                            try:
                                if register_student(student_data, cfg):
                                    # Update face information
                                    from ..db import update_student_face, update_student_roll_no, update_student_verification
                                    update_student_roll_no(stage1_data['student_id'], stage1_data['roll_no'], cfg)
                                    update_student_face(stage1_data['student_id'], face_hash, face_image_path, cfg)
                                    update_student_verification(stage1_data['student_id'], 1, cfg)

                                    st.success("🎉 Registration successful!")
                                    st.info("✅ Your face has been stored in the database.")
                                    st.balloons()

                                    # Clear session
                                    if 'reg_stage1' in st.session_state:
                                        del st.session_state['reg_stage1']
                                    if 'reg_stage2' in st.session_state:
                                        del st.session_state['reg_stage2']
                                    if 'reg_stage3' in st.session_state:
                                        del st.session_state['reg_stage3']
                                    if 'reg_captured_face' in st.session_state:
                                        del st.session_state['reg_captured_face']
                                    if 'camera_permission_granted' in st.session_state:
                                        del st.session_state['camera_permission_granted']

                                    # Redirect to login page
                                    st.session_state['page'] = 'login'
                                    st.session_state['registration_complete'] = True
                                    st.info("🔐 Please proceed to login with your face biometric.")
                                    
                                    # Show button to go to login
                                    if st.button("Go to Login", use_container_width=True, type="primary"):
                                        st.rerun()
                                    
                                    return None
                                else:
                                    st.error("❌ Registration failed!")
                                    st.warning("⚠️ **This Student ID is already registered!**")
                                    st.write(f"Student ID `{stage1_data['student_id']}` already exists in the database.")
                                    st.info("💡 **Solutions:**")
                                    st.write("1. If this is you, use **Face Authentication** to login")
                                    st.write("2. If this is a new student, contact admin to delete the old record")
                                    st.write("3. Or use a different student number in Stage 1")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        # Add button to restart registration
                                        if st.button("🔄 Start Over", use_container_width=True):
                                            if 'reg_stage1' in st.session_state:
                                                del st.session_state['reg_stage1']
                                            if 'reg_stage2' in st.session_state:
                                                del st.session_state['reg_stage2']
                                            if 'reg_captured_face' in st.session_state:
                                                del st.session_state['reg_captured_face']
                                            if 'reg_cropped_face' in st.session_state:
                                                del st.session_state['reg_cropped_face']
                                            st.rerun()
                                    
                                    with col2:
                                        if st.button("🔐 Go to Login", use_container_width=True, type="primary"):
                                            st.session_state['page'] = 'login'
                                            st.rerun()
                                    
                                    return None
                            except Exception as e:
                                st.error(f"❌ Registration error: {str(e)}")
                                st.info("Please contact the administrator if this problem persists.")
                                return None
                        else:
                            st.error(message)
                            st.info("Please try again with a clearer photo")
                            return None
        else:
            st.info("👆 Please capture your face using the camera above")

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
