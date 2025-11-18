# TODO: Enhance Student Attire Verification System

## Current Work: Phone-PC Biometric Integration & Student Management

### Phone-PC Biometric Architecture
- [x] Modify src/biometric.py for phone-side authentication (return only verified student ID)
- [x] Create src/phone_comm.py for secure phone-PC communication simulation
- [x] Update src/verify.py to complete verify_attire_and_safety() function with full attire analysis
- [x] Update app/streamlit_app.py to receive verified student ID and display attire results

### Student Management System
- [x] Add student registration interface in app/streamlit_app.py
- [x] Implement biometric registration during student onboarding
- [x] Update src/db.py for enhanced student data management
- [x] Add student search and management features in admin dashboard

### Attire Verification Logic
- [ ] Complete src/verify.py with gender-specific checks (men: beard, haircut, dress, shoes, ID card; women: ID card, dress, sandals)
- [ ] Implement day-of-week policy checking (Wednesday uniform, Friday casual)
- [ ] Integrate jewelry detection for chain detection in neck area
- [ ] Add comprehensive violation scoring and reporting

### Report Generation
- [ ] Create src/report_generator.py for detailed PDF reports
- [ ] Add student details, violations, and remarks to reports
- [ ] Integrate PDF generation in app/streamlit_app.py

### Testing and Validation
- [ ] Test phone-PC communication simulation
- [ ] Verify student registration and biometric onboarding
- [ ] Test end-to-end verification flow with sample data
- [ ] Validate attire analysis accuracy and reporting
