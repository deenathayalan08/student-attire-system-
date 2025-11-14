# TODO: Enhance Student Attire Verification System

## Current Work: System Enhancement with Advanced Features

### Gender-Specific Analysis
- [ ] Update src/verify.py with gender-specific checks (men: beard, haircut, dress, shoes, ID card; women: ID card, dress, sandals)
- [ ] Enhance gender detection logic in verification pipeline

### Casual/Uniform Day Policies
- [ ] Implement day-of-week policy checking (Wednesday uniform, Friday casual)
- [ ] Update verification logic to apply different policies based on current day

### Biometric Integration
- [ ] Enhance src/biometric.py for mobile phone biometric simulation
- [ ] Update src/verify.py to properly integrate biometric verification
- [ ] Add biometric registration interface in app/streamlit_app.py

### Jewelry Detection
- [ ] Verify jewelry detection implementation in src/features.py
- [ ] Update verification logic for chain detection in neck area

### PDF Report Generation
- [ ] Create src/report_generator.py for detailed PDF reports
- [ ] Add student details, violations, and remarks to reports
- [ ] Integrate PDF generation in app/streamlit_app.py

### College Details Integration
- [ ] Update src/db.py schema for enhanced college details
- [ ] Integrate college information in reports and UI

### Database Schema Extensions
- [ ] Extend src/db.py with biometric data storage
- [ ] Add college details fields to database

### UI Enhancements
- [ ] Update app/streamlit_app.py with biometric registration interface
- [ ] Add report generation and download functionality
- [ ] Enhance admin dashboard with new features

### Testing and Validation
- [ ] Test enhanced system with sample data
- [ ] Validate gender-specific checks
- [ ] Test casual/uniform day policies
- [ ] Verify biometric integration
- [ ] Test PDF report generation
