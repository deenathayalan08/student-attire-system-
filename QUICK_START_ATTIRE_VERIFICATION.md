# Quick Start Guide: Attire Verification Feature

## 🚀 How to Use the New Feature

### Step 1: Login with Face Authentication
1. Open the application
2. Click **"Login (Existing User)"** or navigate to **"Face Authentication"**
3. Capture your face using the camera
4. Crop your face if needed
5. Click **"✅ Verify Face"**
6. Wait for face matching
7. Click **"✅ Confirm Login"**

### Step 2: Automatic Redirect
- After successful login, you'll be automatically redirected to your **Student Dashboard**
- You'll see your complete student information displayed

### Step 3: Upload Image for Verification
Scroll down to the **"👔 Attire Verification"** section

**Option A: Upload from Device**
1. Click the **"📤 Upload Image"** tab
2. Click **"Browse files"** or drag & drop
3. Select a full-body photo (JPG, JPEG, or PNG)
4. Image preview will appear

**Option B: Take Photo with Camera**
1. Click the **"📷 Take Photo"** tab
2. Click **"📷 Capture full-body photo"**
3. Position yourself for a full-body shot
4. Click the camera button to capture

### Step 4: Verify Your Attire
1. Review the uploaded/captured image
2. Click **"✅ Verify Attire"** button
3. Wait for the 4-step verification process:
   - Step 1: Converting image (25%)
   - Step 2: Detecting body pose (50%)
   - Step 3: Analyzing attire features (75%)
   - Step 4: Verifying compliance (100%)

### Step 5: Review Results
You'll see:
- ✅ **Overall Status** (PASS/WARNING/FAIL)
- 📊 **Scores** (Success Score, Fail Score)
- 📸 **Annotated Image** (with violation markers)
- 🚨 **Violation Details** (if any)
- 🆔 **ID Card Detection Status**
- 📋 **Event Log ID**

### Step 6: Download Report (Optional)
- Click **"📥 Download Verification Report"**
- Save the JSON report for your records

---

## 📸 Image Requirements

### ✅ Good Image Examples:
- Full-body visible (head to feet)
- Clear, well-lit photo
- Frontal view
- ID card visible on chest
- Proper footwear visible
- No obstructions
- Good contrast

### ❌ Avoid:
- Cropped images (only upper body)
- Dark/poorly lit photos
- Side or back views
- ID card hidden
- Feet not visible
- Blurry images
- Multiple people in frame

---

## 🎯 What Gets Checked?

### For Male Students:
- ✓ Proper shirt (not t-shirt)
- ✓ Any color pants (configurable)
- ✓ Full-length pants (no shorts)
- ✓ Black shoes required
- ✓ ID card visible

### For Female Students:
- ✓ Kurti with dupatta OR proper top
- ✓ Appropriate bottom wear
- ✓ Footwear (shoes or sandals)
- ✓ ID card visible

---

## 🔍 Understanding Results

### Status Indicators:
- **✅ PASS** - All checks passed, fully compliant
- **⚠️ WARNING** - Minor issues detected, mostly compliant
- **❌ FAIL** - Major violations, not compliant

### Violation Severity:
- **🔴 Critical** - Must fix immediately (e.g., no ID card)
- **🟠 High** - Important issue (e.g., wrong color)
- **🟡 Medium** - Minor issue (e.g., slight deviation)
- **🔵 Low** - Informational (e.g., not visible in image)

### Scores:
- **Success Score** - Higher is better (aim for 70%+)
- **Fail Score** - Lower is better (aim for below 30%)

---

## 💡 Tips for Best Results

1. **Lighting**
   - Use natural daylight or bright indoor lighting
   - Avoid backlighting (don't stand in front of window)
   - Ensure face and body are well-lit

2. **Camera Position**
   - Place camera at chest height
   - Stand 6-8 feet away from camera
   - Ensure full body fits in frame

3. **Attire**
   - Wear complete uniform before taking photo
   - Ensure ID card is visible on chest
   - Check shoes are visible in frame
   - Stand straight with arms at sides

4. **Background**
   - Use plain, uncluttered background
   - Avoid busy patterns or multiple people
   - Ensure good contrast with your attire

5. **Image Quality**
   - Use rear camera (better quality than front)
   - Hold camera steady or use tripod
   - Take multiple photos and choose best one
   - Ensure image is not blurry

---

## ❓ Troubleshooting

### "No face detected"
- Ensure your face is clearly visible
- Check lighting conditions
- Move closer to camera
- Remove any face obstructions

### "ID Card not detected"
- Ensure ID card is visible on chest
- Check ID card is not covered by clothing
- Improve lighting on ID card area
- Position ID card facing camera

### "Footwear not detected"
- Ensure feet are visible in frame
- Stand further from camera to include feet
- Check shoes are not hidden by pants
- Improve lighting on lower body

### "Low confidence score"
- Retake photo with better lighting
- Ensure full body is visible
- Check all uniform items are worn
- Use higher quality camera

### "Multiple violations"
- Review each violation detail
- Check "Required" vs "Detected" for each item
- Read the "Reason" for each violation
- Correct issues and retake photo

---

## 📱 Access Points

You can access the attire verification feature from:

1. **After Login** - Automatic redirect to dashboard
2. **My Dashboard** - Navigate from sidebar menu
3. **Student Dashboard** - Direct page access

---

## 🔐 Privacy & Security

- Your images are processed locally
- No images stored permanently (unless configured by admin)
- Only verification results are logged
- Face authentication required for access
- All data encrypted in transit

---

## 📊 Verification History

To view your past verifications:
1. Navigate to **"My Dashboard"**
2. Scroll to verification history section
3. View event logs with timestamps
4. Check compliance trends

---

## 🎓 Benefits

- **Self-Check** - Verify compliance before entering campus
- **Instant Feedback** - Know immediately if attire is correct
- **Detailed Guidance** - Understand what needs correction
- **Track Record** - Keep history of verifications
- **Save Time** - Avoid manual checks at gate

---

## 📞 Need Help?

If you encounter issues:
1. Check this guide first
2. Review image requirements
3. Try different lighting/angle
4. Contact system administrator
5. Report technical issues to IT support

---

**Happy Verifying! 🎉**

Remember: The goal is to help you ensure compliance, not to penalize. Use this tool to self-check before coming to campus!
