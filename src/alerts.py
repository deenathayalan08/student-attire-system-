import smtplib
from email.message import EmailMessage
from typing import Optional, List, Dict, Any
from datetime import datetime

from .config import AppConfig
from .db import get_setting


def send_email(subject: str, body: str, to_addr: str, cfg: AppConfig | None = None) -> bool:
	cfg = cfg or AppConfig()
	smtp_host = get_setting("smtp_host", cfg=cfg)
	smtp_user = get_setting("smtp_user", cfg=cfg)
	smtp_pass = get_setting("smtp_pass", cfg=cfg)
	smtp_from = get_setting("smtp_from", cfg=cfg) or smtp_user
	if not smtp_host or not smtp_from:
		return False
	msg = EmailMessage()
	msg["Subject"] = subject
	msg["From"] = smtp_from
	msg["To"] = to_addr
	msg.set_content(body)
	try:
		with smtplib.SMTP(smtp_host, 587, timeout=10) as s:
			s.starttls()
			if smtp_user and smtp_pass:
				s.login(smtp_user, smtp_pass)
			s.send_message(msg)
		return True
	except Exception:
		return False


def send_sms(text: str, to_number: str, cfg: AppConfig | None = None) -> bool:
	# Placeholder — integrate Twilio or similar here. For now return True to simulate.
	return True


def notify_non_compliance(student_id: Optional[str], zone: str, details: str, cfg: AppConfig | None = None) -> None:
	cfg = cfg or AppConfig()
	enable_email = (get_setting("alerts_email_enabled", "0", cfg=cfg) == "1")
	enable_sms = (get_setting("alerts_sms_enabled", "0", cfg=cfg) == "1")
	target_email = get_setting("alerts_email_to", cfg=cfg)
	target_sms = get_setting("alerts_sms_to", cfg=cfg)

	subject = f"Non-compliance detected in {zone}"
	body = f"Student: {student_id or 'unknown'}\nZone: {zone}\nDetails: {details}"
	if enable_email and target_email:
		send_email(subject, body, target_email, cfg)
	if enable_sms and target_sms:
		send_sms(f"{subject}: {details}", target_sms, cfg)


def notify_id_card_status(student_id: Optional[str], zone: str, id_card_detected: bool, confidence: float, cfg: AppConfig | None = None) -> None:
	"""
	Send notification about ID card detection status.
	
	Args:
		student_id: Student identifier
		zone: Zone where detection occurred
		id_card_detected: Whether ID card was detected
		confidence: Detection confidence score (0.0 to 1.0)
		cfg: Application configuration
	"""
	cfg = cfg or AppConfig()
	enable_email = (get_setting("alerts_email_enabled", "0", cfg=cfg) == "1")
	enable_sms = (get_setting("alerts_sms_enabled", "0", cfg=cfg) == "1")
	target_email = get_setting("alerts_email_to", cfg=cfg)
	target_sms = get_setting("alerts_sms_to", cfg=cfg)
	
	# Create status message
	if id_card_detected:
		status_msg = f"✅ ID CARD DETECTED"
		details = f"Student ID card successfully detected with {confidence:.1%} confidence"
	else:
		status_msg = f"❌ NO ID CARD DETECTED"
		details = f"No student ID card detected (confidence: {confidence:.1%})"
	
	subject = f"ID Card Status - {zone}"
	body = f"Student: {student_id or 'unknown'}\nZone: {zone}\nStatus: {status_msg}\nDetails: {details}"
	
	if enable_email and target_email:
		send_email(subject, body, target_email, cfg)
	if enable_sms and target_sms:
		send_sms(f"{subject}: {status_msg}", target_sms, cfg)


def get_id_card_status_message(id_card_detected: bool, confidence: float) -> str:
	"""
	Get a formatted message about ID card detection status.
	
	Args:
		id_card_detected: Whether ID card was detected
		confidence: Detection confidence score (0.0 to 1.0)
		
	Returns:
		Formatted status message
	"""
	if id_card_detected:
		if confidence >= 0.8:
			return f"✅ ID CARD DETECTED - High confidence ({confidence:.1%})"
		elif confidence >= 0.6:
			return f"✅ ID CARD DETECTED - Good confidence ({confidence:.1%})"
		else:
			return f"⚠️ ID CARD DETECTED - Low confidence ({confidence:.1%})"
	else:
		return f"❌ NO ID CARD DETECTED (confidence: {confidence:.1%})"


def get_detailed_id_card_message(id_card_result: dict) -> str:
	"""
	Get detailed ID card detection message with additional information.
	
	Args:
		id_card_result: Result dictionary from detect_id_card()
		
	Returns:
		Detailed formatted message
	"""
	detected = id_card_result.get('detected', False)
	confidence = id_card_result.get('confidence', 0.0)
	area = id_card_result.get('area', 0)
	aspect_ratio = id_card_result.get('aspect_ratio', 0)
	
	if detected:
		base_msg = f"✅ ID CARD DETECTED"
		confidence_msg = f"Confidence: {confidence:.1%}"
		size_msg = f"Size: {area:.0f} pixels"
		shape_msg = f"Shape ratio: {aspect_ratio:.2f}"
		
		return f"{base_msg}\n{confidence_msg}\n{size_msg}\n{shape_msg}"
	else:
		return f"❌ NO ID CARD DETECTED\nConfidence: {confidence:.1%}\nPlease ensure your ID card is visible and well-lit"


def alert_unauthorized_entry(student_id: str, zone: str, details: str, cfg: AppConfig | None = None) -> None:
	"""Send alert for unauthorized entry attempt"""
	cfg = cfg or AppConfig()
	enable_email = (get_setting("alerts_email_enabled", "0", cfg=cfg) == "1")
	enable_sms = (get_setting("alerts_sms_enabled", "0", cfg=cfg) == "1")
	target_email = get_setting("alerts_email_to", cfg=cfg)
	target_sms = get_setting("alerts_sms_to", cfg=cfg)
	
	subject = f"🚨 UNAUTHORIZED ENTRY ATTEMPT - {zone}"
	body = f"SECURITY ALERT:\n\nStudent ID: {student_id}\nZone: {zone}\nTime: {datetime.now()}\nDetails: {details}\n\nAction Required: Review unauthorized access attempt immediately."
	
	if enable_email and target_email:
		send_email(subject, body, target_email, cfg)
	if enable_sms and target_sms:
		send_sms(f"UNAUTHORIZED ENTRY: {student_id} at {zone}", target_sms, cfg)


def alert_late_entry_or_exit(student_id: str, zone: str, is_late: bool, details: str, cfg: AppConfig | None = None) -> None:
	"""Send alert for late entry or early exit"""
	cfg = cfg or AppConfig()
	enable_email = (get_setting("alerts_email_enabled", "0", cfg=cfg) == "1")
	enable_sms = (get_setting("alerts_sms_enabled", "0", cfg=cfg) == "1")
	target_email = get_setting("alerts_email_to", cfg=cfg)
	target_sms = get_setting("alerts_sms_to", cfg=cfg)
	
	alert_type = "LATE ENTRY" if is_late else "EARLY EXIT"
	subject = f"⚠️ {alert_type} - {zone}"
	body = f"Student: {student_id}\nZone: {zone}\nType: {alert_type}\nTime: {datetime.now()}\nDetails: {details}"
	
	if enable_email and target_email:
		send_email(subject, body, target_email, cfg)
	if enable_sms and target_sms:
		send_sms(f"{alert_type}: {student_id} at {zone}", target_sms, cfg)


def alert_emergency_incomplete_attire(student_id: str, zone: str, violations: List[Dict[str, Any]], cfg: AppConfig | None = None) -> None:
	"""Send emergency alert for incomplete attire (no ID card, no shoes, etc.)"""
	cfg = cfg or AppConfig()
	enable_email = (get_setting("alerts_email_enabled", "0", cfg=cfg) == "1")
	enable_sms = (get_setting("alerts_sms_enabled", "0", cfg=cfg) == "1")
	target_email = get_setting("alerts_email_to", cfg=cfg)
	target_sms = get_setting("alerts_sms_to", cfg=cfg)
	
	critical_violations = [v for v in violations if v.get('severity') in ['critical', 'high']]
	
	if not critical_violations:
		return
	
	subject = f"🔴 EMERGENCY: Incomplete Attire - {zone}"
	
	violation_list = "\n".join([f"- {v.get('item', 'Unknown')}: {v.get('detected', 'N/A')}" for v in critical_violations])
	body = f"EMERGENCY ALERT:\n\nStudent: {student_id}\nZone: {zone}\nTime: {datetime.now()}\n\nCritical Violations:\n{violation_list}\n\nAction Required: Immediate review needed."
	
	if enable_email and target_email:
		send_email(subject, body, target_email, cfg)
	if enable_sms and target_sms:
		send_sms(f"EMERGENCY: {student_id} incomplete attire at {zone}", target_sms, cfg)


def alert_geofence_violation(student_id: str, event_type: str, location: str, details: str, cfg: AppConfig | None = None) -> None:
	"""Send alert for geofence boundary violation"""
	cfg = cfg or AppConfig()
	enable_email = (get_setting("alerts_email_enabled", "0", cfg=cfg) == "1")
	enable_sms = (get_setting("alerts_sms_enabled", "0", cfg=cfg) == "1")
	target_email = get_setting("alerts_email_to", cfg=cfg)
	target_sms = get_setting("alerts_sms_to", cfg=cfg)
	
	subject = f"🚨 GEOFENCE ALERT: {event_type}"
	body = f"GEOFENCE VIOLATION:\n\nStudent: {student_id}\nEvent Type: {event_type}\nLocation: {location}\nTime: {datetime.now()}\nDetails: {details}\n\nAction Required: Track student location."
	
	if enable_email and target_email:
		send_email(subject, body, target_email, cfg)
	if enable_sms and target_sms:
		send_sms(f"GEOFENCE ALERT: {student_id} - {event_type} at {location}", target_sms, cfg)