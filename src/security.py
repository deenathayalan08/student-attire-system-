"""
Security & Safety Features Module

This module provides:
1. Unauthorized Entry Alert - Detect and alert on non-database attempts
2. Late Entry / Early Exit Logging - Track time-based violations
3. Emergency Notification - Alert for incomplete attire
4. Geofencing - Track campus boundary violations
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, time
import math

from .config import AppConfig
from .db import check_student_exists, log_unauthorized_access, log_access, log_emergency_alert, log_geofence_event, get_student
from .alerts import alert_unauthorized_entry, alert_late_entry_or_exit, alert_emergency_incomplete_attire, alert_geofence_violation


def check_and_alert_unauthorized_student(student_id: str, zone: str, cfg: AppConfig | None = None) -> bool:
	"""
	Check if student exists in database and alert if unauthorized.
	
	Args:
		student_id: Student ID to check
		zone: Zone where access was attempted
		cfg: Application configuration
		
	Returns:
		True if student is authorized, False if unauthorized
	"""
	cfg = cfg or AppConfig()
	
	if not cfg.enable_unauthorized_entry_alerts:
		return True  # Skip check if disabled
	
	# Check if student exists
	if not check_student_exists(student_id, cfg):
		# Log unauthorized access
		details = f"Unauthorized entry attempt by {student_id} at zone {zone}"
		log_unauthorized_access(student_id, zone, "unauthorized_entry", details, cfg)
		
		# Send alert
		alert_unauthorized_entry(student_id, zone, details, cfg)
		
		return False
	
	return True


def check_and_log_entry_time(student_id: str, zone: str, cfg: AppConfig | None = None) -> Dict[str, Any]:
	"""
	Check if entry is late and log accordingly.
	
	Args:
		student_id: Student ID
		zone: Zone of entry
		cfg: Application configuration
		
	Returns:
		Dictionary with entry info and timing status
	"""
	cfg = cfg or AppConfig()
	
	if not cfg.enable_late_entry_tracking:
		return {"is_late": False, "is_authorized": True}
	
	# Parse entry times from config
	entry_start = _parse_time(cfg.normal_entry_start_time)
	entry_end = _parse_time(cfg.normal_entry_end_time)
	
	current_time = datetime.now().time()
	is_late = False
	
	# Check if current time is before entry window (early) or after (late)
	if current_time < entry_start:
		entry_type = "early_entry"
		is_late = False
	elif current_time > entry_end:
		entry_type = "late_entry"
		is_late = True
	else:
		entry_type = "on_time_entry"
		is_late = False
	
	# Log the entry
	details = f"Entry at {current_time.strftime('%H:%M')}, Window: {entry_start.strftime('%H:%M')} - {entry_end.strftime('%H:%M')}"
	log_access(student_id, zone, entry_type, is_late=int(is_late), is_early_exit=0, details=details, cfg=cfg)
	
	# Send alert if late
	if is_late:
		alert_late_entry_or_exit(student_id, zone, is_late=True, details=details, cfg=cfg)
	
	return {"is_late": is_late, "entry_type": entry_type, "is_authorized": True}


def check_and_log_exit_time(student_id: str, zone: str, cfg: AppConfig | None = None) -> Dict[str, Any]:
	"""
	Check if exit is early and log accordingly.
	
	Args:
		student_id: Student ID
		zone: Zone of exit
		cfg: Application configuration
		
	Returns:
		Dictionary with exit info and timing status
	"""
	cfg = cfg or AppConfig()
	
	if not cfg.enable_early_exit_tracking:
		return {"is_early": False, "is_authorized": True}
	
	# Parse exit times from config
	exit_start = _parse_time(cfg.normal_exit_start_time)
	exit_end = _parse_time(cfg.normal_exit_end_time)
	
	current_time = datetime.now().time()
	is_early = False
	
	# Check if current time is before exit window (too early)
	if current_time < exit_start:
		exit_type = "early_exit"
		is_early = True
	elif current_time > exit_end:
		exit_type = "late_exit"
		is_early = False
	else:
		exit_type = "on_time_exit"
		is_early = False
	
	# Log the exit
	details = f"Exit at {current_time.strftime('%H:%M')}, Window: {exit_start.strftime('%H:%M')} - {exit_end.strftime('%H:%M')}"
	log_access(student_id, zone, exit_type, is_late=0, is_early_exit=int(is_early), details=details, cfg=cfg)
	
	# Send alert if early
	if is_early:
		alert_late_entry_or_exit(student_id, zone, is_late=False, details=details, cfg=cfg)
	
	return {"is_early": is_early, "exit_type": exit_type, "is_authorized": True}


def check_and_alert_emergency_violations(student_id: str, zone: str, violations: List[Dict[str, Any]], cfg: AppConfig | None = None) -> None:
	"""
	Check for emergency-level violations and send alerts.
	
	Args:
		student_id: Student ID
		zone: Zone of verification
		violations: List of violations from verification
		cfg: Application configuration
	"""
	cfg = cfg or AppConfig()
	
	if not cfg.enable_emergency_alerts:
		return
	
	# Find critical violations (missing ID card, no shoes, etc.)
	critical_items = ["ID Card", "Footwear", "Lab Safety"]
	critical_violations = [v for v in violations if v.get('item') in critical_items and v.get('severity') in ['critical', 'high']]
	
	if critical_violations:
		# Log emergency alert
		details = "; ".join([f"{v.get('item')}: {v.get('detected', 'N/A')}" for v in critical_violations])
		log_emergency_alert(student_id, "incomplete_attire", "critical", details, cfg)
		
		# Send alert
		alert_emergency_incomplete_attire(student_id, zone, critical_violations, cfg)


def check_geofence(latitude: float, longitude: float, student_id: str, event_type: str, location: str, cfg: AppConfig | None = None) -> bool:
	"""
	Check if location is within campus boundaries (geofencing).
	
	Args:
		latitude: Current latitude
		longitude: Current longitude
		student_id: Student ID
		event_type: Type of geofence event
		location: Location name
		cfg: Application configuration
		
	Returns:
		True if within bounds, False if outside
	"""
	cfg = cfg or AppConfig()
	
	if not cfg.enable_geofencing:
		return True  # Skip geofencing check if disabled
	
	# Calculate distance from campus center
	campus_lat = cfg.campus_latitude
	campus_lon = cfg.campus_longitude
	radius = cfg.campus_radius_meters
	
	distance = _calculate_distance(latitude, longitude, campus_lat, campus_lon)
	
	if distance > radius:
		# Log geofence violation
		details = f"Student {distance:.1f}m outside campus boundary (radius: {radius}m)"
		log_geofence_event(student_id, event_type, location, latitude, longitude, details, cfg)
		
		# Send alert
		alert_geofence_violation(student_id, event_type, location, details, cfg)
		
		return False
	
	return True


def _parse_time(time_str: str) -> time:
	"""Parse time string (HH:MM) to time object"""
	try:
		hour, minute = map(int, time_str.split(':'))
		return time(hour, minute)
	except:
		return time(8, 0)  # Default to 8:00 AM


def _calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
	"""
	Calculate distance between two GPS coordinates using Haversine formula.
	
	Returns distance in meters.
	"""
	R = 6371000  # Earth radius in meters
	
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	delta_phi = math.radians(lat2 - lat1)
	delta_lambda = math.radians(lon2 - lon1)
	
	a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
	c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
	
	return R * c
