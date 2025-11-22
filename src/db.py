import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AppConfig


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
	id TEXT PRIMARY KEY,
	name TEXT,
	class TEXT,
	department TEXT,
	gender TEXT,
	rfid TEXT UNIQUE,
	uniform_type TEXT,
	email TEXT,
	phone TEXT,
	verified INTEGER DEFAULT 0,
	contact_info TEXT
);

CREATE TABLE IF NOT EXISTS users (
	username TEXT PRIMARY KEY,
	password TEXT,
	role TEXT,
	full_name TEXT,
	email TEXT,
	assigned_class TEXT
);

CREATE TABLE IF NOT EXISTS events (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	student_id TEXT,
	timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
	zone TEXT,
	status TEXT,
	score REAL,
	label TEXT,
	details TEXT,
	image_path TEXT,
	FOREIGN KEY(student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS settings (
	key TEXT PRIMARY KEY,
	value TEXT
);

CREATE TABLE IF NOT EXISTS unauthorized_access (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	student_id TEXT,
	timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
	zone TEXT,
	attempt_type TEXT,
	details TEXT,
	alert_sent INTEGER DEFAULT 0,
	FOREIGN KEY(student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS access_log (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	student_id TEXT,
	timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
	zone TEXT,
	entry_type TEXT,
	is_late INTEGER DEFAULT 0,
	is_early_exit INTEGER DEFAULT 0,
	details TEXT,
	FOREIGN KEY(student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS emergency_alerts (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	student_id TEXT,
	timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
	alert_type TEXT,
	severity TEXT,
	details TEXT,
	resolved INTEGER DEFAULT 0,
	FOREIGN KEY(student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS geofence_events (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	student_id TEXT,
	timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
	event_type TEXT,
	location TEXT,
	latitude REAL,
	longitude REAL,
	details TEXT,
	FOREIGN KEY(student_id) REFERENCES students(id)
);

CREATE TABLE IF NOT EXISTS departments (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	department_id TEXT UNIQUE NOT NULL,
	name TEXT UNIQUE NOT NULL,
	short_form TEXT UNIQUE NOT NULL,
	num_classes INTEGER NOT NULL,
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def get_conn(cfg: AppConfig | None = None) -> sqlite3.Connection:
	cfg = cfg or AppConfig()
	db_path = Path(cfg.data_dir) / "attire.db"
	conn = sqlite3.connect(db_path)
	return conn


def init_db(cfg: AppConfig | None = None) -> None:
	conn = get_conn(cfg)
	with conn:
		conn.executescript(SCHEMA)
		# Lightweight migrations for identification fields
		def _has_column(table: str, column: str) -> bool:
			cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
			col_names = {c[1] for c in cols}
			return column in col_names

		def _ensure_column(table: str, column: str, decl: str) -> None:
			if not _has_column(table, column):
				try:
					conn.execute(f"ALTER TABLE {table} ADD COLUMN {decl}")
				except sqlite3.OperationalError:
					# Some SQLite builds disallow constraints in ADD COLUMN or may raise even when safe; continue
					pass
		def _ensure_unique_index(table: str, column: str, index_name: str) -> None:
			if not _has_column(table, column):
				return
			indexes = conn.execute(f"PRAGMA index_list({table})").fetchall()
			index_names = {i[1] for i in indexes}
			if index_name not in index_names:
				try:
					conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table}({column})")
				except sqlite3.OperationalError:
					# If creation fails due to duplicates, leave it and proceed
					pass
		# Students extra columns
		# SQLite cannot add a column with UNIQUE via ALTER TABLE; add column then unique index
		_ensure_column("students", "roll_no", "roll_no TEXT")
		_ensure_unique_index("students", "roll_no", "idx_students_roll_no_unique")
		_ensure_column("students", "id_card_hash", "id_card_hash TEXT")
		_ensure_column("students", "face_hash", "face_hash TEXT")
		
		# Add new columns for students
		_ensure_column("students", "department", "department TEXT")
		_ensure_column("students", "gender", "gender TEXT")
		_ensure_column("students", "uniform_type", "uniform_type TEXT")
		_ensure_column("students", "email", "email TEXT")
		_ensure_column("students", "phone", "phone TEXT")
		_ensure_column("students", "verified", "verified INTEGER DEFAULT 0")
		_ensure_column("students", "contact_info", "contact_info TEXT")
		_ensure_column("students", "face_embedding", "face_embedding TEXT")

		# Ensure departments table has department_id column
		_ensure_column("departments", "department_id", "department_id TEXT UNIQUE")
	conn.close()


def insert_event(row: Dict[str, Any], cfg: AppConfig | None = None) -> int:
	conn = get_conn(cfg)
	with conn:
		cur = conn.execute(
			"INSERT INTO events (student_id, zone, status, score, label, details, image_path) VALUES (?,?,?,?,?,?,?)",
			(
				row.get("student_id"),
				row.get("zone"),
				row.get("status"),
				row.get("score"),
				row.get("label"),
				row.get("details"),
				row.get("image_path"),
			),
		)
		event_id = cur.lastrowid
	return event_id


def list_events(limit: int = 100, cfg: AppConfig | None = None) -> List[Dict[str, Any]]:
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
	conn.close()
	return [dict(r) for r in rows]


def upsert_setting(key: str, value: str, cfg: AppConfig | None = None) -> None:
	conn = get_conn(cfg)
	with conn:
		conn.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
	conn.close()


def get_setting(key: str, default: Optional[str] = None, cfg: AppConfig | None = None) -> Optional[str]:
	conn = get_conn(cfg)
	row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
	conn.close()
	return row[0] if row else default


def check_student_exists(student_id: str, cfg: AppConfig | None = None) -> bool:
	"""Check if a student exists in the database"""
	conn = get_conn(cfg)
	row = conn.execute("SELECT id FROM students WHERE id=?", (student_id,)).fetchone()
	conn.close()
	return row is not None


def log_unauthorized_access(student_id: str, zone: str, attempt_type: str, details: str, cfg: AppConfig | None = None) -> int:
	"""Log unauthorized access attempt"""
	conn = get_conn(cfg)
	with conn:
		cur = conn.execute(
			"INSERT INTO unauthorized_access (student_id, zone, attempt_type, details) VALUES (?,?,?,?)",
			(student_id, zone, attempt_type, details)
		)
		return cur.lastrowid


def log_access(student_id: str, zone: str, entry_type: str, is_late: int = 0, is_early_exit: int = 0, details: str = "", cfg: AppConfig | None = None) -> int:
	"""Log student access with time tracking"""
	conn = get_conn(cfg)
	with conn:
		cur = conn.execute(
			"INSERT INTO access_log (student_id, zone, entry_type, is_late, is_early_exit, details) VALUES (?,?,?,?,?,?)",
			(student_id, zone, entry_type, is_late, is_early_exit, details)
		)
		return cur.lastrowid


def log_emergency_alert(student_id: str, alert_type: str, severity: str, details: str, cfg: AppConfig | None = None) -> int:
	"""Log emergency alert"""
	conn = get_conn(cfg)
	with conn:
		cur = conn.execute(
			"INSERT INTO emergency_alerts (student_id, alert_type, severity, details) VALUES (?,?,?,?)",
			(student_id, alert_type, severity, details)
		)
		return cur.lastrowid


def log_geofence_event(student_id: str, event_type: str, location: str, latitude: float, longitude: float, details: str, cfg: AppConfig | None = None) -> int:
	"""Log geofence event"""
	conn = get_conn(cfg)
	with conn:
		cur = conn.execute(
			"INSERT INTO geofence_events (student_id, event_type, location, latitude, longitude, details) VALUES (?,?,?,?,?,?)",
			(student_id, event_type, location, latitude, longitude, details)
		)
		return cur.lastrowid


def get_student(student_id: str, cfg: AppConfig | None = None) -> Optional[Dict[str, Any]]:
	"""Get student by ID"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	row = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
	conn.close()
	return dict(row) if row else None


def get_all_students(cfg: AppConfig | None = None) -> List[Dict[str, Any]]:
	"""Get all students"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("SELECT * FROM students ORDER BY id").fetchall()
	conn.close()
	return [dict(r) for r in rows]


def add_student(student_data: Dict[str, Any], cfg: AppConfig | None = None) -> None:
	"""Add or update student"""
	conn = get_conn(cfg)
	with conn:
		conn.execute(
			"""INSERT OR REPLACE INTO students
			(id, name, class, department, gender, uniform_type, email, phone, contact_info, face_embedding)
			VALUES (?,?,?,?,?,?,?,?,?,?)""",
			(
				student_data.get("id"),
				student_data.get("name"),
				student_data.get("class"),
				student_data.get("department"),
				student_data.get("gender"),
				student_data.get("uniform_type"),
				student_data.get("email"),
				student_data.get("phone"),
				student_data.get("contact_info"),
				student_data.get("face_embedding"),
			)
		)
	conn.close()


def update_student_verification(student_id: str, verified: int, cfg: AppConfig | None = None) -> None:
	"""Update student verification status"""
	conn = get_conn(cfg)
	with conn:
		conn.execute("UPDATE students SET verified = ? WHERE id = ?", (verified, student_id))
	conn.close()


def get_compliance_stats(date: Optional[str] = None, cfg: AppConfig | None = None) -> Dict[str, Any]:
	"""Get daily compliance statistics"""
	conn = get_conn(cfg)
	
	date_filter = f"WHERE DATE(timestamp) = '{date}'" if date else ""
	
	# Total events
	total = conn.execute(f"SELECT COUNT(*) FROM events {date_filter}").fetchone()[0]
	
	# Compliant events
	compliant = conn.execute(f"SELECT COUNT(*) FROM events {date_filter} WHERE status = 'PASS'").fetchone()[0]
	
	# Non-compliant events
	non_compliant = conn.execute(f"SELECT COUNT(*) FROM events {date_filter} WHERE status != 'PASS'").fetchone()[0]
	
	# Compliance percentage
	compliance_pct = (compliant / total * 100) if total > 0 else 0
	
	# Verified students
	verified_students = conn.execute("SELECT COUNT(*) FROM students WHERE verified = 1").fetchone()[0]
	
	# Total students
	total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
	
	conn.close()
	
	return {
		"total_events": total,
		"compliant_events": compliant,
		"non_compliant_events": non_compliant,
		"compliance_percentage": compliance_pct,
		"verified_students": verified_students,
		"total_students": total_students
	}


def get_user(username: str, cfg: AppConfig | None = None) -> Optional[Dict[str, Any]]:
	"""Get user by username"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
	conn.close()
	return dict(row) if row else None


def add_user(username: str, password: str, role: str, full_name: str, email: str, assigned_class: str = "", cfg: AppConfig | None = None) -> None:
	"""Add user"""
	conn = get_conn(cfg)
	with conn:
		conn.execute(
			"INSERT INTO users (username, password, role, full_name, email, assigned_class) VALUES (?,?,?,?,?,?)",
			(username, password, role, full_name, email, assigned_class)
		)
	conn.close()


def get_events_for_student(student_id: str, limit: int = 10, cfg: AppConfig | None = None) -> List[Dict[str, Any]]:
	"""Get events for a specific student"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("SELECT * FROM events WHERE student_id = ? ORDER BY timestamp DESC LIMIT ?", (student_id, limit)).fetchall()
	conn.close()
	return [dict(r) for r in rows]


def get_policy_settings(cfg: AppConfig | None = None) -> Dict[str, Any]:
	"""Get all policy settings from DB"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("SELECT key, value FROM settings WHERE key LIKE 'policy_%'").fetchall()
	conn.close()
	
	settings = {}
	for row in rows:
		key = row['key']
		value = row['value']
		# Parse boolean values
		if value.lower() in ('true', 'false'):
			settings[key] = value.lower() == 'true'
		# Try to parse as integer
		elif value.isdigit():
			settings[key] = int(value)
		else:
			settings[key] = value
	
	return settings


def update_policy_settings(policy_dict: Dict[str, Any], cfg: AppConfig | None = None) -> None:
	"""Update policy settings in DB"""
	conn = get_conn(cfg)
	with conn:
		for key, value in policy_dict.items():
			if key.startswith('policy_'):
				conn.execute(
					"INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
					(key, str(value))
				)
	conn.close()


def add_department(name: str, short_form: str, num_classes: int, cfg: AppConfig | None = None) -> int:
	"""Add a new department"""
	conn = get_conn(cfg)

	# Generate department ID (2-digit zero-padded sequential number)
	with conn:
		# Get the next department ID
		result = conn.execute("SELECT COUNT(*) FROM departments").fetchone()
		next_id = result[0] + 1
		department_id = f"{next_id:02d}"  # Zero-padded to 2 digits

		cur = conn.execute(
			"INSERT INTO departments (department_id, name, short_form, num_classes) VALUES (?,?,?,?)",
			(department_id, name, short_form, num_classes)
		)
		return cur.lastrowid


def get_all_departments(cfg: AppConfig | None = None) -> List[Dict[str, Any]]:
	"""Get all departments with student counts"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row

	# Get departments with student counts
	rows = conn.execute("""
		SELECT d.*,
			   COUNT(s.id) as students
		FROM departments d
		LEFT JOIN students s ON s.department = d.name
		GROUP BY d.id, d.name, d.short_form, d.num_classes, d.class_prefix, d.created_at
		ORDER BY d.name
	""").fetchall()

	conn.close()
	return [dict(r) for r in rows]


def get_department_by_id(dept_id: int, cfg: AppConfig | None = None) -> Optional[Dict[str, Any]]:
	"""Get department by ID"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	row = conn.execute("SELECT * FROM departments WHERE id=?", (dept_id,)).fetchone()
	conn.close()
	return dict(row) if row else None


def update_department(dept_id: int, name: str, short_form: str, num_classes: int, cfg: AppConfig | None = None) -> None:
	"""Update department"""
	conn = get_conn(cfg)
	with conn:
		conn.execute(
			"UPDATE departments SET name=?, short_form=?, num_classes=? WHERE id=?",
			(name, short_form, num_classes, dept_id)
		)
	conn.close()


def generate_student_id(batch_year: str, department_id: str, class_section: str, student_number: str, cfg: AppConfig | None = None) -> str:
	"""Generate student ID in format: YYDDCSNN
	- YY: Last 2 digits of batch year (e.g., 23 for 2023)
	- DD: Department ID (2-digit, e.g., 01)
	- C: Class section (1-digit, e.g., 1 for A, 2 for B)
	- NN: Student number (2-digit, e.g., 08)
	"""
	# Convert batch year to last 2 digits
	batch_short = batch_year[-2:] if len(batch_year) >= 2 else batch_year.zfill(2)

	# Ensure department_id is 2 digits
	dept_id = department_id.zfill(2)

	# Convert class section to number (A=1, B=2, C=3, etc.)
	if class_section.isalpha():
		class_num = str(ord(class_section.upper()) - ord('A') + 1)
	else:
		class_num = class_section

	# Ensure student number is 2 digits
	student_num = student_number.zfill(2)

	return f"{batch_short}{dept_id}{class_num}{student_num}"


def parse_student_id(student_id: str, cfg: AppConfig | None = None) -> Dict[str, str]:
	"""Parse student ID and return components"""
	if len(student_id) != 7:
		return {}

	try:
		batch_year = f"20{student_id[:2]}"  # Assume 20xx format
		department_id = student_id[2:4]
		class_section_num = int(student_id[4:5])
		student_number = student_id[5:7]

		# Convert class number back to letter (1=A, 2=B, 3=C, etc.)
		class_section = chr(ord('A') + class_section_num - 1)

		return {
			'batch_year': batch_year,
			'department_id': department_id,
			'class_section': class_section,
			'student_number': student_number,
			'full_id': student_id
		}
	except:
		return {}


def delete_department(dept_id: int, cfg: AppConfig | None = None) -> None:
	"""Delete department"""
	conn = get_conn(cfg)
	with conn:
		conn.execute("DELETE FROM departments WHERE id=?", (dept_id,))
	conn.close()


def clear_all_student_data(cfg: AppConfig | None = None) -> None:
	"""Clear all student-related data from the database"""
	conn = get_conn(cfg)
	with conn:
		# Clear all tables in order of dependencies (foreign keys)
		conn.execute("DELETE FROM geofence_events")
		conn.execute("DELETE FROM emergency_alerts")
		conn.execute("DELETE FROM access_log")
		conn.execute("DELETE FROM unauthorized_access")
		conn.execute("DELETE FROM events")
		conn.execute("DELETE FROM students")
		conn.execute("DELETE FROM users")
		conn.execute("DELETE FROM departments")
		conn.execute("DELETE FROM settings WHERE key LIKE 'policy_%'")
	conn.close()


def delete_student(student_id: str, cfg: AppConfig | None = None) -> None:
	"""Delete a single student and all related records (events, logs, alerts)."""
	conn = get_conn(cfg)
	with conn:
		# Remove related rows first to avoid orphan records
		conn.execute("DELETE FROM geofence_events WHERE student_id = ?", (student_id,))
		conn.execute("DELETE FROM emergency_alerts WHERE student_id = ?", (student_id,))
		conn.execute("DELETE FROM access_log WHERE student_id = ?", (student_id,))
		conn.execute("DELETE FROM unauthorized_access WHERE student_id = ?", (student_id,))
		conn.execute("DELETE FROM events WHERE student_id = ?", (student_id,))
		conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
	conn.close()


def update_student_face_embedding(student_id: str, embedding: list | None, cfg: AppConfig | None = None) -> None:
	conn = get_conn(cfg)
	with conn:
		conn.execute(
			"UPDATE students SET face_embedding = ? WHERE id = ?",
			(json.dumps(embedding) if embedding is not None else None, student_id)
		)
	conn.close()


def get_student_face_embedding(student_id: str, cfg: AppConfig | None = None) -> Optional[list]:
	conn = get_conn(cfg)
	row = conn.execute("SELECT face_embedding FROM students WHERE id = ?", (student_id,)).fetchone()
	conn.close()
	if not row or row[0] is None:
		return None
	try:
		return json.loads(row[0])
	except Exception:
		return None


def get_system_settings(cfg: AppConfig | None = None) -> Dict[str, Any]:
	"""Get all system settings from DB"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("SELECT key, value FROM settings WHERE key LIKE 'system_%'").fetchall()
	conn.close()

	settings = {}
	for row in rows:
		key = row['key']
		value = row['value']
		# Parse boolean values
		if value.lower() in ('true', 'false'):
			settings[key] = value.lower() == 'true'
		# Try to parse as float
		elif '.' in value and value.replace('.', '').replace('-', '').isdigit():
			try:
				settings[key] = float(value)
			except ValueError:
				settings[key] = value
		# Try to parse as integer
		elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
			settings[key] = int(value)
		else:
			settings[key] = value

	return settings


def update_system_settings(settings_dict: Dict[str, Any], cfg: AppConfig | None = None) -> None:
	"""Update system settings in DB"""
	conn = get_conn(cfg)
	with conn:
		for key, value in settings_dict.items():
			if key.startswith('system_'):
				conn.execute(
					"INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
					(key, str(value))
				)
	conn.close()
