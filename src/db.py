import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import AppConfig


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
	id TEXT PRIMARY KEY,
	name TEXT,
	class TEXT,
	department TEXT,
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
	name TEXT UNIQUE NOT NULL,
	code TEXT UNIQUE NOT NULL,
	short_form TEXT,
	head_name TEXT,
	head_email TEXT,
	number_of_classes INTEGER DEFAULT 1,
	location TEXT,
	email TEXT,
	phone TEXT,
	description TEXT,
	status TEXT DEFAULT 'active',
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS classes (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	department_id INTEGER NOT NULL,
	class_letter TEXT NOT NULL,
	class_code TEXT UNIQUE NOT NULL,
	class_advisor TEXT,
	room_number TEXT,
	capacity INTEGER DEFAULT 50,
	current_enrollment INTEGER DEFAULT 0,
	status TEXT DEFAULT 'active',
	created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
	FOREIGN KEY(department_id) REFERENCES departments(id)
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
		_ensure_column("students", "uniform_type", "uniform_type TEXT")
		_ensure_column("students", "email", "email TEXT")
		_ensure_column("students", "phone", "phone TEXT")
		_ensure_column("students", "verified", "verified INTEGER DEFAULT 0")
		_ensure_column("students", "contact_info", "contact_info TEXT")
		_ensure_column("students", "gender", "gender TEXT DEFAULT 'U'")  # M=Male, F=Female, U=Unknown


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
			"INSERT OR REPLACE INTO students (id, name, class, department, uniform_type, email, phone, contact_info) VALUES (?,?,?,?,?,?,?,?)",
			(
				student_data.get("id"),
				student_data.get("name"),
				student_data.get("class"),
				student_data.get("department"),
				student_data.get("uniform_type"),
				student_data.get("email"),
				student_data.get("phone"),
				student_data.get("contact_info")
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


# ==================== DEPARTMENT MANAGEMENT ====================

def add_department(dept_data: Dict[str, Any], cfg: AppConfig | None = None) -> Tuple[bool, int, str]:
	"""Add a new department and create classes"""
	conn = get_conn(cfg)
	try:
		with conn:
			# Insert department
			cur = conn.execute(
				"""INSERT INTO departments
				(name, code, short_form, head_name, head_email, number_of_classes,
				 location, email, phone, description, status)
				VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
				(
					dept_data.get("name"),
					dept_data.get("code"),
					dept_data.get("short_form"),
					dept_data.get("head_name"),
					dept_data.get("head_email"),
					dept_data.get("number_of_classes", 1),
					dept_data.get("location"),
					dept_data.get("email"),
					dept_data.get("phone"),
					dept_data.get("description"),
					"active"
				)
			)
			dept_id = cur.lastrowid
			
			# Create classes
			num_classes = int(dept_data.get("number_of_classes", 1))
			for i in range(num_classes):
				class_letter = chr(65 + i)  # A, B, C, D...
				class_code = f"{dept_data.get('code')}-{class_letter}"
				conn.execute(
					"""INSERT INTO classes 
					(department_id, class_letter, class_code, status) 
					VALUES (?,?,?,?)""",
					(dept_id, class_letter, class_code, "active")
				)
		conn.close()
		return True, dept_id, "Department created successfully"
	except sqlite3.IntegrityError as e:
		conn.close()
		return False, 0, str(e)
	except Exception as e:
		conn.close()
		return False, 0, str(e)


def get_all_departments(cfg: AppConfig | None = None) -> List[Dict[str, Any]]:
	"""Get all departments with student count"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("""
		SELECT d.*, COUNT(DISTINCT s.id) as total_students
		FROM departments d
		LEFT JOIN students s ON s.department = d.name
		WHERE d.status = 'active'
		GROUP BY d.id
		ORDER BY d.name
	""").fetchall()
	conn.close()
	return [dict(r) for r in rows]


def get_department_by_id(dept_id: int, cfg: AppConfig | None = None) -> Optional[Dict[str, Any]]:
	"""Get department by ID"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	row = conn.execute("SELECT * FROM departments WHERE id = ?", (dept_id,)).fetchone()
	conn.close()
	return dict(row) if row else None


def get_classes_by_department(dept_id: int, cfg: AppConfig | None = None) -> List[Dict[str, Any]]:
	"""Get all classes in a department"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("""
		SELECT c.*, COUNT(DISTINCT s.id) as student_count
		FROM classes c
		LEFT JOIN students s ON s.class = c.class_code
		WHERE c.department_id = ? AND c.status = 'active'
		GROUP BY c.id
		ORDER BY c.class_letter
	""", (dept_id,)).fetchall()
	conn.close()
	return [dict(r) for r in rows]


def get_students_by_department(dept_name: str, cfg: AppConfig | None = None) -> List[Dict[str, Any]]:
	"""Get all students in a department with gender breakdown"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("""
		SELECT * FROM students WHERE department = ?
		ORDER BY class, name
	""", (dept_name,)).fetchall()
	conn.close()
	return [dict(r) for r in rows]


def get_department_statistics(dept_id: int, cfg: AppConfig | None = None) -> Dict[str, Any]:
	"""Get detailed statistics for a department"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	
	# Get department info
	dept = conn.execute("SELECT * FROM departments WHERE id = ?", (dept_id,)).fetchone()
	if not dept:
		conn.close()
		return {}
	
	dept_name = dept['name']
	
	# Get student statistics
	students = conn.execute("""
		SELECT gender FROM students WHERE department = ?
	""", (dept_name,)).fetchall()
	
	total_students = len(students)
	male_count = sum(1 for s in students if s['gender'] == 'M')
	female_count = sum(1 for s in students if s['gender'] == 'F')
	unknown_count = total_students - male_count - female_count
	
	# Get class statistics
	classes = conn.execute("""
		SELECT c.*, COUNT(DISTINCT s.id) as student_count
		FROM classes c
		LEFT JOIN students s ON s.class = c.class_code
		WHERE c.department_id = ?
		GROUP BY c.id
	""", (dept_id,)).fetchall()
	
	conn.close()
	
	return {
		"total_students": total_students,
		"male_count": male_count,
		"female_count": female_count,
		"unknown_count": unknown_count,
		"classes": [dict(c) for c in classes]
	}


def update_department(dept_id: int, dept_data: Dict[str, Any], cfg: AppConfig | None = None) -> Tuple[bool, str]:
	"""Update department information"""
	conn = get_conn(cfg)
	try:
		with conn:
			conn.execute("""
				UPDATE departments SET
				name = ?, code = ?, short_form = ?, head_name = ?,
				head_email = ?, location = ?, email = ?, phone = ?,
				description = ?, updated_at = CURRENT_TIMESTAMP
				WHERE id = ?
			""", (
				dept_data.get("name"),
				dept_data.get("code"),
				dept_data.get("short_form"),
				dept_data.get("head_name"),
				dept_data.get("head_email"),
				dept_data.get("location"),
				dept_data.get("email"),
				dept_data.get("phone"),
				dept_data.get("description"),
				dept_data.get("department_type", "Academic"),
				dept_id
			))
		conn.close()
		return True, "Department updated successfully"
	except Exception as e:
		conn.close()
		return False, str(e)


def delete_department(dept_id: int, cfg: AppConfig | None = None) -> Tuple[bool, str]:
	"""Delete a department (soft delete)"""
	conn = get_conn(cfg)
	try:
		with conn:
			conn.execute("UPDATE departments SET status = 'inactive' WHERE id = ?", (dept_id,))
			conn.execute("UPDATE classes SET status = 'inactive' WHERE department_id = ?", (dept_id,))
		conn.close()
		return True, "Department deleted successfully"
	except Exception as e:
		conn.close()
		return False, str(e)


def update_class_advisor(class_id: int, advisor_name: str, cfg: AppConfig | None = None) -> Tuple[bool, str]:
	"""Update class advisor"""
	conn = get_conn(cfg)
	try:
		with conn:
			conn.execute("UPDATE classes SET class_advisor = ? WHERE id = ?", (advisor_name, class_id))
		conn.close()
		return True, "Class advisor updated"
	except Exception as e:
		conn.close()
		return False, str(e)


def update_class_room(class_id: int, room_number: str, cfg: AppConfig | None = None) -> Tuple[bool, str]:
	"""Update class room number"""
	conn = get_conn(cfg)
	try:
		with conn:
			conn.execute("UPDATE classes SET room_number = ? WHERE id = ?", (room_number, class_id))
		conn.close()
		return True, "Room number updated"
	except Exception as e:
		conn.close()
		return False, str(e)


def search_departments(search_term: str, cfg: AppConfig | None = None) -> List[Dict[str, Any]]:
	"""Search departments by name or code"""
	conn = get_conn(cfg)
	conn.row_factory = sqlite3.Row
	rows = conn.execute("""
		SELECT d.*, COUNT(DISTINCT s.id) as total_students
		FROM departments d
		LEFT JOIN students s ON s.department = d.name
		WHERE (d.name LIKE ? OR d.code LIKE ?) AND d.status = 'active'
		GROUP BY d.id
		ORDER BY d.name
	""", (f"%{search_term}%", f"%{search_term}%")).fetchall()
	conn.close()
	return [dict(r) for r in rows]


def export_department_report(dept_id: int, cfg: AppConfig | None = None) -> Optional[str]:
	"""Export department data as CSV string"""
	import csv
	import io
	
	stats = get_department_statistics(dept_id, cfg)
	dept = get_department_by_id(dept_id, cfg)
	students = get_students_by_department(dept['name'], cfg) if dept else []
	
	if not dept:
		return None
	
	output = io.StringIO()
	writer = csv.writer(output)
	
	# Header
	writer.writerow(["Department Report"])
	writer.writerow([f"Department: {dept['name']}", f"Code: {dept['code']}"])
	writer.writerow([f"Head: {dept['head_name']}", f"Location: {dept['location']}"])
	writer.writerow([])
	
	# Statistics
	writer.writerow(["Statistics"])
	writer.writerow(["Total Students", "Male", "Female", "Unknown"])
	writer.writerow([stats["total_students"], stats["male_count"], stats["female_count"], stats["unknown_count"]])
	writer.writerow([])
	
	# Student details
	writer.writerow(["Student ID", "Name", "Class", "Gender", "Email"])
	for student in students:
		writer.writerow([
			student.get("id", ""),
			student.get("name", ""),
			student.get("class", ""),
			student.get("gender", ""),
			student.get("email", "")
		])
	
	return output.getvalue()

