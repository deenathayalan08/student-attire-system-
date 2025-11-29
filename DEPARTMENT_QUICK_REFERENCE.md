# 🔧 Department Management - Quick Reference Guide

## Quick Start

### Add a Department Programmatically
```python
from src.db import add_department
from src.config import AppConfig

cfg = AppConfig()
dept_data = {
    "name": "Computer Science",
    "code": "CS",
    "short_form": "CS",
    "head_name": "Prof. Smith",
    "number_of_classes": 3,
    "location": "Block A",
    "email": "cs@college.edu",
    "phone": "555-1234",
    "description": "Computer Science Department"
}

success, dept_id, message = add_department(dept_data, cfg)
if success:
    print(f"Department created with ID: {dept_id}")
    print("Classes created: CS-A, CS-B, CS-C")
```

### Get All Departments
```python
from src.db import get_all_departments

depts = get_all_departments(cfg)
for dept in depts:
    print(f"{dept['name']} ({dept['code']}) - {dept['total_students']} students")
```

### Get Department Statistics
```python
from src.db import get_department_statistics

stats = get_department_statistics(dept_id=1, cfg=cfg)
print(f"Total: {stats['total_students']}")
print(f"Male: {stats['male_count']}, Female: {stats['female_count']}")
print(f"Classes: {len(stats['classes'])}")
```

### Get Students in Department
```python
from src.db import get_students_by_department

students = get_students_by_department("Computer Science", cfg=cfg)
for student in students:
    print(f"{student['name']} ({student['gender']}) - {student['class']}")
```

### Update Department
```python
from src.db import update_department

update_data = {
    "head_name": "Prof. Johnson",
    "location": "Block B",
    "description": "Updated description"
}
success, msg = update_department(dept_id=1, dept_data=update_data, cfg=cfg)
```

### Update Class Advisor
```python
from src.db import update_class_advisor

success, msg = update_class_advisor(class_id=1, advisor_name="Prof. Kumar", cfg=cfg)
```

### Search Departments
```python
from src.db import search_departments

results = search_departments("CS", cfg=cfg)  # Search by name or code
for dept in results:
    print(dept['name'])
```

### Export Department Report
```python
from src.db import export_department_report

csv_data = export_department_report(dept_id=1, cfg=cfg)
with open("department_report.csv", "w") as f:
    f.write(csv_data)
```

---

## Database Schema Reference

### Departments Table
```sql
SELECT * FROM departments;

Columns:
- id (INTEGER PRIMARY KEY)
- name (TEXT UNIQUE)
- code (TEXT UNIQUE)
- short_form (TEXT)
- head_name (TEXT)
- head_email (TEXT)
- number_of_classes (INTEGER)
- location (TEXT)
- email (TEXT)
- phone (TEXT)
- description (TEXT)
- status (TEXT: 'active'/'inactive')
- created_at (DATETIME)
- updated_at (DATETIME)
```

### Classes Table
```sql
SELECT * FROM classes;

Columns:
- id (INTEGER PRIMARY KEY)
- department_id (INTEGER FK → departments.id)
- class_letter (TEXT: 'A', 'B', 'C', ...)
- class_code (TEXT UNIQUE: 'CS-A', 'CS-B', ...)
- class_advisor (TEXT)
- room_number (TEXT)
- capacity (INTEGER)
- current_enrollment (INTEGER)
- status (TEXT: 'active'/'inactive')
- created_at (DATETIME)
```

### Students Table (Updated)
```sql
SELECT * FROM students;

New Column:
- gender (TEXT: 'M', 'F', 'U' [Unknown])
```

---

## Useful SQL Queries

### Count Students by Department
```sql
SELECT 
    d.name, 
    COUNT(s.id) as total,
    SUM(CASE WHEN s.gender = 'M' THEN 1 ELSE 0 END) as male,
    SUM(CASE WHEN s.gender = 'F' THEN 1 ELSE 0 END) as female
FROM departments d
LEFT JOIN students s ON s.department = d.name
GROUP BY d.id
ORDER BY d.name;
```

### Students per Class
```sql
SELECT 
    c.class_code,
    COUNT(s.id) as enrollment
FROM classes c
LEFT JOIN students s ON s.class = c.class_code
GROUP BY c.id;
```

### Classes without Advisor
```sql
SELECT * FROM classes WHERE class_advisor IS NULL;
```

### All Departments with Class Count
```sql
SELECT 
    d.name,
    d.code,
    COUNT(c.id) as total_classes,
    COUNT(DISTINCT s.id) as total_students
FROM departments d
LEFT JOIN classes c ON c.department_id = d.id
LEFT JOIN students s ON s.department = d.name
WHERE d.status = 'active'
GROUP BY d.id;
```

---

## Troubleshooting

### Issue: "Department code already exists"
**Solution:** Use a different department code, or let system auto-generate

### Issue: "No classes created"
**Solution:** Ensure `number_of_classes` > 0. Classes generated as A, B, C... up to Z

### Issue: Gender statistics showing all "Unknown"
**Solution:** Ensure students have gender field set ('M', 'F', or 'U')

### Issue: Department not showing in list
**Solution:** Check if status = 'active'. Soft-deleted depts have status = 'inactive'

### Issue: Export CSV empty
**Solution:** Ensure department has students. CSV only includes enrolled students.

---

## Performance Tips

1. **Avoid frequent schema queries** - Use cached department list
2. **Batch operations** - Create multiple departments in loop if needed
3. **Index commonly searched fields:**
   ```sql
   CREATE INDEX idx_dept_code ON departments(code);
   CREATE INDEX idx_dept_status ON departments(status);
   CREATE INDEX idx_class_dept ON classes(department_id);
   CREATE INDEX idx_student_dept ON students(department);
   ```

4. **Limit results** when exporting large departments
5. **Archive old departments** - Mark as 'inactive' instead of deleting

---

## Common Workflows

### Workflow 1: Create Department with Students
```python
# 1. Create department
success, dept_id, msg = add_department({
    "name": "Mechanical Engineering",
    "code": "ME",
    "number_of_classes": 2
}, cfg)

# 2. Get created classes
classes = get_classes_by_department(dept_id, cfg)
class_a_code = classes[0]['class_code']  # "ME-A"

# 3. Add students to that class
for i, student_data in enumerate(students_list):
    student_data['department'] = "Mechanical Engineering"
    student_data['class'] = class_a_code
    add_student(student_data, cfg)
```

### Workflow 2: Department Head Report
```python
# 1. Get department
dept = get_department_by_id(dept_id, cfg)

# 2. Get statistics
stats = get_department_statistics(dept_id, cfg)

# 3. Export
csv_report = export_department_report(dept_id, cfg)

# 4. Email to head (future feature)
# send_email(dept['head_email'], "Department Report", csv_report)
```

### Workflow 3: Compliance by Department
```python
from src.db import get_compliance_stats, get_students_by_department

# Global compliance
global_stats = get_compliance_stats(cfg=cfg)

# Per-department compliance
students = get_students_by_department("Computer Science", cfg)
dept_students = [s['id'] for s in students]

# Query events for those students
# (custom query needed - not in current db)
```

---

## API Reference

### `add_department(dept_data, cfg=None)`
Creates a new department and auto-generates classes

**Parameters:**
- `dept_data` (Dict): Department information
- `cfg` (AppConfig, optional): Configuration

**Returns:** `(success: bool, dept_id: int, message: str)`

**Raises:** `sqlite3.IntegrityError` if code/name duplicate

---

### `get_all_departments(cfg=None)`
Retrieves all active departments with student count

**Parameters:**
- `cfg` (AppConfig, optional): Configuration

**Returns:** `List[Dict]` with keys: id, name, code, ..., total_students

---

### `get_department_statistics(dept_id, cfg=None)`
Gets detailed statistics for a department

**Parameters:**
- `dept_id` (int): Department ID
- `cfg` (AppConfig, optional): Configuration

**Returns:** `Dict` with keys: total_students, male_count, female_count, unknown_count, classes

---

### `search_departments(search_term, cfg=None)`
Searches departments by name or code

**Parameters:**
- `search_term` (str): Search query
- `cfg` (AppConfig, optional): Configuration

**Returns:** `List[Dict]` matching departments

---

### `export_department_report(dept_id, cfg=None)`
Exports department data as CSV

**Parameters:**
- `dept_id` (int): Department ID
- `cfg` (AppConfig, optional): Configuration

**Returns:** `str` (CSV formatted data) or `None` if not found

---

### `delete_department(dept_id, cfg=None)`
Soft-deletes a department (marks as inactive)

**Parameters:**
- `dept_id` (int): Department ID
- `cfg` (AppConfig, optional): Configuration

**Returns:** `(success: bool, message: str)`

---

## Integration Checklist

- [x] Database tables created
- [x] Migration system working
- [x] UI tabs added to admin dashboard
- [x] Forms functional
- [x] Search implemented
- [x] Export working
- [x] Statistics calculated
- [x] Gender tracking functional
- [ ] Email notifications (future)
- [ ] Department head login (future)
- [ ] Mobile app support (future)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Nov 29, 2025 | Initial implementation - All phases complete |

---

For issues or questions, refer to `DEPARTMENT_FEATURE_IMPLEMENTATION.md` for full documentation.
