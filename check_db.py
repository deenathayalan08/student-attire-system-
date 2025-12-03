import sqlite3
import os

if not os.path.exists('data/attire.db'):
    print('❌ Database not found at data/attire.db')
    print('Please run the application first to initialize the database.')
    exit(1)

try:
    conn = sqlite3.connect('data/attire.db')
    cursor = conn.cursor()

    print('Students:', cursor.execute('SELECT COUNT(*) FROM students').fetchone()[0])
    print('Users:', cursor.execute('SELECT COUNT(*) FROM users').fetchone()[0])
except sqlite3.Error as e:
    print(f'❌ Database error: {e}')
    exit(1)

print('\nUsers table:')
for row in cursor.execute('SELECT username, role, email FROM users').fetchall():
    print(f'  {row[0]} - {row[1]} - {row[2]}')

print('\nStudents table:')
for row in cursor.execute('SELECT id, name, email FROM students').fetchall():
    print(f'  {row[0]} - {row[1]} - {row[2]}')

conn.close()
except Exception as e:
    print(f'❌ Error: {e}')
    if 'conn' in locals():
        conn.close()
    exit(1)
