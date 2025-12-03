import sqlite3
import os

if not os.path.exists('data/attire.db'):
    print('❌ Database not found at data/attire.db')
    print('Please run the application first to initialize the database.')
    exit(1)

try:
    conn = sqlite3.connect('data/attire.db')
    cursor = conn.cursor()
except sqlite3.Error as e:
    print(f'❌ Database connection error: {e}')
    exit(1)

print('=== STUDENTS TABLE ===')
cursor.execute('SELECT id, name, email, verified FROM students')
students = cursor.fetchall()
if students:
    for row in students:
        print(f'ID: {row[0]}, Name: {row[1]}, Email: {row[2]}, Verified: {row[3]}')
else:
    print('No students found')

print('\n=== USERS TABLE ===')
try:
    cursor.execute('SELECT username, role, full_name, email FROM users')
    users = cursor.fetchall()
    if users:
        for row in users:
            print(f'Username: {row[0]}, Role: {row[1]}, Name: {row[2]}, Email: {row[3]}')
    else:
        print('No users found')
except sqlite3.Error as e:
    print(f'❌ Error reading users table: {e}')

conn.close()
