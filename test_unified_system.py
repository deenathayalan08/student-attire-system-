"""
Test script to verify unified student/user system
"""
import sqlite3

def test_unified_system():
    conn = sqlite3.connect('data/attire.db')
    cursor = conn.cursor()
    
    print("=" * 60)
    print("UNIFIED STUDENT/USER SYSTEM TEST")
    print("=" * 60)
    
    # Get all students
    cursor.execute("SELECT id, name, email FROM students")
    students = cursor.fetchall()
    
    print(f"\n📚 Total Students: {len(students)}")
    
    if students:
        print("\nStudent Records:")
        for student in students:
            student_id, name, email = student
            print(f"  • {student_id} - {name} ({email})")
            
            # Check if user account exists
            cursor.execute("SELECT username, role FROM users WHERE username = ?", (student_id,))
            user = cursor.fetchone()
            
            if user:
                print(f"    ✅ Login Account: YES (Username: {user[0]}, Role: {user[1]})")
            else:
                print(f"    ❌ Login Account: NO")
    else:
        print("  No students found")
    
    # Get all users
    cursor.execute("SELECT username, role, full_name FROM users")
    users = cursor.fetchall()
    
    print(f"\n👥 Total Users: {len(users)}")
    
    if users:
        print("\nUser Accounts:")
        for user in users:
            username, role, full_name = user
            print(f"  • {username} - {full_name} (Role: {role})")
    else:
        print("  No users found")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    conn.close()

if __name__ == "__main__":
    test_unified_system()
