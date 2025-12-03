"""
Clear all student data from the database
This will remove:
- All students from students table
- All student users from users table
- All events
- All face images
"""

import sqlite3
import os
import shutil
from pathlib import Path

def clear_student_data():
    """Clear all student data from database and files"""
    
    # Database path
    db_path = Path("data/attire.db")
    
    if not db_path.exists():
        print("❌ Database not found at data/attire.db")
        return
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Get counts before deletion
        student_count = cursor.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        user_count = cursor.execute("SELECT COUNT(*) FROM users WHERE role='student'").fetchone()[0]
        event_count = cursor.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        
        print(f"\n📊 Current Data:")
        print(f"   Students: {student_count}")
        print(f"   Student Users: {user_count}")
        print(f"   Events: {event_count}")
        
        # Confirm deletion
        print(f"\n⚠️  WARNING: This will delete ALL student data!")
        confirm = input("Type 'YES' to confirm deletion: ")
        
        if confirm != "YES":
            print("❌ Deletion cancelled")
            return
        
        print("\n🗑️  Deleting data...")
        
        # Delete from tables
        cursor.execute("DELETE FROM events")
        print(f"   ✅ Deleted {event_count} events")
        
        cursor.execute("DELETE FROM students")
        print(f"   ✅ Deleted {student_count} students")
        
        cursor.execute("DELETE FROM users WHERE role='student'")
        print(f"   ✅ Deleted {user_count} student users")
        
        # Delete related tables if they exist
        try:
            cursor.execute("DELETE FROM unauthorized_access")
            print(f"   ✅ Cleared unauthorized_access table")
        except:
            pass
        
        try:
            cursor.execute("DELETE FROM access_log")
            print(f"   ✅ Cleared access_log table")
        except:
            pass
        
        try:
            cursor.execute("DELETE FROM emergency_alerts")
            print(f"   ✅ Cleared emergency_alerts table")
        except:
            pass
        
        # Commit changes
        conn.commit()
        
        # Clear face storage directory
        face_storage = Path("data/face_storage")
        if face_storage.exists():
            file_count = len(list(face_storage.glob("*")))
            shutil.rmtree(face_storage)
            face_storage.mkdir(exist_ok=True)
            print(f"   ✅ Deleted {file_count} face images")
        
        print("\n✅ All student data has been cleared!")
        print("   You can now register new students.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        conn.rollback()
    
    finally:
        conn.close()


if __name__ == "__main__":
    print("=" * 50)
    print("  CLEAR ALL STUDENT DATA")
    print("=" * 50)
    clear_student_data()
