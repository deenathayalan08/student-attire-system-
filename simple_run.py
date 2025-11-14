#!/usr/bin/env python3
"""
Simple runner for student attire verification project.
"""

import os
import sys
import subprocess

def main():
    print("=== Student Attire Verification System ===\n")

    print("1. Launch Web App (Streamlit)")
    print("2. Check Dataset Accuracy")
    print("q. Quit")

    while True:
        try:
            choice = input("\nEnter choice (1, 2, or q): ").strip().lower()

            if choice == "1":
                print("\nLaunching Streamlit Web App...")
                print("Open http://localhost:8501 in your browser")
                print("Press Ctrl+C to stop the server\n")
                subprocess.run([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py"])

            elif choice == "2":
                print("\nChecking Dataset Accuracy...")
                subprocess.run([sys.executable, "debug_accuracy.py"])

            elif choice == "q":
                print("\nGoodbye!")
                break

            else:
                print("Invalid choice. Enter 1, 2, or q.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
