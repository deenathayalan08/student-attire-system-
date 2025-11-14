#!/usr/bin/env python3
"""
Script to run the student attire verification project.
"""

import os
import sys
import subprocess

def main():
    print("=== Student Attire Verification System ===\n")

    print("Available options:")
    print("1. Run Streamlit App (Web Interface)")
    print("2. Train Model")
    print("3. Import Dataset")
    print("4. Preprocess Data (Augmentation)")
    print("5. Check Dataset Accuracy")
    print("6. Evaluate Model")
    print("7. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-7): ").strip()

            if choice == "1":
                print("\nStarting Streamlit App...")
                print("Open http://localhost:8501 in your browser")
                subprocess.run([sys.executable, "app/streamlit_app.py"])

            elif choice == "2":
                print("\nTraining model...")
                subprocess.run([sys.executable, "train_model.py"])

            elif choice == "3":
                print("\nImporting dataset...")
                subprocess.run([sys.executable, "import_all_datasets.py"])

            elif choice == "4":
                print("\nPreprocessing data with augmentation...")
                subprocess.run([sys.executable, "preprocess_data.py"])

            elif choice == "5":
                print("\nChecking dataset accuracy...")
                subprocess.run([sys.executable, "debug_accuracy.py"])

            elif choice == "6":
                print("\nEvaluating model...")
                subprocess.run([sys.executable, "evaluate_dataset.py"])

            elif choice == "7":
                print("\nGoodbye!")
                break

            else:
                print("Invalid choice. Please enter 1-7.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
