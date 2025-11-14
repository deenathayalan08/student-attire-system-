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
                print("Choose import type:")
                print("a. Individual datasets (limited samples)")
                print("b. Bulk import (all available datasets)")
                import_choice = input("Enter a or b: ").strip().lower()
                if import_choice == "a":
                    subprocess.run([sys.executable, "import_datasets.py", "--individual"])
                elif import_choice == "b":
                    subprocess.run([sys.executable, "import_datasets.py", "--bulk"])
                else:
                    print("Invalid choice, skipping import.")

            elif choice == "4":
                print("\nPreprocessing data with augmentation...")
                subprocess.run([sys.executable, "preprocess_data.py"])

            elif choice == "5":
                print("\nChecking dataset accuracy...")
                print("Choose accuracy check type:")
                print("a. Basic accuracy check")
                print("b. Comprehensive metrics")
                print("c. Debug mode (balanced, no warnings)")
                acc_choice = input("Enter a, b, or c: ").strip().lower()
                if acc_choice == "a":
                    subprocess.run([sys.executable, "evaluate_model.py", "--basic"])
                elif acc_choice == "b":
                    subprocess.run([sys.executable, "evaluate_model.py", "--comprehensive"])
                elif acc_choice == "c":
                    subprocess.run([sys.executable, "evaluate_model.py", "--debug"])
                else:
                    print("Invalid choice, skipping accuracy check.")

            elif choice == "6":
                print("\nEvaluating model...")
                print("Choose evaluation type:")
                print("a. Basic evaluation")
                print("b. Comprehensive evaluation")
                print("c. Debug evaluation")
                eval_choice = input("Enter a, b, or c: ").strip().lower()
                if eval_choice == "a":
                    subprocess.run([sys.executable, "evaluate_model.py", "--basic"])
                elif eval_choice == "b":
                    subprocess.run([sys.executable, "evaluate_model.py", "--comprehensive"])
                elif eval_choice == "c":
                    subprocess.run([sys.executable, "evaluate_model.py", "--debug"])
                else:
                    print("Invalid choice, skipping evaluation.")

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
