#!/usr/bin/env python3
"""Verify all project dependencies are installed and working.

Usage:
    python check_setup.py

Output: ✅ or ❌ for each package, with fixes suggested if missing.
"""

import sys
from pathlib import Path

print("🔍 Checking Student Attire Detection System Setup...\n")

# Check Python version
print(f"Python Version: {sys.version}")
if sys.version_info < (3, 8):
    print("❌ ERROR: Python 3.8+ required!")
    sys.exit(1)
print("✅ Python version OK\n")

# List of required packages
packages = {
    'cv2': 'opencv-python',
    'streamlit': 'streamlit',
    'sklearn': 'scikit-learn',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'joblib': 'joblib',
    'tqdm': 'tqdm',
    'PIL': 'Pillow',
    'yaml': 'pyyaml',
}

print("Checking packages:")
print("-" * 50)

failed = []

for import_name, package_name in packages.items():
    try:
        __import__(import_name)
        print(f"✅ {package_name:20} installed")
    except ImportError:
        print(f"❌ {package_name:20} NOT installed")
        failed.append(package_name)

print("-" * 50)

if failed:
    print(f"\n❌ Missing {len(failed)} package(s):")
    for pkg in failed:
        print(f"   • {pkg}")
    
    print("\n🔧 Fix it with:")
    print("   pip install " + " ".join(failed))
    print("\n   Or reinstall everything:")
    print("   pip install -r requirements.txt --upgrade")
else:
    print("\n✅ All packages installed!")

# Check key files
print("\n\nChecking project files:")
print("-" * 50)

key_files = {
    'app/streamlit_app.py': 'Web interface',
    'src/model.py': 'ML model',
    'src/features.py': 'Feature extraction',
    'evaluate_dataset.py': 'Dataset evaluator',
    'requirements.txt': 'Dependencies',
}

ROOT = Path(__file__).resolve().parent
missing_files = []

for file_path, description in key_files.items():
    full_path = ROOT / file_path
    if full_path.exists():
        print(f"✅ {file_path:30} {description}")
    else:
        print(f"❌ {file_path:30} {description} — NOT FOUND")
        missing_files.append(file_path)

print("-" * 50)

if missing_files:
    print(f"\n❌ Missing {len(missing_files)} file(s):")
    for f in missing_files:
        print(f"   • {f}")
else:
    print("\n✅ All key files present!")

# Check datasets
print("\n\nChecking datasets:")
print("-" * 50)

datasets = {
    'datasets/uniform 1': 'Main dataset',
    'datasets/uniform 2': 'Backup dataset',
}

datasets_ok = True
for ds_path, description in datasets.items():
    full_path = ROOT / ds_path
    if full_path.exists():
        img_count = len(list(full_path.rglob('*.jpg'))) + len(list(full_path.rglob('*.png')))
        print(f"✅ {ds_path:30} {description} ({img_count} images)")
    else:
        print(f"❌ {ds_path:30} {description} — NOT FOUND")
        datasets_ok = False

print("-" * 50)

# Final summary
print("\n" + "="*50)
if not failed and not missing_files and datasets_ok:
    print("✅ ✅ ✅  SETUP OK! YOU'RE READY TO GO!  ✅ ✅ ✅")
    print("\nNext step:")
    print("  streamlit run app/streamlit_app.py")
    print("\nThen open: http://localhost:8501")
elif not failed and not missing_files:
    print("⚠️  ALMOST READY")
    print("\nIssue: Datasets not found (can restore from archive/)")
    print("\nTo run anyway:")
    print("  streamlit run app/streamlit_app.py")
else:
    print("❌ SETUP INCOMPLETE")
    print("\nFix errors above, then run:")
    print("  python check_setup.py")
print("="*50)
