# -*- coding: utf-8 -*-
"""
ENVIRONMENT & DATASET CHECKER (Advanced Pipeline)
-------------------------------------------------
Verifies all dependencies, CUDA GPU acceleration, and DeepSense 6G data files.
"""

import sys
import os
import platform
import importlib
from importlib.metadata import version, PackageNotFoundError

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "utm": "utm",
    "torch": "torch",
}

print("=" * 60)
print(" 6G BEAM PREDICTION - ADVANCED PIPELINE ENVIRONMENT CHECK")
print("=" * 60)
print(f"Python Executable : {sys.executable}")
print(f"Python Version    : {sys.version.split()[0]}")
print(f"Platform          : {platform.platform()}")
print("-" * 60)
print("1. Dependency Checks:")

all_ok = True
for mod_name, dist_name in REQUIRED_PACKAGES.items():
    v = "N/A"
    try:
        mod = importlib.import_module(mod_name)
        v = getattr(mod, "__version__", version(dist_name))
        status = "OK"
    except Exception as e:
        status = f"FAILED: {e}"
        all_ok = False
    print(f"   - {dist_name:<16} : {status:<10} ({v})")

print("-" * 60)
print("2. Hardware & CUDA Acceleration:")
try:
    import torch
    cuda_avail = torch.cuda.is_available()
    print(f"   - PyTorch Version  : {torch.__version__}")
    print(f"   - CUDA Available   : {cuda_avail}")
    if cuda_avail:
        print(f"   - GPU Device       : {torch.cuda.get_device_name(0)}")
        print(f"   - CUDA Version     : {torch.version.cuda}")
    else:
        print("   - NOTICE: No GPU detected. Training will run on CPU.")
except Exception as e:
    print(f"   - PyTorch check error: {e}")

print("-" * 60)
print("3. DeepSense 6G Dataset Detection:")
_cand_local = os.path.join(os.getcwd(), 'Gathered_data_DEV')
_cand_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Gathered_data_DEV'))
data_path = _cand_local if os.path.exists(_cand_local) else _cand_parent

if os.path.exists(data_path):
    files = os.listdir(data_path)
    npy_count = len([f for f in files if f.endswith('.npy')])
    print(f"   - Dataset Folder   : FOUND ({data_path})")
    print(f"   - NumPy Data Files : {npy_count} files detected")
else:
    print(f"   - Dataset Folder   : NOT FOUND (Checked: {data_path})")
    print("   - Please place Gathered_data_DEV in the project root.")

print("=" * 60)
if all_ok:
    print("ALL CHECKS PASSED: Environment is ready to run advanced_loader.py!")
else:
    print("WARNING: Some dependencies are missing. Run: pip install -r requirements.txt")
print("=" * 60)
