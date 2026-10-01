import importlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "xgboost": "xgboost",
    "utm": "utm",
    "torch": "torch",
}


def get_distribution_version(dist_name):
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def check_module(module_name, dist_name):
    try:
        module = importlib.import_module(module_name)
        module_version = getattr(module, "__version__", get_distribution_version(dist_name))
        status = "OK"
    except Exception as exc:
        module_version = get_distribution_version(dist_name)
        status = f"IMPORT FAILED: {exc}"
    return module_version, status


print("-" * 60)
print(f"Python Executable : {sys.executable}")
print(f"Python Version    : {sys.version.split()[0]}")
print(f"Platform          : {platform.platform()}")
print("-" * 60)
print("Package Checks")

for module_name, dist_name in REQUIRED_MODULES.items():
    module_version, status = check_module(module_name, dist_name)
    print(f"{dist_name:<15} version={module_version:<20} status={status}")

print("-" * 60)
print("CUDA / GPU Checks")
print(f"PyTorch Version   : {torch.__version__}")
print(f"CUDA Available    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version      : {torch.version.cuda}")
    print(f"GPU Count         : {torch.cuda.device_count()}")
    for gpu_index in range(torch.cuda.device_count()):
        print(f"GPU {gpu_index} Name      : {torch.cuda.get_device_name(gpu_index)}")
else:
    print("WARNING: GPU NOT DETECTED. Training will run on CPU.")

print("-" * 60)