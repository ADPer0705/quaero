"""
Utilities for the quaero core
"""

import sys
import psutil
import subprocess

import importlib.util
from importlib.metadata import version, PackageNotFoundError

import tomllib
import tomli_w

from typing import Optional
from pathlib import Path

# ==================================================
# Reading and Writing configs 
# ==================================================

def read_config(config_path: Path) -> Optional[dict]:
    if not config_path.exists():
        return None
    with open(config_path, "rb") as f:
        return tomllib.load(f)

def write_config(config_path: Path, config: dict) -> None:
    with open(config_path, "wb") as f:
        tomli_w.dump(config, f)

# ==================================================
# runtime dependency management for intelligence SDK
# ==================================================

def is_library_installed(lib_name: str) -> bool:
    """Checks if a library is available in the current environment."""
    return importlib.util.find_spec(lib_name) is not None

def bootstrap_pip_if_missing():
    """Ensures pip is available in the running environment using standard library ensurepip."""
    try:
        importlib.util.find_spec("pip")
    except ImportError:
        # pip is missing entirely. Use the standard library to inject it.
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--default-pip", "--quiet"],
            check=True,
            capture_output=True
        )

def install_provider_sdk(provider_name: str, library_package_name: str) -> bool:
    if is_library_installed(library_package_name):
        return True

    # Guard against environments like uv that skip pip seeding
    bootstrap_pip_if_missing()

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", library_package_name],
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError:
        return False
    

# ==================================================
# Miscallaneous utilities
# ==================================================

def get_system_memory_gb() -> float:
    """Returns the total system memory in gigabytes."""
    try:
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # psutil is not installed, return a default or raise an error
        return 0.0
    

def get_app_version() -> str:
    """Dynamically retrieves the application version from package metadata."""
    try:
        return version("quaero")
    except PackageNotFoundError:
        # Fallback if the package isn't installed in the environment yet (e.g., during raw local testing)
        return "0.1.0-dev"