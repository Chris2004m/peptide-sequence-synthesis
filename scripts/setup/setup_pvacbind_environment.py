#!/usr/bin/env python3
"""
Setup script for pVACbind environment and dependencies
"""

import subprocess
import sys
from pathlib import Path

def install_pvactools():
    """Install pVACtools suite including pVACbind"""
    print("Installing pVACtools (includes pVACbind)...")
    
    commands = [
        # Install pVACtools via pip
        ["pip", "install", "pvactools"],
        
        # Install IEDB tools (required for pVACbind)
        ["pvactools", "install_vep_plugins"],
    ]
    
    for cmd in commands:
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✓ Success")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return False
    
    return True

def setup_iedb():
    """Download and setup IEDB tools"""
    print("\nSetting up IEDB tools...")
    print("Note: You may need to manually download IEDB tools from:")
    print("https://www.iedb.org/downloader")
    print("Follow pVACtools documentation for IEDB setup.")

def test_environment():
    """Test if pVACbind is properly installed"""
    try:
        result = subprocess.run(["pvacbind", "--help"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ pVACbind is properly installed")
            return True
        else:
            print("✗ pVACbind installation issue")
            return False
    except FileNotFoundError:
        print("✗ pVACbind not found")
        return False

def main():
    print("=== pVACbind Environment Setup ===")
    
    if test_environment():
        print("Environment already set up!")
        return
    
    print("\n1. Installing pVACtools...")
    success = install_pvactools()
    
    if success:
        print("\n2. Setting up IEDB...")
        setup_iedb()
        
        print("\n3. Testing installation...")
        if test_environment():
            print("\n✓ Setup complete!")
        else:
            print("\n✗ Setup incomplete. See pVACtools documentation.")
    else:
        print("\n✗ Installation failed. Please install manually:")
        print("pip install pvactools")

if __name__ == "__main__":
    main()
