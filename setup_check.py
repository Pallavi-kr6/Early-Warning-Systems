#!/usr/bin/env python
"""
Quick compatibility check for TensorFlow installation
"""
import sys

def main():
    print(f"Python version: {sys.version}")
    print(f"Python version info: {sys.version_info}")
    
    if sys.version_info.minor >= 13:
        print("\nWARNING: Python 3.13+ detected")
        print("TensorFlow may not be available for this Python version.")
        print("\nOptions:")
        print("1. Install Python 3.11 or 3.12 from https://www.python.org/downloads/")
        print("2. Try installing TensorFlow nightly build:")
        print("   pip install tf-nightly")
        print("3. Use Docker with a compatible Python version")
        return
    
    print("\n✅ Python version should be compatible with TensorFlow")
    print("You can proceed with: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
