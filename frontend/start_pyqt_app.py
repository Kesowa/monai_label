#!/usr/bin/env python3
"""
Simple PyQt5 Tree Detection Application Launcher
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the PyQt5 application"""
    print("🌟 PyQt5 Tree Detection Application")
    print("=" * 40)

    # Add current directory to Python path
    current_dir = str(Path(__file__).parent)
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    # Set environment variables for Qt
    os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
    os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '1'

    try:
        # Import and run the main application
        from pyqt_app import main
        main()

    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all required packages are installed in your venv")
        return False

    except Exception as e:
        print(f"❌ Application Error: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if not success:
        input("\nPress Enter to exit...")
        sys.exit(1)