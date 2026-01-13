"""
Streamlit entry point for AI Council v2.

This file configures the Python path and then imports the actual app.
This solves the "ModuleNotFoundError: No module named 'council'" error.
"""

import sys
from pathlib import Path

# Add project root to Python path BEFORE any other imports
# This allows "from council.xxx import" to work
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the app from the council package
from council.app import main

if __name__ == "__main__":
    main()
