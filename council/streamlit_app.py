"""
Streamlit entry point for AI Council v2.

This file is the main entry point that Streamlit executes.
It properly configures the Python path and then runs the council app.
"""

import sys
from pathlib import Path

# Add project root to Python path BEFORE any other imports
# This allows council to be imported as a package
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now run the actual app
if __name__ == "__main__":
    # Import from council package using absolute imports
    from council.app import main
    main()