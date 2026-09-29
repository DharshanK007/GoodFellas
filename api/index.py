import os
import sys
from pathlib import Path

# Add the project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the FastAPI application
from skyguard.dashboard.app import app
