import os
import sys

# Add the project root to sys.path so we can import backend and modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app

# Vercel looks for `app` as the ASGI handler
