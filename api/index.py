import os
import sys

# Ensure the project root is in the Python path so that
# `backend.main` can import `modules.*` correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app

# Vercel looks for `app` as the ASGI/WSGI handler
