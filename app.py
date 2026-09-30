"""
Root entry-point proxy for Streamlit Dashboard.
Allows running either:
    streamlit run dashboard/app.py
or
    streamlit run app.py
"""

import os
import sys

# Ensure root is in path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Execute the main dashboard app
import runpy
runpy.run_path(os.path.join(ROOT_DIR, "dashboard", "app.py"), run_name="__main__")
