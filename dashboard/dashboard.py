# dashboard/dashboard.py
import webbrowser
import os

def open_dashboard():
    """
    Opens a static HTML dashboard (placeholder for now).
    """
    path = os.path.abspath("dashboard.html")
    if os.path.exists(path):
        webbrowser.open("file://" + path)
    else:
        print("Dashboard HTML not found. Generate it first.")
