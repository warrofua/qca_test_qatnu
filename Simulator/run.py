#!/usr/bin/env python3
"""
QATNU First Principles Simulator - Entry Point

This script:
1. Initializes the database
2. Starts the FastAPI backend (in background)
3. Launches the Textual frontend

Usage:
    python Simulator/run.py
    
Or separately:
    # Terminal 1: Backend
    cd Simulator && python -m backend.main
    
    # Terminal 2: Frontend
    cd Simulator && python -m frontend.app
"""

import sys
import os
import subprocess
import time
import signal
from pathlib import Path

# Add parent directory to path for importing qca_core
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def init_database():
    """Initialize SQLite database."""
    print("[INIT] Initializing database...")
    
    from backend.models import init_db
    engine = init_db()
    print(f"[OK] Database ready at: {Path(__file__).parent / 'qca.db'}")
    return engine


def start_backend():
    """Start FastAPI backend in subprocess."""
    print("[START] Starting backend server...")
    
    # Start uvicorn
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for server to start
    time.sleep(2)
    
    print("[OK] Backend running at: http://localhost:8000")
    print("   API docs: http://localhost:8000/docs")
    
    return process


def start_frontend():
    """Start Textual frontend."""
    print("[START] Starting frontend...")
    
    from frontend.app import QATNUSimulator
    
    app = QATNUSimulator(api_url="ws://localhost:8000")
    app.run()


def main():
    """Main entry point."""
    print("=" * 60)
    print("QATNU First Principles Simulator")
    print("=" * 60)
    print()
    
    # Initialize database
    init_database()
    print()
    
    # Start backend
    backend_process = start_backend()
    print()
    
    # Setup cleanup
    def cleanup(signum, frame):
        print("\n[STOP] Shutting down...")
        backend_process.terminate()
        backend_process.wait()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Run frontend (blocks until exit)
        start_frontend()
    finally:
        cleanup(None, None)


if __name__ == "__main__":
    main()
