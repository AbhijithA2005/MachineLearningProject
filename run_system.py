import subprocess
import os
import sys
import time
import signal

def run():
    # Detect Python path with pandas/sklearn
    python_path = "/Users/abhi/miniconda3/bin/python"
    
    print("Starting ML Paradigm Decision Support System...")
    
    # 1. Start Backend
    print("Launching Backend (FastAPI)...")
    backend_proc = subprocess.Popen(
        [python_path, "main.py"],
        cwd="backend"
    )
    
    # 2. Start Frontend
    print("Launching Frontend (Vite)...")
    frontend_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd="frontend"
    )
    
    print("\nSystem running!")
    print("Backend: http://localhost:8000")
    print("Frontend: http://localhost:3000")
    print("\nPress Ctrl+C to stop all services.")
    
    try:
        while True:
            time.sleep(1)
            if backend_proc.poll() is not None:
                print("Backend service stopped unexpectedly.")
                break
            if frontend_proc.poll() is not None:
                print("Frontend service stopped unexpectedly.")
                break
    except KeyboardInterrupt:
        print("\nStopping services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    run()
