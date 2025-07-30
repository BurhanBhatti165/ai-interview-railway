#!/usr/bin/env python3
"""
Startup check script for Railway deployment
This script checks if the FastAPI app can start successfully
"""

import os
import sys
import time
import requests
from urllib.parse import urljoin

def check_app_health(base_url="http://localhost:8000", max_retries=30, retry_interval=2):
    """Check if the FastAPI app is healthy"""
    print(f"Checking app health at {base_url}")
    
    for attempt in range(max_retries):
        try:
            # Try to connect to the health endpoint
            response = requests.get(urljoin(base_url, "/health"), timeout=10)
            if response.status_code == 200:
                print(f"✅ App is healthy! Response: {response.json()}")
                return True
            else:
                print(f"❌ App responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⏳ Attempt {attempt + 1}/{max_retries}: App not ready yet ({e})")
        
        if attempt < max_retries - 1:
            time.sleep(retry_interval)
    
    print("❌ App failed to start within the expected time")
    return False

if __name__ == "__main__":
    # Get port from environment or use default
    port = os.environ.get("PORT", "8000")
    base_url = f"http://localhost:{port}"
    
    print("🚀 Starting health check for AI Interview API...")
    success = check_app_health(base_url)
    
    if success:
        print("✅ Startup check passed!")
        sys.exit(0)
    else:
        print("❌ Startup check failed!")
        sys.exit(1) 