#!/usr/bin/env python3
"""
Debug script for Railway deployment
This script helps identify common issues
"""

import os
import sys
import importlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if required environment variables are set"""
    print("🔍 Checking environment variables...")
    
    required_vars = ["GOOGLE_API_KEY"]
    optional_vars = ["ELEVEN_API_KEY", "PORT"]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
        else:
            print(f"✅ {var}: {'*' * (len(os.getenv(var)) - 4)}{os.getenv(var)[-4:]}")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var}: {os.getenv(var)}")
        else:
            print(f"⚠️  {var}: Not set (optional)")
    
    if missing_required:
        print(f"❌ Missing required environment variables: {missing_required}")
        return False
    
    return True

def check_dependencies():
    """Check if all required packages can be imported"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "sqlalchemy",
        "langchain",
        "langchain_google_genai",
        "sentence_transformers",
        "faiss",
        "whisper",
        "numpy",
        "requests"
    ]
    
    failed_imports = []
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"❌ Failed to import: {failed_imports}")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    print("\n🔍 Checking files...")
    
    required_files = [
        "main.py",
        "ai_interview_agent.py",
        "database.py",
        "models.py",
        "schemas.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}: Not found")
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def check_directories():
    """Check if required directories exist"""
    print("\n🔍 Checking directories...")
    
    required_dirs = ["resumes"]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
        else:
            print(f"⚠️  {dir_name}/: Not found (will be created)")
            try:
                os.makedirs(dir_name, exist_ok=True)
                print(f"✅ Created {dir_name}/")
            except Exception as e:
                print(f"❌ Failed to create {dir_name}/: {e}")
                missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    return True

def main():
    """Run all checks"""
    print("🚀 Railway Deployment Debug Script")
    print("=" * 40)
    
    checks = [
        ("Environment Variables", check_environment),
        ("Dependencies", check_dependencies),
        ("Files", check_files),
        ("Directories", check_directories)
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n📋 {name}")
        print("-" * 20)
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 40)
    if all_passed:
        print("✅ All checks passed! Your app should deploy successfully.")
        sys.exit(0)
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 