#!/usr/bin/env python
'''
Quick start script for Physics Course Recommender API
Run: python quick_start.py
'''

import os
import sys
import subprocess


def run_command(cmd):
    '''Run a shell command'''
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    print(result.stdout)
    return True


def main():
    print("🚀 Quick Start - Physics Course Recommender API\n")

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)

    # Install minimal requirements
    print("📦 Installing minimal requirements...")
    if not run_command("pip install django djangorestframework requests beautifulsoup4 lxml numpy scikit-learn"):
        print("❌ Failed to install requirements")
        print("Try: pip install -r requirements_minimal.txt")
        sys.exit(1)

    # Create Django project if it doesn't exist
    if not os.path.exists('manage.py'):
        print("🔨 Creating Django project...")
        run_command("django-admin startproject physics_course_api .")
        run_command("python manage.py startapp api")

    # Run migrations
    print("🗄️ Running migrations...")
    run_command("python manage.py makemigrations")
    run_command("python manage.py migrate")

    # Create superuser
    print("\n👤 Create superuser (optional - press Ctrl+C to skip)")
    try:
        run_command("python manage.py createsuperuser")
    except KeyboardInterrupt:
        print("\nSkipped superuser creation")

    print("\n✅ Setup complete!")
    print("\n🚀 Starting development server...")
    print("Access the API at: http://localhost:8000/api/")
    print("Health check: http://localhost:8000/api/health/")
    print("\nPress Ctrl+C to stop the server\n")

    # Start server
    run_command("python manage.py runserver")


if __name__ == "__main__":
    main()