#!/usr/bin/env python3
"""
Enterprise Knowledge Assistant - Quick Launcher

This script provides an easy way to start different components of the
Enterprise Knowledge Assistant system.

Usage:
    python run_enterprise_assistant.py [component]

Components:
    - demo: Run comprehensive demo
    - streamlit: Start Streamlit dashboard
    - api: Start FastAPI backend
    - interactive: Interactive command-line interface
    - all: Start all services
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def print_banner():
    """Print the application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║          🏢 Enterprise Knowledge Assistant with RAG                          ║
║                                                                              ║
║     Powered by LangChain • OpenAI GPT-4 • FAISS • Streamlit • FastAPI      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies():
    """Check if required packages are installed"""
    # Map package names to their actual import names
    package_imports = {
        'streamlit': 'streamlit',
        'fastapi': 'fastapi', 
        'uvicorn': 'uvicorn',
        'langchain': 'langchain',
        'openai': 'openai',
        'sentence-transformers': 'sentence_transformers',
        'faiss-cpu': 'faiss',  # faiss-cpu installs as 'faiss' module
        'pandas': 'pandas'
    }
    
    missing_packages = []
    
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"💡 Install with: pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_api_key():
    """Check if OpenAI API key is configured"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OpenAI API key not found!")
        print("   Set OPENAI_API_KEY environment variable for full functionality")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'")
        print("   Note: System will work with limited functionality without API key")
        return False
    else:
        print("✅ OpenAI API key configured")
        return True

def run_component(component: str):
    """Run the specified component"""
    
    if component == "demo":
        print("🎬 Running comprehensive demo...")
        subprocess.run([sys.executable, "enterprise_assistant_demo.py", "--mode", "demo"])
    
    elif component == "streamlit":
        print("🌐 Starting Streamlit dashboard...")
        print("📊 Dashboard will be available at: http://localhost:8501")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "streamlit_dashboard.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ])
    
    elif component == "api":
        print("🚀 Starting FastAPI backend...")
        print("📖 API documentation: http://localhost:8000/docs")
        print("📚 ReDoc documentation: http://localhost:8000/redoc")
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "fastapi_backend:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    
    elif component == "interactive":
        print("💬 Starting interactive mode...")
        subprocess.run([sys.executable, "enterprise_assistant_demo.py", "--mode", "interactive"])
    
    elif component == "all":
        print("🚀 Starting all services...")
        subprocess.run([sys.executable, "enterprise_assistant_demo.py", "--mode", "all"])
    
    else:
        print(f"❌ Unknown component: {component}")
        print_help()

def print_help():
    """Print help information"""
    help_text = """
Available Components:

📊 demo        - Run comprehensive demo showcasing all features
🌐 streamlit   - Start Streamlit web dashboard (port 8501)
🚀 api         - Start FastAPI backend server (port 8000)  
💬 interactive - Interactive command-line interface
🔄 all         - Start all services (Streamlit + FastAPI)

Examples:
    python run_enterprise_assistant.py demo
    python run_enterprise_assistant.py streamlit
    python run_enterprise_assistant.py api

Quick Start:
    1. Set your OpenAI API key: export OPENAI_API_KEY='your-key'
    2. Run demo: python run_enterprise_assistant.py demo
    3. Try web interface: python run_enterprise_assistant.py streamlit

Documentation:
    - Streamlit Dashboard: http://localhost:8501
    - FastAPI Docs: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""
    print(help_text)

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Enterprise Knowledge Assistant Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'component',
        nargs='?',
        choices=['demo', 'streamlit', 'api', 'interactive', 'all'],
        help='Component to run'
    )
    
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check dependencies if requested
    if args.check_deps:
        print("🔍 Checking dependencies...")
        deps_ok = check_dependencies()
        api_key_ok = check_api_key()
        
        if deps_ok and api_key_ok:
            print("✅ All dependencies satisfied!")
        elif deps_ok:
            print("⚠️  Dependencies OK, but API key missing (limited functionality)")
        else:
            print("❌ Please install missing dependencies")
        
        return
    
    # Check dependencies
    print("🔍 Checking system requirements...")
    if not check_dependencies():
        print("\n💡 Install requirements with: pip install -r requirements.txt")
        return
    
    # Check API key
    check_api_key()
    print()
    
    # Run component or show help
    if args.component:
        run_component(args.component)
    else:
        print("🤔 No component specified. Here's what you can run:")
        print_help()

if __name__ == "__main__":
    main() 