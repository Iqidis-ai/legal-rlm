#!/usr/bin/env python3
"""
Quick Start Script - Test the Knowledge Graph System

This script helps you verify that everything is set up correctly.
"""
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def check_setup():
    """Check if the system is properly set up."""
    print("=" * 60)
    print("Knowledge Graph System - Setup Check")
    print("=" * 60)
    
    errors = []
    warnings = []
    
    # Check Python version
    print("\n[1] Checking Python version...")
    if sys.version_info < (3, 10):
        errors.append(f"Python 3.10+ required. Found: {sys.version}")
        print(f"   ❌ Python version too old: {sys.version}")
    else:
        print(f"   ✅ Python {sys.version.split()[0]}")
    
    # Check required packages
    print("\n[2] Checking required packages...")
    required_packages = [
        'google.generativeai',
        'fitz',  # PyMuPDF
        'docx',  # python-docx
        'faiss',
        'numpy',
        'flask',
        'flask_cors',
        'dotenv',
        'tiktoken'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            errors.append(f"Missing package: {package}")
            print(f"   ❌ {package} - Run: pip install {package}")
    
    # Check for .env file
    print("\n[3] Checking environment configuration...")
    env_file = Path(".env")
    if env_file.exists():
        print("   ✅ .env file found")
        
        # Check if API key is set
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print(f"   ✅ GEMINI_API_KEY is set ({api_key[:10]}...)")
        else:
            warnings.append("GEMINI_API_KEY not found in .env file")
            print("   ⚠️  GEMINI_API_KEY not found in .env")
    else:
        warnings.append(".env file not found")
        print("   ⚠️  .env file not found - create it with GEMINI_API_KEY=your-key")
    
    # Check if we can import the system
    print("\n[4] Checking system imports...")
    try:
        from src.core import KnowledgeGraph
        print("   ✅ KnowledgeGraph can be imported")
    except Exception as e:
        errors.append(f"Cannot import KnowledgeGraph: {e}")
        print(f"   ❌ Import error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ Found {len(errors)} error(s):")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease fix these errors before proceeding.")
        return False
    else:
        print("\n✅ All checks passed!")
        
        if warnings:
            print(f"\n⚠️  {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"   - {warning}")
            print("\nYou can proceed, but some features may not work.")
        
        return True

def test_basic_functionality():
    """Test basic functionality with a simple example."""
    print("\n" + "=" * 60)
    print("Testing Basic Functionality")
    print("=" * 60)
    
    try:
        from src.core import KnowledgeGraph
        from src.core.config import GEMINI_API_KEY
        
        if not GEMINI_API_KEY:
            print("\n⚠️  Cannot test - GEMINI_API_KEY not set")
            print("   Create a .env file with: GEMINI_API_KEY=your-key")
            return False
        
        print("\n[1] Initializing Knowledge Graph...")
        kg = KnowledgeGraph("test_setup", api_key=GEMINI_API_KEY)
        print("   ✅ Knowledge Graph initialized")
        
        print("\n[2] Checking database...")
        stats = kg.get_stats()
        print(f"   ✅ Database accessible")
        print(f"      Entities: {stats.get('entities', 0)}")
        print(f"      Edges: {stats.get('edges', 0)}")
        print(f"      Documents: {stats.get('documents', 0)}")
        
        print("\n[3] Testing query engine...")
        if stats.get('entities', 0) > 0:
            result = kg.query("What entities are in this graph?")
            print(f"   ✅ Query engine working")
            print(f"      Answer: {result.answer[:100]}...")
        else:
            print("   ⚠️  No entities in graph (this is OK for a new setup)")
            print("      Process some documents to test queries")
        
        kg.close()
        print("\n✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function."""
    print("\n" + "=" * 60)
    print("Knowledge Graph System - Quick Start Check")
    print("=" * 60)
    
    # Run setup check
    setup_ok = check_setup()
    
    if not setup_ok:
        print("\n" + "=" * 60)
        print("Setup incomplete. Please fix the errors above.")
        print("=" * 60)
        return 1
    
    # Ask if user wants to test functionality
    print("\n" + "=" * 60)
    response = input("Run functionality test? (requires API key) [y/N]: ").strip().lower()
    
    if response == 'y':
        test_ok = test_basic_functionality()
        if test_ok:
            print("\n" + "=" * 60)
            print("✅ System is ready to use!")
            print("=" * 60)
            print("\nNext steps:")
            print("1. Add documents to process:")
            print("   python -m src.cli.extract --matter my_case --dir ./documents")
            print("\n2. Query the graph:")
            print("   python -m src.cli.query --matter my_case \"Who are the parties?\"")
            print("\n3. Start visualization server:")
            print("   python visualization_server.py --matter my_case")
            return 0
        else:
            print("\n⚠️  Functionality test had issues, but setup looks OK")
            return 0
    else:
        print("\n" + "=" * 60)
        print("✅ Setup check complete!")
        print("=" * 60)
        print("\nTo test functionality, run:")
        print("  python quick_start.py")
        print("\nOr start using the system:")
        print("  python -m src.cli.extract --matter my_case --dir ./documents")
        return 0

if __name__ == "__main__":
    sys.exit(main())


