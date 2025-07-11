#!/usr/bin/env python3
"""
Setup script for KYC-based Loan Origination System
"""
import os
import sys
import subprocess
import shutil

def create_directories():
    """Create necessary directories"""
    directories = [
        'models/saved_models',
        'data/face_database',
        'static/css',
        'static/js', 
        'static/uploads',
        'templates',
        'utils'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_init_files():
    """Create __init__.py files for Python packages"""
    init_files = [
        'models/__init__.py',
        'utils/__init__.py'
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('# Package initialization file\n')
        print(f"✓ Created file: {init_file}")

def check_dataset():
    """Check if dataset exists"""
    if os.path.exists('data/Insurance_Enhanced.csv'):
        print("✓ Dataset found: data/Insurance_Enhanced.csv")
        return True
    else:
        print("✗ Dataset not found: data/Insurance_Enhanced.csv")
        print("  Please copy your Insurance_Enhanced.csv file to the data/ directory")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False

def add_sample_face_image():
    """Add a sample face image for testing"""
    face_db_path = 'data/face_database'
    sample_image_path = os.path.join(face_db_path, 'sample_user.jpg')
    
    if not os.path.exists(sample_image_path):
        print("Note: Add sample face images to data/face_database/ for testing")
        print("  Example: data/face_database/user_001.jpg")
        
        # Create a placeholder file
        with open(os.path.join(face_db_path, 'README.txt'), 'w') as f:
            f.write("Add face images here for KYC verification\n")
            f.write("Supported formats: .jpg, .jpeg, .png\n")
            f.write("Example files: user_001.jpg, user_002.jpg\n")
        
        print("✓ Created face database README")

def main():
    """Main setup function"""
    print("=" * 50)
    print("KYC-based Loan Origination System Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("Error: Please run this script from the project root directory")
        sys.exit(1)
    
    print("\n1. Creating directories...")
    create_directories()
    
    print("\n2. Creating Python package files...")
    create_init_files()
    
    print("\n3. Checking dataset...")
    dataset_exists = check_dataset()
    
    print("\n4. Installing dependencies...")
    deps_installed = install_dependencies()
    
    print("\n5. Setting up face database...")
    add_sample_face_image()
    
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    
    if dataset_exists and deps_installed:
        print("✓ Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add face images to data/face_database/ directory")
        print("2. Run: python models/train_models.py")
        print("3. Run: python app.py")
        print("4. Open browser: http://localhost:5000")
    else:
        print("✗ Setup incomplete. Please resolve the issues above.")
        
        if not dataset_exists:
            print("\n• Copy Insurance_Enhanced.csv to data/ directory")
        if not deps_installed:
            print("• Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()