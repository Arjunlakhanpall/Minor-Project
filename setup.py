#!/usr/bin/env python3
"""
Setup script for Object Classification System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False

def check_model_file():
    """Check if model file exists"""
    model_files = ["model.pt", "yolo.pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✓ Found model file: {model_file}")
            return True
    
    print("⚠ No model file found. Please download a YOLO model:")
    print("   - Download from: https://github.com/ultralytics/ultralytics")
    print("   - Or use: yolo.pt (default YOLOv8 model)")
    print("   - Place the .pt file in the project directory")
    return False

def test_camera():
    """Test camera availability"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("✓ Camera test passed")
                return True
        print("⚠ Camera test failed - check camera connection")
        return False
    except ImportError:
        print("⚠ OpenCV not installed - run pip install -r requirements.txt first")
        return False

def main():
    """Main setup function"""
    print("=== Object Classification System Setup ===\n")
    
    # Install requirements
    if not install_requirements():
        print("Setup failed at requirements installation")
        return False
    
    print()
    
    # Check model file
    model_ok = check_model_file()
    
    print()
    
    # Test camera
    camera_ok = test_camera()
    
    print()
    
    if model_ok and camera_ok:
        print("✓ Setup completed successfully!")
        print("Run: python main.py")
    else:
        print("⚠ Setup completed with warnings")
        print("Please address the issues above before running the application")
    
    return True

if __name__ == "__main__":
    main()







