#!/usr/bin/env python3
"""
Test camera access and permissions
"""

import cv2
import sys

def test_camera_access():
    """Test camera access with detailed error reporting"""
    print("=== Camera Access Test ===")
    
    # Try different camera indices
    for i in range(3):
        print(f"\nTrying camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if not cap.isOpened():
            print(f"  ❌ Camera {i}: Failed to open")
            continue
        
        print(f"  ✓ Camera {i}: Opened successfully")
        
        # Try to read a frame
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  ✓ Camera {i}: Frame captured successfully ({frame.shape})")
            cap.release()
            return i
        else:
            print(f"  ❌ Camera {i}: Failed to capture frame")
        
        cap.release()
    
    return None

def main():
    """Main test function"""
    print("Testing camera access...")
    print("\nIf you see permission dialogs, please click 'Allow' or 'OK'")
    print("This may take a few seconds...\n")
    
    working_camera = test_camera_access()
    
    if working_camera is not None:
        print(f"\n✅ SUCCESS: Camera {working_camera} is working!")
        print("You can now run the main application with:")
        print("python3 main.py")
    else:
        print("\n❌ FAILED: No cameras are accessible")
        print("\nTroubleshooting steps:")
        print("1. Go to System Preferences → Security & Privacy → Privacy → Camera")
        print("2. Make sure Terminal (or your Python app) has camera access")
        print("3. Try running the app again")
        print("4. If using VS Code or another IDE, you may need to grant it camera access too")

if __name__ == "__main__":
    main()







