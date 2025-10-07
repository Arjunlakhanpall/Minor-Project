"""
Enhanced Object Classification System
Main application for real-time object detection and classification
"""

import cv2
import sys
import os
from ultralytics import YOLO
from config import *
from utils.camera_utils import initialize_webcam, initialize_realsense, get_frame_webcam, get_frame_realsense, test_camera
from utils.detection_utils import draw_detections, get_detection_summary, draw_summary_text

def load_model(model_path: str):
    """
    Load YOLO model with error handling
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Loaded YOLO model or None if failed
    """
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            print("Please ensure you have a YOLO model file (e.g., model.pt)")
            return None
        
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
        print(f"Available classes: {model.names}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def initialize_camera():
    """
    Initialize camera (webcam or RealSense)
    
    Returns:
        Tuple of (camera_object, get_frame_function)
    """
    # Try RealSense first if available
    try:
        realsense_pipeline = initialize_realsense()
        if realsense_pipeline:
            print("RealSense camera initialized successfully")
            return realsense_pipeline, get_frame_realsense
    except Exception as e:
        print(f"RealSense camera not available: {e}")
    
    # Fallback to webcam
    print("Using webcam...")
    if not test_camera(CAMERA_INDEX):
        print(f"Camera {CAMERA_INDEX} not available. Trying camera 1...")
        if not test_camera(1):
            print("No cameras available!")
            return None, None
    
    webcam = initialize_webcam(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    return webcam, get_frame_webcam

def main():
    """
    Main application loop
    """
    print("=== Object Classification System ===")
    
    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    # Initialize camera
    camera, get_frame_func = initialize_camera()
    if camera is None:
        print("Failed to initialize camera. Exiting...")
        return
    
    print("Starting detection... Press 'q' to quit, 's' to save frame")
    
    try:
        while True:
            # Get frame
            depth_frame, color_frame = get_frame_func(camera)
            
            if color_frame is None:
                print("Failed to read frame")
                continue
            
            # Run detection
            if TRACK_OBJECTS:
                results = model.track(color_frame, classes=DETECT_CLASSES, 
                                   conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE)
            else:
                results = model(color_frame, classes=DETECT_CLASSES, 
                              conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE)
            
            # Get detection summary
            summary = get_detection_summary(results)
            
            # Draw detections and summary
            annotated_frame = draw_detections(color_frame, results, 
                                           show_confidence=SHOW_CONFIDENCE,
                                           show_class_names=SHOW_CLASS_NAMES)
            annotated_frame = draw_summary_text(annotated_frame, summary, 
                                             TEXT_POSITION, FONT_SCALE, 
                                             FONT_THICKNESS, TEXT_COLOR)
            
            # Display frame
            cv2.imshow(WINDOW_NAME, annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"detection_frame_{cv2.getTickCount()}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        if hasattr(camera, 'release'):
            camera.release()
        elif hasattr(camera, 'stop'):
            camera.stop()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()
