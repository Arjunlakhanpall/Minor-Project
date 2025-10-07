#!/usr/bin/env python3
"""
Comprehensive Object Detection System
Detects and identifies ALL objects in camera feed without saving images
"""

import cv2
import sys
import os
import json
import time
from datetime import datetime
from ultralytics import YOLO
from config import *
from utils.camera_utils import initialize_webcam, initialize_realsense, get_frame_webcam, get_frame_realsense, test_camera
from utils.detection_utils import draw_detections, get_detection_summary, draw_summary_text

class ComprehensiveDetector:
    def __init__(self):
        self.detection_log = []
        self.frame_count = 0
        self.start_time = time.time()
        self.detected_objects = set()  # Track unique objects detected
        self.object_counts = {}  # Count occurrences of each object
        
    def log_detection(self, frame, results, summary):
        """Log detection information without saving images"""
        timestamp = datetime.now().isoformat()
        frame_data = {
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'total_objects': summary['total_objects'],
            'average_confidence': summary['average_confidence'],
            'detections': []
        }
        
        if results and len(results) > 0 and results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = results[0].names[class_id] if hasattr(results[0], 'names') else f"class_{class_id}"
                
                # Track unique objects
                self.detected_objects.add(class_name)
                self.object_counts[class_name] = self.object_counts.get(class_name, 0) + 1
                
                detection = {
                    'object_id': i,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence,
                    'bounding_box': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    },
                    'center': {
                        'x': float((x1 + x2) / 2),
                        'y': float((y1 + y2) / 2)
                    }
                }
                frame_data['detections'].append(detection)
        
        self.detection_log.append(frame_data)
        self.frame_count += 1
        
        # Print real-time detection info
        if summary['total_objects'] > 0:
            objects = [det['class_name'] for det in frame_data['detections']]
            print(f"🔍 Frame {self.frame_count}: {', '.join(objects)}")
        
        return frame_data

    def print_detection_summary(self):
        """Print comprehensive detection summary"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DETECTION SUMMARY")
        print("="*60)
        
        total_detections = sum(len(frame['detections']) for frame in self.detection_log)
        unique_objects = len(self.detected_objects)
        
        print(f"📈 Session Duration: {time.time() - self.start_time:.1f} seconds")
        print(f"📈 Total Frames Processed: {self.frame_count}")
        print(f"📈 Total Objects Detected: {total_detections}")
        print(f"📈 Unique Object Types: {unique_objects}")
        
        print(f"\n🎯 ALL OBJECTS DETECTED:")
        print("-" * 40)
        for obj, count in sorted(self.object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {obj}: {count} detections")
        
        print(f"\n🔍 DETECTED OBJECT TYPES:")
        print("-" * 40)
        for obj in sorted(self.detected_objects):
            print(f"  ✓ {obj}")
        
        return {
            'total_detections': total_detections,
            'unique_objects': unique_objects,
            'object_counts': self.object_counts,
            'detected_objects': list(self.detected_objects)
        }

def load_model(model_path: str):
    """Load YOLO model with error handling"""
    try:
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
        
        model = YOLO(model_path)
        print(f"Model loaded successfully: {model_path}")
        print(f"Available classes: {model.names}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def initialize_camera():
    """Initialize camera (webcam or RealSense)"""
    try:
        realsense_pipeline = initialize_realsense()
        if realsense_pipeline:
            print("RealSense camera initialized successfully")
            return realsense_pipeline, get_frame_realsense
    except Exception as e:
        print(f"RealSense camera not available: {e}")
    
    print("Using webcam...")
    if not test_camera(CAMERA_INDEX):
        print(f"Camera {CAMERA_INDEX} not available. Trying camera 1...")
        if not test_camera(1):
            print("No cameras available!")
            return None, None
    
    webcam = initialize_webcam(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    return webcam, get_frame_webcam

def main():
    """Main application with comprehensive object detection"""
    print("="*60)
    print("🔍 COMPREHENSIVE OBJECT DETECTION SYSTEM")
    print("="*60)
    print("📋 Detecting ALL objects: glasses, purse, wall, fan, etc.")
    print("📋 NO images will be saved - detection only")
    print("📋 Press 'q' to quit and see summary")
    print("="*60)
    
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
    
    # Initialize detector
    detector = ComprehensiveDetector()
    
    print("\n🎥 Starting comprehensive detection...")
    print("Point camera at different objects to see them detected!")
    
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
            
            # Log detection data (no image saving)
            frame_data = detector.log_detection(color_frame, results, summary)
            
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
                # Print current summary
                detector.print_detection_summary()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        if hasattr(camera, 'release'):
            camera.release()
        elif hasattr(camera, 'stop'):
            camera.stop()
        cv2.destroyAllWindows()
        
        # Print final comprehensive summary
        print("\n" + "="*60)
        print("📊 FINAL DETECTION SUMMARY")
        print("="*60)
        summary = detector.print_detection_summary()
        
        # Save summary to file
        with open('detection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Summary saved to: detection_summary.json")
        
        print("Application closed")

if __name__ == "__main__":
    main()




