#!/usr/bin/env python3
"""
Enhanced Object Classification System with Comprehensive Logging
Captures all detection data and saves it to files
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

class DetectionLogger:
    def __init__(self):
        self.detection_log = []
        self.frame_count = 0
        self.start_time = time.time()
        
    def log_detection(self, frame, results, summary):
        """Log detailed detection information"""
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
        
        # Save frame with detections
        if summary['total_objects'] > 0:
            filename = f"detection_frame_{self.frame_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved frame {self.frame_count}: {filename} - {summary['total_objects']} objects detected")
        
        return frame_data

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

def save_detection_report(logger):
    """Save comprehensive detection report"""
    report = {
        'session_info': {
            'start_time': datetime.fromtimestamp(logger.start_time).isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_frames': logger.frame_count,
            'duration_seconds': time.time() - logger.start_time
        },
        'summary': {
            'total_detections': sum(len(frame['detections']) for frame in logger.detection_log),
            'unique_objects': len(set(det['class_name'] for frame in logger.detection_log for det in frame['detections'])),
            'object_counts': {}
        },
        'detections': logger.detection_log
    }
    
    # Count object types
    for frame in logger.detection_log:
        for det in frame['detections']:
            class_name = det['class_name']
            report['summary']['object_counts'][class_name] = report['summary']['object_counts'].get(class_name, 0) + 1
    
    # Save JSON report
    with open('detection_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save summary text
    with open('detection_summary.txt', 'w') as f:
        f.write("=== OBJECT DETECTION SUMMARY ===\n\n")
        f.write(f"Session Duration: {report['session_info']['duration_seconds']:.1f} seconds\n")
        f.write(f"Total Frames Processed: {report['session_info']['total_frames']}\n")
        f.write(f"Total Objects Detected: {report['summary']['total_detections']}\n")
        f.write(f"Unique Object Types: {report['summary']['unique_objects']}\n\n")
        
        f.write("OBJECT COUNTS:\n")
        for obj, count in sorted(report['summary']['object_counts'].items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {obj}: {count} detections\n")
    
    print(f"\n📊 Detection Report Saved:")
    print(f"   - detection_report.json (detailed data)")
    print(f"   - detection_summary.txt (summary)")
    print(f"   - {logger.frame_count} saved frames")

def main():
    """Main application with comprehensive logging"""
    print("=== Enhanced Object Classification System ===")
    print("📊 All detection data will be logged and saved")
    print("📸 Frames with detections will be automatically saved")
    print("📝 Press 'q' to quit and generate report\n")
    
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
    
    # Initialize logger
    logger = DetectionLogger()
    
    print("🎥 Starting detection with full logging...")
    
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
            
            # Log detection data
            frame_data = logger.log_detection(color_frame, results, summary)
            
            # Print real-time detection info
            if summary['total_objects'] > 0:
                objects = [det['class_name'] for det in frame_data['detections']]
                print(f"🔍 Frame {logger.frame_count}: {', '.join(objects)}")
            
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
                # Manual save
                filename = f"manual_save_{cv2.getTickCount()}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"📸 Manual save: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup and save report
        if hasattr(camera, 'release'):
            camera.release()
        elif hasattr(camera, 'stop'):
            camera.stop()
        cv2.destroyAllWindows()
        
        # Generate comprehensive report
        save_detection_report(logger)
        print("Application closed")

if __name__ == "__main__":
    main()




