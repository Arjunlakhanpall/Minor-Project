#!/usr/bin/env python3
"""
Enhanced Object Detection System
Detects common objects like pen, glasses, fan, bag, etc.
Uses multiple detection strategies for comprehensive object identification
"""

import cv2
import sys
import os
import json
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from config import *
from utils.camera_utils import initialize_webcam, initialize_realsense, get_frame_webcam, get_frame_realsense, test_camera
from utils.detection_utils import draw_detections, get_detection_summary, draw_summary_text

class EnhancedDetector:
    def __init__(self):
        self.detection_log = []
        self.frame_count = 0
        self.start_time = time.time()
        self.detected_objects = set()
        self.object_counts = {}
        self.custom_objects = {
            'pen': ['pen', 'pencil', 'marker', 'writing instrument'],
            'glasses': ['glasses', 'sunglasses', 'spectacles', 'eyeglasses'],
            'fan': ['fan', 'ceiling fan', 'desk fan', 'electric fan'],
            'bag': ['bag', 'handbag', 'backpack', 'purse', 'tote bag', 'shopping bag'],
            'book': ['book', 'notebook', 'magazine', 'journal'],
            'phone': ['cell phone', 'mobile phone', 'smartphone'],
            'laptop': ['laptop', 'computer', 'notebook computer'],
            'chair': ['chair', 'seat', 'stool'],
            'table': ['dining table', 'desk', 'table'],
            'bottle': ['bottle', 'water bottle', 'drink bottle'],
            'cup': ['cup', 'mug', 'glass', 'drinking vessel'],
            'keyboard': ['keyboard', 'computer keyboard'],
            'mouse': ['mouse', 'computer mouse'],
            'clock': ['clock', 'watch', 'timepiece'],
            'lamp': ['lamp', 'light', 'desk lamp'],
            'wall': ['wall', 'surface', 'background']
        }
        
    def detect_custom_objects(self, frame):
        """Detect custom objects using color and shape analysis"""
        detected_custom = []
        
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangular objects (books, phones, laptops)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check for rectangular shapes (books, phones, etc.)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 0.5 < aspect_ratio < 2.0:  # Reasonable aspect ratio
                        # Classify based on size and position
                        if area > 5000 and aspect_ratio > 1.2:
                            detected_custom.append(('book', 0.6, (x, y, x+w, y+h)))
                        elif area > 2000 and 0.8 < aspect_ratio < 1.2:
                            detected_custom.append(('phone', 0.5, (x, y, x+w, y+h)))
        
        # Detect circular objects (cups, bottles)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if r > 20:  # Filter small circles
                    detected_custom.append(('cup', 0.5, (x-r, y-r, x+r, y+r)))
        
        # Detect dark objects (glasses, pens)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))
        dark_contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in dark_contours:
            area = cv2.contourArea(contour)
            if 100 < area < 2000:  # Pen/glasses size range
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if aspect_ratio > 3:  # Long thin objects (pens)
                    detected_custom.append(('pen', 0.4, (x, y, x+w, y+h)))
                elif 0.5 < aspect_ratio < 2.0:  # Square-ish objects (glasses)
                    detected_custom.append(('glasses', 0.4, (x, y, x+w, y+h)))
        
        return detected_custom

    def log_detection(self, frame, results, summary):
        """Enhanced detection logging with custom objects"""
        timestamp = datetime.now().isoformat()
        frame_data = {
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'total_objects': summary['total_objects'],
            'average_confidence': summary['average_confidence'],
            'detections': [],
            'custom_detections': []
        }
        
        # Process YOLO detections
        if results and len(results) > 0 and results[0].boxes:
            for i, box in enumerate(results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = results[0].names[class_id] if hasattr(results[0], 'names') else f"class_{class_id}"
                
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
        
        # Process custom object detections
        custom_detections = self.detect_custom_objects(frame)
        for obj_name, confidence, bbox in custom_detections:
            self.detected_objects.add(obj_name)
            self.object_counts[obj_name] = self.object_counts.get(obj_name, 0) + 1
            
            custom_detection = {
                'object_name': obj_name,
                'confidence': confidence,
                'bounding_box': {
                    'x1': float(bbox[0]),
                    'y1': float(bbox[1]),
                    'x2': float(bbox[2]),
                    'y2': float(bbox[3])
                },
                'center': {
                    'x': float((bbox[0] + bbox[2]) / 2),
                    'y': float((bbox[1] + bbox[3]) / 2)
                }
            }
            frame_data['custom_detections'].append(custom_detection)
        
        self.detection_log.append(frame_data)
        self.frame_count += 1
        
        # Print real-time detection info
        all_objects = [det['class_name'] for det in frame_data['detections']]
        all_objects.extend([det['object_name'] for det in frame_data['custom_detections']])
        
        if all_objects:
            print(f"🔍 Frame {self.frame_count}: {', '.join(all_objects)}")
        
        return frame_data

    def print_detection_summary(self):
        """Print comprehensive detection summary"""
        print("\n" + "="*60)
        print("📊 ENHANCED DETECTION SUMMARY")
        print("="*60)
        
        total_detections = sum(len(frame['detections']) + len(frame['custom_detections']) for frame in self.detection_log)
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
    """Main application with enhanced object detection"""
    print("="*60)
    print("🔍 ENHANCED OBJECT DETECTION SYSTEM")
    print("="*60)
    print("📋 Detecting: pen, glasses, fan, bag, book, phone, laptop, etc.")
    print("📋 Using YOLO + Custom Detection for comprehensive coverage")
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
    
    # Initialize enhanced detector
    detector = EnhancedDetector()
    
    print("\n🎥 Starting enhanced detection...")
    print("Point camera at different objects: pen, glasses, fan, bag, etc.")
    
    try:
        while True:
            # Get frame
            depth_frame, color_frame = get_frame_func(camera)
            
            if color_frame is None:
                print("Failed to read frame")
                continue
            
            # Run YOLO detection
            if TRACK_OBJECTS:
                results = model.track(color_frame, classes=DETECT_CLASSES, 
                                   conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE)
            else:
                results = model(color_frame, classes=DETECT_CLASSES, 
                              conf=CONFIDENCE_THRESHOLD, imgsz=IMAGE_SIZE)
            
            # Get detection summary
            summary = get_detection_summary(results)
            
            # Log enhanced detection data
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
        
        # Print final enhanced summary
        print("\n" + "="*60)
        print("📊 FINAL ENHANCED DETECTION SUMMARY")
        print("="*60)
        summary = detector.print_detection_summary()
        
        # Save summary to file
        with open('enhanced_detection_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 Summary saved to: enhanced_detection_summary.json")
        
        print("Application closed")

if __name__ == "__main__":
    main()



