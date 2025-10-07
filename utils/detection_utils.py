"""
Detection utility functions for object classification project
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO

def draw_detections(frame: np.ndarray, results, show_confidence: bool = True, 
                   show_class_names: bool = True, font_scale: float = 1, 
                   font_thickness: int = 2) -> np.ndarray:
    """
    Draw detection results on frame
    
    Args:
        frame: Input frame
        results: YOLO detection results
        show_confidence: Whether to show confidence scores
        show_class_names: Whether to show class names
        font_scale: Font scale for text
        font_thickness: Font thickness for text
    
    Returns:
        Frame with drawn detections
    """
    if not results or len(results) == 0 or not results[0].boxes:
        return frame
    
    annotated_frame = frame.copy()
    
    for box in results[0].boxes:
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = box.conf[0].cpu().numpy()
        class_id = int(box.cls[0].cpu().numpy())
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Prepare label text
        label_parts = []
        if show_class_names and hasattr(results[0], 'names'):
            class_name = results[0].names[class_id]
            label_parts.append(class_name)
        
        if show_confidence:
            label_parts.append(f"{confidence:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Get text size for background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw background rectangle for text
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), (0, 255, 0), -1)
            
            # Draw text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 
                       font_thickness, cv2.LINE_AA)
    
    return annotated_frame

def get_detection_summary(results) -> dict:
    """
    Get summary of detection results
    
    Args:
        results: YOLO detection results
    
    Returns:
        Dictionary with detection summary
    """
    if not results or len(results) == 0 or not results[0].boxes:
        return {"total_objects": 0, "class_counts": {}, "average_confidence": 0.0}
    
    boxes = results[0].boxes
    total_objects = len(boxes)
    class_counts = {}
    confidences = []
    
    for box in boxes:
        class_id = int(box.cls[0].cpu().numpy())
        confidence = float(box.conf[0].cpu().numpy())
        
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        confidences.append(confidence)
    
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "total_objects": total_objects,
        "class_counts": class_counts,
        "average_confidence": average_confidence,
        "confidences": confidences
    }

def draw_summary_text(frame: np.ndarray, summary: dict, position: Tuple[int, int] = (50, 50),
                    font_scale: float = 1, font_thickness: int = 2, 
                    text_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    Draw detection summary on frame
    
    Args:
        frame: Input frame
        summary: Detection summary dictionary
        position: Text position
        font_scale: Font scale
        font_thickness: Font thickness
        text_color: Text color in BGR format
    
    Returns:
        Frame with summary text
    """
    annotated_frame = frame.copy()
    
    # Draw total objects count
    total_text = f"Total Objects: {summary['total_objects']}"
    cv2.putText(annotated_frame, total_text, position, cv2.FONT_HERSHEY_SIMPLEX, 
               font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # Draw average confidence
    if summary['total_objects'] > 0:
        conf_text = f"Avg Confidence: {summary['average_confidence']:.2f}"
        cv2.putText(annotated_frame, conf_text, (position[0], position[1] + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return annotated_frame







