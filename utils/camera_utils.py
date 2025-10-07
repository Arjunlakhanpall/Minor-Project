"""
Camera utility functions for object classification project
"""

import cv2
import numpy as np
from typing import Tuple, Optional

def initialize_webcam(camera_index: int = 0, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    """
    Initialize webcam with specified parameters
    
    Args:
        camera_index: Camera index (usually 0 for default camera)
        width: Frame width
        height: Frame height
    
    Returns:
        VideoCapture object
    """
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def initialize_realsense():
    """
    Initialize Intel RealSense camera
    
    Returns:
        RealSense pipeline object
    """
    try:
        import pyrealsense2 as rs
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Enable depth and color streams
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        pipeline.start(config)
        return pipeline
    except ImportError:
        print("pyrealsense2 not installed. Install with: pip install pyrealsense2")
        return None
    except Exception as e:
        print(f"Failed to initialize RealSense camera: {e}")
        return None

def get_frame_webcam(cap: cv2.VideoCapture) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get frame from webcam
    
    Args:
        cap: VideoCapture object
    
    Returns:
        Tuple of (depth_image, color_image) - depth will be None for webcam
    """
    ret, frame = cap.read()
    return None, frame if ret else None

def get_frame_realsense(pipeline) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get frame from RealSense camera
    
    Args:
        pipeline: RealSense pipeline object
    
    Returns:
        Tuple of (depth_image, color_image)
    """
    try:
        import pyrealsense2 as rs
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
            
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return depth_image, color_image
    except Exception as e:
        print(f"Error getting RealSense frame: {e}")
        return None, None

def test_camera(camera_index: int = 0) -> bool:
    """
    Test if camera is working
    
    Args:
        camera_index: Camera index to test
    
    Returns:
        True if camera is working, False otherwise
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False
    
    ret, frame = cap.read()
    cap.release()
    return ret and frame is not None







