# Configuration file for Object Classification Project

# Camera settings
CAMERA_INDEX = 0  # Default webcam index
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# YOLO model settings
MODEL_PATH = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.3  # Lower threshold to catch more objects
IMAGE_SIZE = 640  # Higher resolution for better detection
TRACK_OBJECTS = False

# Detection settings
DETECT_CLASSES = None  # None for all classes, or list like [0, 1, 2] for specific classes
SHOW_CONFIDENCE = True
SHOW_CLASS_NAMES = True

# Display settings
WINDOW_NAME = "Object Classification"
FONT_SCALE = 1
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 255)  # Red color in BGR
TEXT_POSITION = (50, 50)

# RealSense settings (if using Intel RealSense camera)
REALSENSE_DEPTH_RESOLUTION = (640, 480)
REALSENSE_COLOR_RESOLUTION = (640, 480)
REALSENSE_FPS = 30
