# Object Classification System

A real-time object detection and classification system using YOLO (You Only Look Once) and OpenCV. This project supports both webcam and Intel RealSense cameras for object detection and tracking.

## Features

- **Real-time Object Detection**: Detect and classify objects in real-time using YOLO models
- **Multi-Camera Support**: Works with webcams and Intel RealSense cameras
- **Object Tracking**: Track objects across frames with unique IDs
- **Configurable Detection**: Adjustable confidence thresholds and class filtering
- **Visual Feedback**: Real-time display with bounding boxes, labels, and confidence scores
- **Frame Saving**: Save detection frames with 's' key
- **Error Handling**: Robust error handling for camera and model loading

## Installation

1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a YOLO model**:
   - Place a YOLO model file (e.g., `model.pt`) in the project directory
   - You can download pre-trained models from [Ultralytics](https://github.com/ultralytics/ultralytics)
   - Or use YOLOv8 models: `yolo.pt`, `yolov8n.pt`, `yolov8s.pt`, etc.

## Usage

### Basic Usage
```bash
python main.py
```

### Configuration
Edit `config.py` to adjust settings:
- Camera settings (resolution, FPS)
- Detection parameters (confidence threshold, classes)
- Display options (font size, colors)
- Model path

### Controls
- **'q'**: Quit the application
- **'s'**: Save current frame with detections

## Project Structure

```
object-classification/
├── main.py                 # Main entry point
├── script.py              # Core application logic
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── camera_utils.py    # Camera initialization and frame capture
│   └── detection_utils.py # Detection drawing and analysis
└── README.md              # This file
```

## Configuration Options

### Camera Settings
- `CAMERA_INDEX`: Camera device index (default: 0)
- `CAMERA_WIDTH/HEIGHT`: Frame resolution
- `CAMERA_FPS`: Frames per second

### Detection Settings
- `MODEL_PATH`: Path to YOLO model file
- `CONFIDENCE_THRESHOLD`: Minimum confidence for detections (0.0-1.0)
- `DETECT_CLASSES`: Specific classes to detect (None for all)
- `TRACK_OBJECTS`: Enable object tracking

### Display Settings
- `WINDOW_NAME`: Display window title
- `SHOW_CONFIDENCE`: Display confidence scores
- `SHOW_CLASS_NAMES`: Display class names
- `TEXT_COLOR`: Color for overlay text

## Supported Cameras

### Webcam
- Any USB webcam or built-in camera
- Automatically detects available cameras

### Intel RealSense
- Intel RealSense D400 series cameras
- Provides both color and depth data
- Requires `pyrealsense2` package

## Model Requirements

- YOLO model file (`.pt` format)
- Compatible with Ultralytics YOLO models
- Supports YOLOv8, YOLOv5, and other YOLO variants

## Troubleshooting

### Camera Issues
- Ensure camera is not being used by another application
- Try different camera indices (0, 1, 2, etc.)
- Check camera permissions on your system

### Model Issues
- Verify model file exists and is accessible
- Ensure model is compatible with Ultralytics
- Check model file is not corrupted

### Dependencies
- Install all requirements: `pip install -r requirements.txt`
- For RealSense: `pip install pyrealsense2`
- Ensure OpenCV is properly installed

## Examples

### Detect All Objects
```python
# In config.py
DETECT_CLASSES = None  # Detect all classes
CONFIDENCE_THRESHOLD = 0.5
```

### Detect Only People
```python
# In config.py
DETECT_CLASSES = [0]  # Person class in COCO dataset
CONFIDENCE_THRESHOLD = 0.8
```

### High Accuracy Mode
```python
# In config.py
CONFIDENCE_THRESHOLD = 0.9
IMAGE_SIZE = 640  # Higher resolution
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.