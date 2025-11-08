# 🚁 TESA Defence - Drone Detection & Tracking System

ระบบตรวจจับและติดตามโดรนแบบ Real-time สำหรับ Raspberry Pi 5

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Output Format](#-output-format)
- [Performance](#-performance)
- [API Integration](#-api-integration)

---

## ✨ Features

### Image Processing (OpenCV)
- ✅ **Video Processing** - Support for video files and webcam
- ✅ **Centroid Tracking** - Track objects with unique IDs
- ✅ **Euclidean Distance** - Distance-based object matching
- ✅ **Path Drawing** - Visualize tracking trajectories
- ✅ **FPS Calculation** - Real-time performance monitoring
- ✅ **Contour Detection** - Available in utils module
- ✅ **Morphology** - Erode/Dilate operations

### Deep Learning Model
- ✅ **YOLO-OBB** - Oriented Bounding Box detection
- ✅ **Multi-model Support** - Nano to XLarge variants
- ✅ **Custom Training** - Fine-tune on Google Colab
- ✅ **Raspberry Pi Optimized** - Works on Pi 5 without AI Board

### Object Tracking
- ✅ **Unique IDs** - Each drone gets persistent ID
- ✅ **Velocity Calculation** - Speed in m/s
- ✅ **Direction Tracking** - Movement direction in degrees
- ✅ **Path History** - Store and display trajectories

### Visualization
- ✅ **Bounding Boxes** - OBB with rotation
- ✅ **Colored Paths** - Different color per drone
- ✅ **Statistics Overlay** - FPS, drone count, frame info
- ✅ **Real-time Display** - Live video preview

### Data Logging
- ✅ **CSV Export** - Frame-by-frame tracking data
- ✅ **Timestamps** - Precise time tracking
- ✅ **Complete Metrics** - Position, velocity, direction

---

## 📁 Project Structure

```
tesa/
├── problem1_detection.py         # Image-based detection (static images)
├── problem1_video_tracking.py    # Video tracking system (main)
├── centroid_tracker.py           # Tracking algorithm
├── config.py                     # Configuration file
├── utils.py                      # Utility functions
│
├── yolov8m-obb.pt               # YOLO-OBB model (download separately)
│
├── videos/                       # Test videos
│   ├── video_01.mp4
│   └── ...
│
├── images/                       # Test images
│   └── drones.jpg
│
├── output/                       # Output videos (generated)
├── logs/                         # CSV logs (generated)
│
├── p1_detection_obb.csv         # Image detection results
├── p1_tracking_log.csv          # Video tracking log
│
└── README.md                     # This file
```

---

## 🔧 Requirements

### Hardware
- **Development**: PC with CUDA GPU (recommended)
- **Deployment**: Raspberry Pi 5 (no AI Board required)
- **Camera**: USB webcam or Pi Camera

### Software
```
Python >= 3.8
CUDA (optional, for GPU acceleration)
```

### Python Packages
```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
torch>=2.0.0
```

---

## 📦 Installation

### 1. Clone Repository
```bash
git clone https://github.com/tesa-defence/drone-tracking.git
cd drone-tracking
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLO-OBB Model
```bash
# Nano (fastest, for Pi5)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-obb.pt

# Medium (development)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-obb.pt
```

### 4. Create Directories
```bash
mkdir -p output logs models data
```

---

## 🚀 Usage

### 1. Image Detection (Static Images)
```bash
# Process folder of images
python problem1_detection.py --images images/drones --output detections.csv

# With custom model
python problem1_detection.py --images images/drones --model yolov8n-obb.pt --conf 0.3
```

### 2. Video Tracking (Main System)
```bash
# Process video file
python problem1_video_tracking.py --video videos/video_01.mp4 --output output/result.mp4

# Use webcam (real-time)
python problem1_video_tracking.py --video 0

# With custom settings
python problem1_video_tracking.py \
  --video videos/video_01.mp4 \
  --output output/tracked.mp4 \
  --log logs/tracking.csv \
  --model yolov8n-obb.pt \
  --conf 0.25 \
  --device cuda

# Headless mode (no display, for servers)
python problem1_video_tracking.py --video test.mp4 --output result.mp4 --no-display
```

### 3. Webcam Controls
While running:
- Press `q` - Quit
- Press `p` - Pause/Resume
- Press `s` - Save screenshot

---

## ⚙️ Configuration

Edit `config.py` to customize:

### Model Settings
```python
MODEL_CONFIG = {
    'detection_model': 'yolov8n-obb.pt',  # Model file
    'confidence_threshold': 0.25,          # Detection confidence
    'device': 'auto',                      # cuda/cpu/auto
}
```

### Tracking Parameters
```python
TRACKING_CONFIG = {
    'max_disappeared': 30,      # Frames before removing object
    'max_distance': 100,        # Max pixels to match
    'track_history': 30,        # Path length
    'pixels_per_meter': 100,    # Calibration
}
```

### Visualization
```python
VISUALIZATION_CONFIG = {
    'show_fps': True,
    'show_velocity': True,
    'path_fade_effect': True,
}
```

---

## 📊 Output Format

### CSV Log Format (`p1_tracking_log.csv`)
```csv
frame,timestamp,object_id,center_x,center_y,speed_ms,direction_deg,distance_pixels
1,1699437865.123,0,320,240,0.0,0.0,0.0
2,1699437865.156,0,325,245,2.5,45.2,7.1
3,1699437865.189,0,330,250,2.8,46.8,7.3
```

### Detection Format (`p1_detection_obb.csv`)
```csv
img_file,center_x,center_y,w,h,theta
drone1.jpg,0.512,0.345,0.123,0.089,15.2
drone1.jpg,0.678,0.234,0.098,0.076,-8.5
```

---

## 🎯 Performance

### Raspberry Pi 5 (No AI Board)
| Model | Resolution | FPS | RAM Usage |
|-------|-----------|-----|-----------|
| YOLOv8n-obb | 640x480 | 15-20 | 500 MB |
| YOLOv8s-obb | 640x480 | 10-15 | 800 MB |
| YOLOv8m-obb | 640x480 | 5-8 | 1.2 GB |

### PC with RTX 3060
| Model | Resolution | FPS | VRAM |
|-------|-----------|-----|------|
| YOLOv8n-obb | 1920x1080 | 120+ | 2 GB |
| YOLOv8m-obb | 1920x1080 | 60+ | 4 GB |
| YOLOv8x-obb | 1920x1080 | 30+ | 6 GB |

---

## 🌐 API Integration

### JSON Format for Satellite Communication
```json
{
  "time": 1699437865,
  "object": [
    {
      "frame": 120,
      "id": 1,
      "type": "DJI_Mavic",
      "lat": 13.7563,
      "lon": 100.5018,
      "velocity": 15.2,
      "direction": 45.3
    }
  ],
  "image_base64": "iVBORw0KGgoAAAANS..."
}
```

### Send Alert (TODO)
```python
# Example API integration
import requests

def send_alert(drone_data):
    response = requests.post(
        API_CONFIG['api_url'],
        json=drone_data,
        headers={'Authorization': f"Bearer {API_CONFIG['api_key']}"}
    )
    return response.status_code == 200
```

---

## 🎓 Training Custom Model

### 1. Prepare Dataset (Google Colab)
```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8n-obb.pt')

# Train
model.train(
    data='drone_dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)

# Export
model.export(format='onnx')
```

### 2. Dataset Structure
```
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/  # YOLO-OBB format
    ├── val/
    └── test/
```

### 3. Label Format (YOLO-OBB)
```
class_id center_x center_y width height angle
0 0.512 0.345 0.123 0.089 0.265
```

---

## 🐛 Troubleshooting

### No CUDA GPU detected
```bash
# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Low FPS on Raspberry Pi
```python
# Use smaller model
--model yolov8n-obb.pt

# Reduce resolution in config.py
VIDEO_CONFIG['resize_width'] = 640
```

### Import Error: scipy
```bash
pip install scipy
```

---

## 📚 Documentation

- [YOLO-OBB Documentation](https://docs.ultralytics.com/tasks/obb/)
- [Centroid Tracking Algorithm](https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

## 👥 Team

**TESA Defence Team**
- วันที่พัฒนา: 8 พฤศจิกายน 2568
- Session: อังคาร 11 พ.ย. 0900-1200 (Image Processing + DL)
- Session: พุธ 12 พ.ย. 0900-1200 (Raspberry Pi Deployment)

---

## 📝 License

This project is developed for TESA Defence training program.

---

## 🎯 Next Steps

### Phase 1: Completed ✅
- [x] Image detection with YOLO-OBB
- [x] Video processing
- [x] Centroid tracking
- [x] Velocity & direction calculation
- [x] Path visualization
- [x] FPS display
- [x] Data logging

### Phase 2: In Progress 🔄
- [ ] Custom model training
- [ ] API integration
- [ ] First alarm system
- [ ] GPS coordinate mapping
- [ ] Behavior analysis

### Phase 3: Planned 📋
- [ ] Raspberry Pi 5 deployment
- [ ] Camera integration
- [ ] Real-time performance optimization
- [ ] Multi-camera support
- [ ] Web dashboard

---

## 📞 Support

For issues and questions:
1. Check [Troubleshooting](#-troubleshooting)
2. Review [Configuration](#-configuration)
3. Contact TESA Defence Team

---

**Happy Drone Tracking! 🚁**
