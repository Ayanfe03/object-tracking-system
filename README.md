# YOLOv8 Object Tracking System

## Overview
This project implements object detection and tracking on videos using the YOLOv8 model from the Ultralytics library.  
The script processes each frame, tracks objects across frames, saves an annotated video, and exports a JSON file with per-frame detections.

## Features
- Detects and tracks objects across video frames.
- Saves annotated video with bounding boxes, object IDs and class names.
- Exports a `detections.json` file with structure:
  ```json
  {
    "Frame_1": [{"id":1,"class":"car"}, {"id":2,"class":"person"}],
    "Frame_2": [...]
  }
  ```
- Displays key metrics:
  - Total frames processed
  - Total objects detected
  - Average detections per frame
  - Unique IDs tracked
  - Processing speed (FPS)

---

## Step-by-Step Approach

1. **Model Selection**  
   - Use `yolov8s.pt` (small version of YOLOv8) for real-time detection and tracking.
   
2. **Video Reading**  
   - Open video using `cv2.VideoCapture` to extract frames.
   
3. **Object Detection & Tracking**  
   - Use `model.track()` with `stream=True` for efficient memory usage.
   - Extract `track_id` and `class_name` for each object.
   
4. **Annotations**  
   - Draw bounding boxes, object IDs, and class names on frames.
   
5. **Output Storage**  
   - Save annotated video using `cv2.VideoWriter`.
   - Collect per-frame detections into a Python list.
   
6. **Metrics Calculation**  
   - Count total frames, detections, unique tracked IDs, and calculate FPS.
   
7. **JSON Export**  
   - Save all detections in a structured JSON file.
   
8. **Results**  
   - Produce both a playable annotated video and a JSON log for further analysis.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/object-tracking-system.git
   cd object-tracking-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model**  
   The script will auto-download `yolov8s.pt` if it’s not in your directory.

---

## Usage

```bash
python src/tracker.py \
    --video /path/to/video.avi \
    --model yolov8s.pt \
    --output-folder tracked \
    --output-video realtime_tracked.mp4 \
    --output-json detections.json
```

### Example
```bash
python src/tracker.py --video "sample_video.avi"
```

---

## Output
- **Annotated Video** → saved in `tracked/`
- **Detections JSON** → saved in `tracked/detections.json`

---

## Requirements
- Python 3.8+
- GPU recommended for faster processing
- OpenCV
- Ultralytics YOLOv8
- NumPy

---

## Author
Oluwasuan Ayanfeoluwa
