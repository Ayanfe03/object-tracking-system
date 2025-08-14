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

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/tracker.py --video /path/to/video.avi --model yolov8s.pt --output-folder tracked --output-video realtime_tracked.mp4 --output-json detections.json
```

Example:
```bash
python src/main.py --video /sample_video.avi
```

## Note
- If yolov8s.pt is not found locally, Ultralytics will automatically download it.

## Output
- **Annotated video** saved in `tracked/`
- **detections.json** saved in `tracked/`

## Requirements
- Python 3.8+
- GPU recommended for faster processing
- YOLOv8 model (`yolov8s.pt` by default)

## Author
Oluwasuan Ayanfeoluwa
