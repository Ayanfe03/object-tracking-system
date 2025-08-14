import cv2
from ultralytics import YOLO
import os
import time
import json
import argparse


def main(video_path, model_path, output_folder, output_filename, json_filename):
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    all_detections = []
    unique_ids = set()
    total_objects_detected = 0

    start_time = time.time()

    results = model.track(source=video_path, stream=True, persist=True, show=False)

    for frame_result in results:
        frame_objects = []
        for obj in frame_result.boxes.data:
            if len(obj) >= 7:
                x1, y1, x2, y2, track_id, score, class_id = obj
                class_name = model.names[int(class_id)]
                frame_objects.append({
                    "id": int(track_id),
                    "class": class_name
                })
                total_objects_detected += 1
                unique_ids.add(int(track_id))
            else:
                print(f"Warning: Skipping object with insufficient data: {obj}")

        all_detections.append(frame_objects)

        annotated_frame = frame_result.plot()
        out.write(annotated_frame)

    end_time = time.time()

    cap.release()
    out.release()

    total_frames = len(all_detections)
    avg_detections_per_frame = total_objects_detected / total_frames if total_frames > 0 else 0
    processing_fps = total_frames / (end_time - start_time)

    print("\n----- SUMMARY METRICS -----")
    print(f"Total frames processed: {total_frames}")
    print(f"Total objects detected: {total_objects_detected}")
    print(f"Average detections per frame: {avg_detections_per_frame:.2f}")
    print(f"Unique IDs tracked: {len(unique_ids)}")
    print(f"Processing speed (FPS): {processing_fps:.2f}")

    print("\n-----Preview of detections (first 5 frames): ------")
    for i, frame in enumerate(all_detections[:5]):
        print(f"Frame {i+1}: {frame}")

    detections_with_frames = {f"Frame_{i+1}": frame for i, frame in enumerate(all_detections)}

    json_path = os.path.join(output_folder, json_filename)
    with open(json_path, "w") as f:
        json.dump(detections_with_frames, f, indent=2)

    print(f"\nDetections saved to: {json_path}")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Tracking")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="yolov8s.pt", help="Path to YOLOv8 model file")
    parser.add_argument("--output-folder", type=str, default="tracked", help="Folder to save outputs")
    parser.add_argument("--output-video", type=str, default="realtime_tracked.mp4", help="Output video filename")
    parser.add_argument("--output-json", type=str, default="detections.json", help="Output JSON filename")

    args = parser.parse_args()

    main(args.video, args.model, args.output_folder, args.output_video, args.output_json)
