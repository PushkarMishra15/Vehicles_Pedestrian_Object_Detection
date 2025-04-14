from ultralytics import YOLO
import time
import cv2

# Load the YOLO model
model = YOLO("/home/orbo/pushkar/yolo_detection/runs/detect/train6/weights/best.pt")

# Video path
video_path = "/home/orbo/pushkar/vehicles_pedestrian/testing_data/vedio2.mp4"

# Count frames using OpenCV
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# Start timer
start_time = time.time()

# Run inference on the full video
results = model(video_path, save=True)

# End timer
end_time = time.time()

# Calculate total time and FPS
total_time = end_time - start_time
fps = frame_count / total_time if total_time > 0 else 0

print(f"\n📹 Processed {frame_count} frames in {total_time:.2f} seconds.")
print(f"⚡ Average FPS: {fps:.2f}")
