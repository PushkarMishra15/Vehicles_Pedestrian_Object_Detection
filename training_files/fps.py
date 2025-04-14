import cv2
from ultralytics import YOLO
import os
import time

# Disable GUI functionality to avoid display errors
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Load your trained YOLO model
model = YOLO("/home/orbo/pushkar/vehicles_pedestrian/runs/detect/train6/weights/best.pt")

# Load video
video_path = "/home/orbo/pushkar/vehicles_pedestrian/testing_data/vedio2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output directory and file
output_dir = "/home/orbo/pushkar/vehicles_pedestrian/runs/detect/predict17/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "tracked_output2.mp4")

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_path}")

frame_count = 0
start_time = time.time()  # Start timing

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    # Track the frame
    results = model.track(frame, persist=True)

    # Annotate without confidence scores
    annotated_frame = results[0].plot(conf=False, labels=True)

    # Save the annotated frame
    out.write(annotated_frame)

    if frame_count % 30 == 0:
        print(f"Processed {frame_count} frames...")

# End timing
end_time = time.time()
total_time = end_time - start_time

# Calculate FPS
avg_fps = frame_count / total_time
print(f"\n🚀 Total frames: {frame_count}")
print(f"⏱️  Total time: {total_time:.2f} seconds")
print(f"⚡ Average FPS: {avg_fps:.2f}")

# Clean up
cap.release()
out.release()
