import cv2
from ultralytics import YOLO
import os
# Disable GUI functionality to avoid display errors
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# Load the YOLO model with your trained weights
model = YOLO("/home/orbo/pushkar/yolo_detection/runs/detect/train6/weights/best.pt")
# Open the video file
video_path = "/home/orbo/pushkar/yolo_detection/20221532-hd_1920_1080_60fps.mp4"
cap = cv2.VideoCapture(video_path)
# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# Create output directory if it doesn't exist
output_dir = "/home/orbo/pushkar/yolo_detection/runs/detect/predict16"
os.makedirs(output_dir, exist_ok=True)
# Define the output video path with filename
output_path = os.path.join(output_dir, "tracked_output2.mp4")
# Define the output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print(f"Processing video: {video_path}")
print(f"Output will be saved to: {output_path}")
frame_count = 0
# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        frame_count += 1
        if frame_count % 30 == 0:  # Print status every 30 frames
            print(f"Processing frame: {frame_count}")
            
        # Run tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        
        # Visualize the results on the frame - customize to show only ID and class name
        annotated_frame = results[0].plot(conf=False, labels=True, tracker="bytetrack.yaml")
        
        # Write the frame to the output video
        out.write(annotated_frame)
    else:
        # Break the loop if the end of the video is reached
        break
# Release the video capture object and writer
cap.release()
out.release()
