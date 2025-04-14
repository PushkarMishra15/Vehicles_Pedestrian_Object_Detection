from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data=r"C:/Users/Pushkar/Downloads/Project/vehicles_pedestrian/dataset/data.yaml",
    epochs=120, 
    imgsz=640, 
    batch=16
)



