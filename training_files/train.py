from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data=r"/home/orbo/pushkar/yolo_detection/dataset/vd_dataset/data.yaml",
    epochs=120, 
    imgsz=640, 
    batch=16
)



