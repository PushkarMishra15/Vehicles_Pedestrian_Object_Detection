
# Vehicles and Pedestrians Detection

## About

This project aims to detect and track vehicles (cars, cyclists) and pedestrians in video streams using cutting-edge deep learning techniques. It utilizes the YOLOv8n (You Only Look Once) object detection model in combination with a multi-object tracking algorithm (such as ByteTrack) to identify and assign persistent IDs to objects across frames. This enables accurate and consistent tracking of each object throughout the video footage.




## Dataset

This project uses the KITTI Vision Benchmark Suite, a widely-used dataset for autonomous driving and computer vision tasks. It consists of annotated images containing vehicles, pedestrians, and cyclists in various urban traffic scenarios. 
The link to the dataset is provided below.

https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

## Python Libraries Required


```bash
  pip install ultralytics
```

## Running Script

If you want to run the training script, then download the dataset and configure the data.yaml file and run the command

```bash
  cd training_files/python train.py
```

If you want to run the inference script with tracking


```bash
  cd training_files/python test_track.py
```

If you want to run the training script without tracking

```bash
  cd training_files/python test.py
```


## Performance Metrics

- Average FPS without tracking : 29.01 
- Average FPS with tracking : 18.73
- Precision : 89.8 %
- Recall : 83.3 %



## Input _Output

The Input and Output video can be find in the below link 

https://drive.google.com/drive/folders/1VRBmdEXamC-gyMik6ECrXTa65Q2TaI_e?usp=sharing