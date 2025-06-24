from ultralytics.ultralytics import YOLO

# Load a model
#YOLOv8-P2S3A
model = YOLO("ultralytics/cfg/models/v8/yolov8-S3A-v2.yaml").float()
#YOLOv8-HWD3A
#model = YOLO("ultralytics/cfg/models/v8/yolov8-H3Av2.yaml").float()

#K:\v8_star\runs\detect\train57
# Train the model
train_results = model.train(
    data=r"K:\Parents\config.yaml",  # path to dataset YAML
    epochs=100,  # number of training epochs
    imgsz=640,  # training image size
    device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    batch=32,
    workers=0
)