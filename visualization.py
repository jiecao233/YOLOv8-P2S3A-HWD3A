import cv2
from ultralytics.ultralytics import YOLO

# Load a model
model = YOLO(r"K:\yolo_football\v8-H3A-PLAYER-BEST\weights\best.pt")

results = model(r"K:\data_ball\train\images\frame_850.jpg", visualize=True)