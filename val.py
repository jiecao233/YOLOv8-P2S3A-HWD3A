from ultralytics.ultralytics import YOLO

# Load a model
model = YOLO(r"K:\yolo_football\v6n_player\weights\best.pt").float()

# Train the model
metrics = model.val()
print(metrics.box.map)