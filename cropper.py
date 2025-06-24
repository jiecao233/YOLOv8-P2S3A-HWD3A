import os

import cv2
from sklearn import show_versions

from ultralytics.solutions.object_cropper import ObjectCropper
from visualization import results

#K:\v8_star\runs\detect\train57
# Train the model
cropper = ObjectCropper(
    show=True,  # display the output
    model=r"K:\v8_star\runs\detect\train57\weights\best.pt",  # model for object cropping i.e yolo11x.pt.
    classes=[0],  # crop specific classes i.e. person and car with COCO pretrained model.
    # conf=0.5,  # adjust confidence threshold for the objects.
    # crop_dir="cropped-detections",  # set the directory name for cropped detections
    crop_dir=r"K:\cropped"
)

root = r"K:\qinshu"
dst = r"K:\cropped"
for i in os.listdir(root):
    for j in os.listdir(f'{root}\\{i}'):
        if len(os.listdir(f'{root}\\{i}')) == 0:
            break
        results = cropper(f'{root}\\{i}\\{j}')
        break
    break

print(f"Total cropped objects: {results.total_crop_objects}")