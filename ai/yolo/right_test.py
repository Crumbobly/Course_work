import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("runs/old/detect/train/weights/best.pt")


def apply_yolo(image_path, project, name):
    model.predict(source=image_path, conf=0.5, save=True, project=project, name=name, imgsz=[15, 1000], show_labels=False)


if __name__ == '__main__':
    input_folder = "../../data/Исходные данные/images/right/right_2"
    output_folder = "right_test"

    for file in os.listdir(input_folder):
        if file.endswith(('.jpg', '.png')):
            apply_yolo(os.path.join(input_folder, file), output_folder, file)


