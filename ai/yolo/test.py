import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO


model = YOLO("runs/detect/train/weights/best.pt")


def apply_yolo(image_path, project, name):
    model.predict(source=image_path, conf=0.7, save=True, project=project, name=name, imgsz=[15, 1000], show_labels=False, save_txt=True)


if __name__ == '__main__':
    input_folder = "../Diplom_v2-1/test/images"
    output_folder = "./runs/test"

    for file in os.listdir(input_folder):
        if file.endswith(('.jpg', '.png')):
            apply_yolo(os.path.join(input_folder, file), output_folder, file)


