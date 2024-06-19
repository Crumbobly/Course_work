import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

# путь к весам модели
model = YOLO("/media/valery/Новый том/Курсач/ai/train/weights/best.pt")

def apply_yolo(image_path, project, name):
    model.predict(source=image_path, conf=0.75, save=True, project=project, name=name, imgsz=[45, 50000])


if __name__ == '__main__':
    input_folder = "/media/valery/Новый том/Курсач/Project/data/Исходные данные/images/right"
    output_folder = "/media/valery/Новый том/Курсач/Project/data/Исходные данные/AI_images/right"

    for dir in os.listdir(input_folder):
        apply_yolo(os.path.join(input_folder, dir), output_folder, dir)

    img = "/media/valery/Новый том/Курсач/Project/data/Исходные данные/big/right_1.jpg"
    output_folder = "/media/valery/Новый том/Курсач/Project/data/Исходные данные/big"
    
    # применяем нашу нейронную сеть к img
    apply_yolo(img, output_folder, "right_1_ai.jpg")


