import ultralytics
from ultralytics import YOLO
import torch

torch.cuda.empty_cache()
ultralytics.checks()

api_key = "eFZhOjfjEp15HGkIWOyX"
project_name = "-3-1-45-8hkhk"
# https://app.roboflow.com/workspace-q9svk/-3-1-45-8hkhk/1

def install_dataset(version: int):

    from roboflow import Roboflow

    rf = Roboflow(api_key)
    project = rf.workspace("workspace-q9svk").project(project_name)
    version = project.version(version)
    dataset = version.download("yolov8")


def push(version: int, weight_path: str):

    from roboflow import Roboflow

    rf = Roboflow(api_key)
    project = rf.workspace("workspace-q9svk").project(project_name)

    project.version(version).deploy("yolov8", weight_path)


if __name__ == "__main__":
    pass
    # загрузка датасета
    # install_dataset(1)
    
    # Обучение модели
    # model = YOLO("yolov8n.yaml")
    # result = model.train(data="datasets/dataset/data.yaml", epochs=125, imgsz=(45, 1000), batch=-1)
    
    # Проверка обученной модели
    # model = YOLO("runs/detect/train/weights/best.pt")
    # model.val(data="datasets/dataset/data.yaml")
    # model.predict(source="test/images", conf=0.6, save=True)
        
    # Зазгрузка модели на Roboflow
    # push(1, "runs/detect/train")
