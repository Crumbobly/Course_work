import ultralytics
from ultralytics import YOLO

import torch

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
#
# print(ultralytics.checks())
# exit()


model = YOLO("yolo11n.pt")

results = model.train(data="../Diplom_v2-1/data.yaml", epochs=50, imgsz=[15, 1000], batch=4)


