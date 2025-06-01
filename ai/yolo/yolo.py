import ultralytics
from ultralytics import YOLO

import torch

# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))
# print(ultralytics.checks())



if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="../Diplom_v3-1/data.yaml",
        epochs=50,
        imgsz=[15, 1000],
        batch=4,
        patience=15
    )


# YOLO11n summary: 181 layers, 2,590,425 parameters, 2,590,409 gradients, 6.4 GFLOPs

# EarlyStopping: Training stopped early as no improvement observed in last 10 epochs. Best results observed at epoch 17, best model saved as best.pt.
# To update EarlyStopping(patience=10) pass a new patience value, i.e. `patience=300` or use `patience=0` to disable EarlyStopping.
#
# 27 epochs completed in 0.329 hours.
# Optimizer stripped from runs\detect\train\weights\last.pt, 5.4MB
# Optimizer stripped from runs\detect\train\weights\best.pt, 5.4MB
#
# Validating runs\detect\train\weights\best.pt...
# Ultralytics 8.3.141  Python-3.13.3 torch-2.7.0+cu128 CUDA:0 (NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096MiB)
# YOLO11n summary (fused): 100 layers, 2,582,737 parameters, 0 gradients, 6.3 GFLOPs
#                  Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 32/32 [00:02<00:00, 13.06it/s]
#                    all        249        169      0.907      0.955      0.993      0.782
#         Aluminothremic         37         37          1        0.9      0.991      0.853
#              Flashbutt        127        127          1      0.966      0.994      0.679
#                RailEnd          5          5       0.72          1      0.995      0.816
# Speed: 0.4ms preprocess, 3.0ms inference, 0.0ms loss, 2.1ms postprocess per image
# Results saved to runs\detect\train
#
# Process finished with exit code 0


