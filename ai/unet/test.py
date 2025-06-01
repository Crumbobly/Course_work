import os

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ai.unet.UNetModel import UNetModel

num_classes = 4
class_labels = ['BG', 'Aluminotrermic', 'FlashButt', 'RailEnd']  # Укажи свои названия классов
colors = [
    [255, 255, 255],
    [255, 0, 0],
    [0, 0, 255],
    [0, 255, 255],
]

model = UNetModel(in_channels=3, out_channels=4)
model.load_state_dict(torch.load('runs/23_05_25/unet/unet_v2_epoch99.pth'))
model.eval()

# ======== Подготовка изображения ========
transform = transforms.Compose([
    transforms.ToTensor()
])


path = '../Diplom_v2-1/test/images'
for img in os.listdir(path):
    img_path = path + '/' + img
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # добавляем batch размер

    with torch.no_grad():
        output = model(input_tensor)  # [1, classes, H, W]
    pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # [H, W]

    color_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_id in range(num_classes):
        color_mask[pred_mask == class_id] = colors[class_id]

    # Подсчёт пикселей по классам
    unique_classes, counts = np.unique(pred_mask, return_counts=True)
    # print("Количество пикселей по классам:")
    # for cls, cnt in zip(unique_classes, counts):
    #     print(f"{class_labels[cls]} (class {cls}): {cnt} пикселей")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.title('Predicted Mask')
    plt.imshow(color_mask)

    patches = [mpatches.Patch(color=np.array(color) / 255, label=label) for color, label in zip(colors, class_labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"./runs/unet/test/{img}", bbox_inches='tight')
    plt.close()

