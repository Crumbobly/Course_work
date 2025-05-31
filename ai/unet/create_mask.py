import os
from PIL import Image, ImageDraw
import numpy as np


def yolo_txt_to_mask(txt_path, image_size):
    width, height = image_size
    mask = Image.new('L', (width, height), 0)  # L = grayscale, фон = 0
    draw = ImageDraw.Draw(mask)

    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, x_center, y_center, w, h = map(float, parts)
            h = 1

            # YOLO координаты → абсолютные
            x1 = int((x_center - w / 2) * width)
            y1 = int((y_center - h / 2) * height)
            x2 = int((x_center + w / 2) * width)
            y2 = int((y_center + h / 2) * height)

            draw.rectangle([x1, y1, x2, y2], fill=int(cls)+1)

    return mask


def convert_yolo_dataset_to_masks(image_dir, label_dir, output_mask_dir):
    os.makedirs(output_mask_dir, exist_ok=True)

    for file_name in os.listdir(image_dir):
        if not file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, file_name)
        label_path = os.path.join(label_dir, os.path.splitext(file_name)[0] + ".txt")
        mask_path = os.path.join(output_mask_dir, os.path.splitext(file_name)[0] + "_mask.png")

        if not os.path.exists(label_path):
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        mask = yolo_txt_to_mask(label_path, (width, height))
        mask.save(mask_path)


if __name__ == "__main__":
    convert_yolo_dataset_to_masks(
        image_dir="../Diplom_v2-1/train/images",
        label_dir="../Diplom_v2-1/train/labels",
        output_mask_dir="./train_mask",
    )
    convert_yolo_dataset_to_masks(
        image_dir="../Diplom_v2-1/test/images",
        label_dir="../Diplom_v2-1/test/labels",
        output_mask_dir="./test_mask",
    )

    convert_yolo_dataset_to_masks(
        image_dir="../Diplom_v2-1/valid/images",
        label_dir="../Diplom_v2-1/valid/labels",
        output_mask_dir="./valid_mask",
    )

