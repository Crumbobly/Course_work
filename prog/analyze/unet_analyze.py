import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import cv2


# Предобработка изображения
transform = transforms.Compose([
    transforms.ToTensor()  # Преобразует PIL.Image → torch.Tensor [C, H, W] с нормализацией [0, 1]
])

colors = [
    (255, 255, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
]

class_labels = ['BG', 'Aluminotrermic', 'FlashButt', 'RailEnd']


def unet_predict(defectograms_500, unet_model):

    masks_raw = []
    masks_post = []
    all_bboxes = []

    for i, img in enumerate(defectograms_500):
        input_tensor = transform(img).unsqueeze(0)  # [1, C, H, W]

        with torch.no_grad():
            output = unet_model(input_tensor)  # [1, classes, H, W]

        pred_mask_raw = torch.argmax(output.squeeze(), dim=0).cpu().numpy()  # [H, W]
        color_mask = np.zeros((pred_mask_raw.shape[0], pred_mask_raw.shape[1], 3), dtype=np.uint8)
        for class_id in range(output.shape[1]):
            color = colors[class_id]
            color_mask[pred_mask_raw == class_id] = color

        pred_mask_post = postprocess_mask(pred_mask_raw)
        color_mask_post = np.zeros((pred_mask_post.shape[0], pred_mask_post.shape[1], 3), dtype=np.uint8)
        for class_id in range(output.shape[1]):
            color = colors[class_id]
            color_mask_post[pred_mask_post == class_id] = color

        bboxes = extract_bboxes_from_mask(pred_mask_post, i)
        all_bboxes.extend(bboxes)

        masks_raw.append(Image.fromarray(color_mask))
        masks_post.append(Image.fromarray(color_mask_post))

    final_bboxes = bboxes_remove_intersection(all_bboxes)
    return masks_raw, masks_post, final_bboxes


def postprocess_mask(mask: np.ndarray, min_area: int = 3 * 15, margin: int = 125, max_gap: int = 25,
                     final_min_area: int = 35 * 15) -> np.ndarray:
    height, width = mask.shape
    mask_copy = mask.copy()

    # приоритет постпроцессинга для классов
    class_priority = [1, 2, 3]

    # 1. Удаляем маленькие компоненты на границе и с маленькой площадью
    for class_id in class_priority:
        class_mask = (mask_copy == class_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if x < margin or x + w > width - margin or area < min_area:
                mask_copy[labels == i] = 0

    for class_id in class_priority:

        for y in range(height):
            line = (mask_copy[y] == class_id).astype(np.uint8)
            segments = []
            in_segment = False
            for x in range(width):
                if line[x] == 1 and not in_segment:
                    seg_start = x
                    in_segment = True
                elif line[x] == 0 and in_segment:
                    seg_end = x - 1
                    segments.append((seg_start, seg_end))
                    in_segment = False
            if in_segment:
                segments.append((seg_start, width - 1))

            for i in range(1, len(segments)):
                prev_end = segments[i - 1][1]
                curr_start = segments[i][0]
                if 0 < (curr_start - prev_end - 1) <= max_gap:
                    mask_copy[y, prev_end + 1:curr_start] = class_id

    # 3. Повторное удаление мелких компонентов, уже с более высоким порогом
    for class_id in class_priority:
        class_mask = (mask_copy == class_id).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < final_min_area:
                mask_copy[labels == i] = 0

    return mask_copy


def extract_bboxes_from_mask(mask: np.ndarray, index):
    bboxes = []
    num_classes = int(mask.max()) + 1

    for class_id in range(1, num_classes):  # class 0 — фон
        class_mask = (mask == class_id).astype(np.uint8)

        # Находим компоненты
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):  # пропускаем фон
            x, y, w, h, area = stats[i]
            bboxes.append((x + index * 500, x + w + index * 500, class_id))

    return bboxes


def bboxes_remove_intersection(bboxes):

    merged = []

    for bbox in sorted(bboxes):
        start, end, class_id = bbox

        if not merged:
            merged.append([start, end, class_id])
            continue

        last_start, last_end, last_class = merged[-1]

        if start <= last_end:
            if class_id == last_class:
                merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end, class_id])

    return [tuple(b) for b in merged]
