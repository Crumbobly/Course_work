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
        pred_mask_post = postprocess_mask(pred_mask_raw)

        bboxes = extract_bboxes_from_mask(pred_mask_post, i)
        all_bboxes.extend(bboxes)

        # Цветная маска
        color_mask = np.zeros((pred_mask_raw.shape[0], pred_mask_raw.shape[1], 3), dtype=np.uint8)
        for class_id in range(output.shape[1]):
            color = colors[class_id]
            color_mask[pred_mask_raw == class_id] = color

        # Цветная маска
        color_mask_post = np.zeros((pred_mask_post.shape[0], pred_mask_post.shape[1], 3), dtype=np.uint8)
        for class_id in range(output.shape[1]):
            color = colors[class_id]
            color_mask_post[pred_mask_post == class_id] = color

        masks_raw.append(Image.fromarray(color_mask))
        masks_post.append(Image.fromarray(color_mask_post))

    return masks_raw, masks_post, all_bboxes


def postprocess_mask(mask: np.ndarray, min_area: int = 25 * 15, margin: int = 125, max_gap: int = 7, class_priority: list[int] = None) -> np.ndarray:
    num_classes = int(mask.max()) + 1
    height, width = mask.shape
    clean_mask = np.zeros_like(mask, dtype=np.uint8)

    if class_priority is None:
        class_priority = list(range(1, num_classes))  # по умолчанию: классы 1, 2, 3...

    for class_id in class_priority:
        class_mask = (mask == class_id).astype(np.uint8)

        # connectedComponents для фильтрации по площади и по margin
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if x < margin or x + w > width - margin:
                continue
            # if area >= min_area:
            #     clean_mask[labels == i] = class_id

        # После всех крупных объектов — заполняем пробелы
        clean_mask = fill_gaps_for_class(clean_mask, class_id, max_gap=max_gap)

    return clean_mask


def fill_gaps_for_class(mask: np.ndarray, class_id: int, max_gap: int = 25) -> np.ndarray:
    mask_filled = mask.copy()
    height, width = mask.shape

    for y in range(height):
        x = 0
        while x < width:
            if mask[y, x] == class_id:
                # ищем правый конец текущей маски
                end = x + 1
                while end < width and mask[y, end] == class_id:
                    end += 1

                # ищем начало следующего сегмента того же класса
                gap_start = end
                while gap_start < width and mask[y, gap_start] == 0:
                    gap_start += 1

                if gap_start < width and mask[y, gap_start] == class_id:
                    gap_size = gap_start - end
                    if gap_size <= max_gap:
                        mask_filled[y, end:gap_start] = class_id
                        x = gap_start
                    else:
                        x = gap_start
                else:
                    break
            else:
                x += 1

    return mask_filled


def extract_bboxes_from_mask(mask: np.ndarray, index, min_area: int = 100):
    bboxes = []
    num_classes = int(mask.max()) + 1

    for class_id in range(1, num_classes):  # class 0 — фон
        class_mask = (mask == class_id).astype(np.uint8)

        # Находим компоненты
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for i in range(1, num_labels):  # пропускаем фон
            x, y, w, h, area = stats[i]
            print("stats ", stats[i])

            if area >= min_area:
                bboxes.append((x + index * 500, x + w + index * 500, class_id))

    return bboxes
    # bboxes.sort()
    # merged = []
    # for bbox in bboxes:
    #     start, end, class_id = bbox
    #
    #     if not merged:
    #         merged.append([start, end, class_id])
    #         continue
    #
    #     last_start, last_end, last_class = merged[-1]
    #
    #     if start <= last_end:
    #         if class_id == last_class:
    #             merged[-1][1] = max(last_end, end)
    #     else:
    #         merged.append([start, end, class_id])
    #
    # return [tuple(b) for b in merged]
