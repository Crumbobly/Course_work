from ultralytics import YOLO


from PIL import ImageDraw
import numpy as np
import cv2


def yolo_predict(defectograms_500, defectograms_1000, yolo_model, margin=125):

    bbox_dict = dict()

    for i, image in enumerate(defectograms_500):
        results = yolo_model.predict(
            source=image,
            conf=0.6,
            save=False,
            imgsz=[15, 1000]
        )

        result = results[0]
        boxes = result.boxes
        image_boxes = []

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy()  # [N, 4]
            conf = boxes.conf.cpu().numpy()  # [N]
            cls = boxes.cls.cpu().numpy()  # [N]

            for j in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[j]
                confidence = conf[j]
                class_id = int(cls[j])

                # Пропустить боксы слишком близко к краям окна
                if x1 < margin or x2 > 1000 - margin:
                    continue

                image_boxes.append((x1, y1, x2, y2, confidence, class_id))

        bbox_dict[i] = image_boxes

    bbox_list = postporecess_bbox(bbox_dict)
    draw_bboxes_on_defectograms(defectograms_1000, bbox_list)
    return bbox_list


def postporecess_bbox(bbox_dict):
    bbox_list = []

    for k, v_list in bbox_dict.items():

        if not v_list:
            continue

        for v in v_list:
            start_x = v[0] + 500 * k
            end_x = v[2] + 500 * k
            class_id = v[5]
            bbox_list.append((start_x, end_x, class_id))

    # Сортируем по началу координаты
    bbox_list.sort()

    merged = []
    for bbox in bbox_list:
        start, end, class_id = bbox

        if not merged:
            merged.append([start, end, class_id])
            continue

        last_start, last_end, last_class = merged[-1]

        if start <= last_end:
            if class_id != last_class:
                raise ValueError(f"Пересечение bbox разных классов: {class_id} и {last_class}")
            merged[-1][1] = max(last_end, end)
        else:
            merged.append([start, end, class_id])

    return [tuple(b) for b in merged]


def draw_bboxes_on_defectograms(defectograms_1000, bbox_list):

    def img_draw(img, x, color, in_left=False):
        thickness = 5
        coef = -1 if in_left else 1
        width, height = img.size
        draw = ImageDraw.Draw(img)

        for t in range(thickness):
            if 0 <= x + t < width:
                draw.line([(x + t * coef, 0), (x + t * coef, height)], fill=color)

    class_colors = [
        (0, 255, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    for bbox in bbox_list:
        x1 = bbox[0]
        x2 = bbox[1]
        cls = bbox[2]

        img_number_1 = int(x1) // 1000
        img_number_2 = int(x2) // 1000
        local_x1 = int(x1) % 1000
        local_x2 = int(x2) % 1000

        img1 = defectograms_1000[img_number_1]
        img2 = defectograms_1000[img_number_2]
        color = class_colors[cls]

        img_draw(img1, local_x1, color, in_left=True)
        img_draw(img2, local_x2, color)



