

def get_cls_string_yolo(num):
    if num == 0:
        return "Aluminothermic"
    elif num == 1:
        return "Flashbutt"
    elif num == 2:
        return "RailEnd"
    return "Undefined"


def get_cls_string_unet(num):
    if num == 0:
        return "BG"
    if num == 1:
        return "Aluminothermic"
    elif num == 2:
        return "Flashbutt"
    elif num == 3:
        return "RailEnd"
    return "Undefined"


def get_result_for_all(defectograms, bbox_list, model):
    s = f"Всего дефектограмм: {len(defectograms)}\n" + f"Найдено элементов: {len(bbox_list)}\n"
    for box in bbox_list:
        x1 = int(box[0])
        x2 = int(box[1])
        cls = "None"
        if model.lower() == "yolo":
            cls = get_cls_string_yolo(box[2])
        elif model.lower() == "unet":
            cls = get_cls_string_unet(box[2])

        s += f"Начало {x1} - конец {x2}, класс {cls} (Начало на {x1 // 1000}.img - конец на {x2 // 1000}.img)\n"

    return s
