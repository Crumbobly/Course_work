import numpy as np

import os
import shutil
import random
from PIL import Image
import numpy as np


def channel_dropout(img):
    data = np.array(img).astype(np.float32)
    for i in range(data.shape[0]):
        if np.random.rand() < 0.2:
            data[i, :] = [255, 255, 255]

    data = np.clip(data, 0, 255).astype(np.uint8)
    return Image.fromarray(data)


def horizontal_flip(img):
    return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def grayscale(img: Image.Image) -> Image.Image:
    return img.convert('L').convert('RGB')  # переводим в L, затем обратно в RGB, чтобы сохранить 3 канала


def flip_yolo_annotation(ann_path):
    with open(ann_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        cls, x_center, y_center, width, height = parts
        x_center = str(1.0 - float(x_center))
        new_line = f"{cls} {x_center} {y_center} {width} {height}"
        new_lines.append(new_line)

    return '\n'.join(new_lines)


def process_dir(src_dir, dst_dir, fraction=0.6):
    files = os.listdir(src_dir)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    n_aug = int(len(image_files) * fraction)
    to_augment = set(random.sample(image_files, n_aug))

    for fname in image_files:
        src_img_path = os.path.join(src_dir, fname)
        src_ann_path = os.path.join(src_dir, os.path.splitext(fname)[0] + '.txt')

        if fname in to_augment:
            img = Image.open(src_img_path)

            do_dropout, do_flip, do_grayscale = False, False, False
            while not do_dropout and not do_flip and not do_grayscale:
                do_dropout = random.random() < 0.4
                do_flip = random.random() < 0.4
                do_grayscale = random.random() < 0.3

            if do_dropout:
                img = channel_dropout(img)
            if do_flip:
                img = horizontal_flip(img)
            if do_grayscale:
                img = grayscale(img)

            new_fname = 'aug_' + fname
            img.save(os.path.join(dst_dir, new_fname))

            # Копируем анотацию с новым именем (т.к. для дефектограмм обычно не меняется)
            if os.path.exists(src_ann_path):
                if do_flip:
                    ann_txt = flip_yolo_annotation(src_ann_path)
                    with open(os.path.join(dst_dir, os.path.splitext(new_fname)[0] + '.txt'), 'w') as out_f:
                        out_f.write(ann_txt)
                else:
                    shutil.copy(src_ann_path, os.path.join(dst_dir, os.path.splitext(new_fname)[0] + '.txt'))


if __name__ == "__main__":
    process_dir(
        "../../data/Размеченные данные/01_06_25_deriv_v2/original/annotated",
        "../../data/Размеченные данные/01_06_25_deriv_v2/augmentation/annotated",
    )

    process_dir(
        "../../data/Размеченные данные/01_06_25_deriv_v2/original/null",
        "../../data/Размеченные данные/01_06_25_deriv_v2/augmentation/null",
    )

    pass
