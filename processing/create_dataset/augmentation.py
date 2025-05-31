import numpy as np

import os
import shutil
import random
from PIL import Image
import numpy as np


def channel_dropout(img):
    data = np.array(img).astype(np.float32)
    for i in range(data.shape[0]):
        if np.random.rand() < 0.3:
            data[i, :] = [255, 255, 255]

    data = np.clip(data, 0, 255).astype(np.uint8)
    return Image.fromarray(data)


def process_dir(src_dir, dst_dir, fraction=0.3):
    files = os.listdir(src_dir)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    n_aug = int(len(image_files) * fraction)
    to_augment = set(random.sample(image_files, n_aug))

    for fname in image_files:
        src_img_path = os.path.join(src_dir, fname)
        src_ann_path = os.path.join(src_dir, os.path.splitext(fname)[0] + '.txt')

        if fname in to_augment:

            img = Image.open(src_img_path)
            img_aug = channel_dropout(img)

            new_fname = 'aug_' + fname
            img_aug.save(os.path.join(dst_dir, new_fname))

            # Копируем анотацию с новым именем (т.к. для дефектограмм обычно не меняется)
            if os.path.exists(src_ann_path):
                shutil.copy(src_ann_path, os.path.join(dst_dir, os.path.splitext(new_fname)[0] + '.txt'))


if __name__ == "__main__":
    # process_dir(
    #     "../../data/Размеченные данные/23_05_25_deriv/original/annotated",
    #     "../../data/Размеченные данные/23_05_25_deriv/augmentation/annotated",
    # )
    #
    # process_dir(
    #     "../../data/Размеченные данные/23_05_25_deriv/original/null",
    #     "../../data/Размеченные данные/23_05_25_deriv/augmentation/null",
    # )

    pass
