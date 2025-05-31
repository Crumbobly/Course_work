
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import cm
from matplotlib.colors import hsv_to_rgb


def create_image_from_file(data: np.ndarray, filename: str = None, save: bool = False):

    data = data.astype(np.float32)
    # Производная по времени (осуществляется по столбцам)
    derivative = np.diff(data, axis=1, prepend=data[:, [0]])

    # Ограничение экстремальных значений (для контраста)
    vlim = np.percentile(np.abs(derivative), 99)
    derivative = np.clip(derivative, -vlim, vlim)

    # Нормализация [-vlim, vlim] -> [0, 1]
    norm = (derivative + vlim) / (2 * vlim)
    norm = np.clip(norm, 0, 1)

    # Применение colormap `seismic` из matplotlib
    colormap = cm.get_cmap('seismic')
    colored = (colormap(norm)[:, :, :3] * 255).astype(np.uint8)  # RGB

    # Создание и сохранение изображения
    image = Image.fromarray(colored)
    if save:
        image.save(filename)
    return image

