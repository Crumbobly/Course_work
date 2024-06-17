from typing import List, Tuple, Union

from PIL import Image

from STATIC_DATA import IMAGE_GRAPH_INTERVAL


def create_image_from_file(
        flaw_detector_data: List[List[int]],
        output_img_file: str,
        interval_number: int,
        interval_size: int = IMAGE_GRAPH_INTERVAL,
        color_mode: str = "RGB",
):
    """
    :param flaw_detector_data: Данные с дефектоскопа
    :param output_img_file: Имя выходного изображения
    :param interval_size: размер интервала
    :param interval_number: номер интервала
    :param color_mode: L or RGB

    Функция создаёт изображения по данным дефектоскопа. Т.к. эти данные слишком большие, функция создаёт изображения не для всего потока записей, а для его конкретного интервала.
    """

    # Определение списка цветов, для генерации изображения
    colors_ch1: Union[List[Tuple[int, int, int]], List[int]]
    colors_ch2: Union[List[Tuple[int, int, int]], List[int]]
    empty_color: Tuple[int, int, int] | int
    color_mode: str

    if color_mode == "RGB":
        colors_ch1 = [(255, i, i) for i in range(255, -1, -1)]
        colors_ch2 = [(i, i, 255) for i in range(255, -1, -1)]
        empty_color = (255, 255, 255)

    elif color_mode == "L":
        colors_ch1 = [i for i in range(128, 256)]
        colors_ch2 = [i for i in range(127, -1, -1)]
        empty_color = 128

    else:
        raise Exception("Цветовая схема не найдена.")

    # Определения начала и конца интервала по его номеру
    if interval_number is not None:
        start = interval_size * interval_number
        stop = interval_size * (interval_number + 1)
    else:
        start = None
        stop = None

    #
    rgb_data = [[] for _ in flaw_detector_data]

    m = 75
    for i in range(len(flaw_detector_data)):
        for number in flaw_detector_data[i][start: stop]:

            ###
            if -m <= number <= m:
                rgb_data[i].append(empty_color)
                continue
            ###

            if number > 0:
                rgb_data[i].append(colors_ch1[number - m + 1] if number - m + 1 < len(colors_ch1) else colors_ch1[-1])
            else:
                rgb_data[i].append(colors_ch2[-number - m + 1] if -number - m + 1 < len(colors_ch2) else colors_ch2[-1])

    # Определяем размер изображения
    width = len(rgb_data[0])
    height = len(rgb_data)

    # Создаем новое изображение с белым фоном
    image = Image.new(color_mode, (width, height * 3), empty_color)

    # Заполняем изображение цветами из данных RGB
    for y in range(0, height * 3, 3):
        for x in range(width):
            for i in range(3):
                image.putpixel((x, y), rgb_data[y // 3][x])
                image.putpixel((x, y + 1), rgb_data[y // 3][x])
                image.putpixel((x, y + 2), rgb_data[y // 3][x])

    # Сохраняем изображение
    image.save(output_img_file)
