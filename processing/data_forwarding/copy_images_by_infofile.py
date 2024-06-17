import os
import shutil

from processing.data_forwarding.parse_info_string import parse_string


def copy_images(info_file, images_folder, output_folder):
    """
    :param info_file: Файл с записями о найденных объектах
    :param images_folder: Папка с папками изображений
    :param output_folder: Папка для копирования изображений
    """

    objects = ["Сварка", "RailEnd"]
    image_counter = 1

    with open(info_file, 'r', encoding="UTF-8") as file:

        for line in file:

            # парсим строку
            parsed_string = parse_string(line)

            # если не тот формат переходим на следующую строку
            if parsed_string is not None:
                folder_name, image_name, types = parsed_string
            else:
                continue

            # имя должно быть число (убираем случаи x-y)
            try:
                int(image_name)
            except ValueError:
                continue

            for o in objects:
                if o.lower() in [string.lower() for string in types]:
                    break

            else:
                # объект не подходит под паттерн, идём к следующей строке.
                continue

            # Получение имени изображения и пути к нему
            image_name = image_name + ".jpg"
            image_path = os.path.join(images_folder, folder_name, image_name)


            # Создание нового имени для изображения
            new_image_name = str(image_counter) + ".jpg"
            new_image_path = os.path.join(output_folder, new_image_name)

            # Копирование изображения с новым именем
            shutil.copy(image_path, new_image_path)

            # Увеличение счетчика для следующего изображения
            image_counter += 1


if __name__ == "__main__":
    copy_images("../../data/info.txt", "../../data/Исходные данные/images/left", "../../data/Размеченные данные/3/Dataset")
