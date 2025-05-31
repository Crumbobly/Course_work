import os
import shutil


def parse_string(input_string: str):
    # Разбиваем входную строку на отдельные части по пробелам
    parts = [x.strip() for x in input_string.split()]

    if len(parts) < 3:
        return None

    p1 = parts[0]
    p2 = parts[1]
    args = parts[2:]

    return p1, p2, args


def copy_images(info_file, images_folder, output_folder, prefix="", objects=None):
    """
    :param info_file: Файл с записями о найденных объектах
    :param images_folder: Папка с папками изображений
    :param output_folder: Папка для копирования изображений
    :param prefix: Префикс для названия файла
    :param objects: Список имен элементов
    """

    if objects is None:
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

            image_name = image_name + ".png"
            image_path = os.path.join(images_folder, folder_name, image_name)

            new_image_name = prefix + str(image_counter) + ".png"
            new_image_path = os.path.join(output_folder, new_image_name)

            shutil.copy(image_path, new_image_path)
            print(folder_name + "/" + image_name + " : " + new_image_name)

            image_counter += 1


if __name__ == "__main__":

    # copy_images("../../data/info_left.txt",
    #             "../../data/Исходные данные/deriv_images/left",
    #             "../../data/Размеченные данные/23_05_25_deriv/dataset"
    #             )
    # copy_images(
    #     "../../data/info_null_left.txt",
    #     "../../data/Исходные данные/deriv_images/left",
    #     "../../data/Размеченные данные/23_05_25_deriv/original/null",
    #     prefix="e",
    #     objects=["null"]
    # )

    # copy_images(
    #     "../../data/info_null_left.txt",
    #     "../../data/Исходные данные/images/left",
    #     "../../data/Размеченные данные/23_05_25/null",
    #     prefix="e",
    #     objects=["null"]
    # )

    pass
