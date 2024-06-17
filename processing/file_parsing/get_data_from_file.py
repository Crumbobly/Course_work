
import re
from typing import List

from STATIC_DATA import DETECTORS_COUNT


def get_uninterrupted_data_from_file(url: str) -> List[int]:
    """
    Читает данные из файла и преобразует их в один большой список
    """

    flaw_detectors_data: List[int] = list()
    data = get_original_data_from_file(url)

    for lst in data:
        for item in lst:
            flaw_detectors_data.append(item)

    return flaw_detectors_data


def get_original_data_from_file(url: str) -> List[List[int]]:
    """
    Читает данные из файла и преобразует их в список списков.
    """

    flaw_detectors_data: List[List[int]] = list()

    with open(url, 'r', encoding="utf-8") as file:
        for line in file.readlines():

            find_numbers = re.findall(r"-?\d+", line)

            if len(find_numbers) == 0:
                continue

            if len(find_numbers) != 15:
                raise IndexError(f"Количество столбцов в файле с данными дефектоскопов не соотвествует их указанному числу {DETECTORS_COUNT}")

            flaw_detectors_data.append([int(match) for match in re.findall(r"-?\d+", line)])

    return flaw_detectors_data


def get_transposed_data_from_file(url: str) -> List[List[int]]:
    return list(map(list, zip(*get_original_data_from_file(url))))
