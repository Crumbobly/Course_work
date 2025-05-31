
import re
from typing import List

import numpy as np


def get_transpose_data_from_file(url: str) -> np.ndarray:
    """
    Читает данные из файла и преобразует их в numpy-массив размера [15, N],
    где N — количество строк (временных точек), 15 — каналов дефектоскопа.
    """
    data = []

    with open(url, 'r', encoding="utf-8") as file:
        for line_num, line in enumerate(file, start=1):
            numbers = re.findall(r"-?\d+", line)

            if not numbers:
                continue

            if len(numbers) != 15:
                raise ValueError(f"[Строка {line_num}] Ожидалось 15 значений, получено {len(numbers)}")

            data.append([int(x) for x in numbers])

    return np.array(data, dtype=np.int32).T  # shape: [15, N]
