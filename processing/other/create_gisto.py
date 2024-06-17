from typing import List

import matplotlib.pyplot as plt
from collections import Counter


def create_gisto(data: List[int]):

    # Отфильтруем значения в диапазоне от -255 до 255
    # filtered_data = [x for x in data if -255 <= x <= 255]

    # Считаем количество каждого числа в отфильтрованном списке
    # counter = Counter(filtered_data)
    counter = Counter(data)

    # Получаем значения и их количество
    values = list(counter.keys())
    counts = list(counter.values())

    # Построение гистограммы
    plt.bar(values, counts)
    plt.xlabel('Значение')
    plt.ylabel('Количество')
    plt.title('Гистограмма RailEnd')
    plt.grid(True)

    plt.ylim(0, 50)
    plt.xlim(-800, 800)

    # Показать график
    plt.show()


if __name__ == "__main__":
    data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6]
    create_gisto(data)
