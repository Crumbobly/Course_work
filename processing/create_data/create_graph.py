from typing import List

import matplotlib.pyplot as plt


def create_graph_from_file(
        flaw_detector_data: List[List[int]],
        output_graph_file: str,
        interval_number: int,
        interval_size: int = 1000
):

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive', 'lime', 'teal', 'magenta', 'navy', 'maroon']

    # Отображение данных на графиках
    plt.figure(figsize=(150, 20))
    plt.xlim(left=0, right=interval_size)

    if interval_number is not None:
        start = interval_size * interval_number
        stop = interval_size * (interval_number + 1)
    else:
        start = None
        stop = None

    # Отображение данных на графиках
    for i in range(len(flaw_detector_data)):
        g = [x - 40 * i for x in flaw_detector_data[i][start: stop]]
        plt.plot(g, label=f'Graph {i}', color=colors[i])

    plt.savefig(output_graph_file, bbox_inches="tight")
    plt.close()

