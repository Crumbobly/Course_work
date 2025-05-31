from typing import List
import matplotlib.pyplot as plt
import numpy as np


def create_graph_from_file(
        flaw_detector_data: np.ndarray,
        output_graph_file: str
):

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'olive', 'lime', 'teal', 'magenta', 'navy', 'maroon']

    # Отображение данных на графиках
    plt.figure(figsize=(20, 4))
    plt.yticks([])
    plt.xlim(left=0, right=1000)

    # Отображение данных на графиках
    for i in range(len(flaw_detector_data)):
        g = [x - 40 * i for x in flaw_detector_data[i]]
        plt.plot(g, label=f'Graph {i}', color=colors[i])

    plt.savefig(output_graph_file, bbox_inches="tight")
    plt.close()

