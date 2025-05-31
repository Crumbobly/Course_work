import os

from processing.create_raw_data.create_graph import create_graph_from_file
from processing.create_raw_data.create_image import create_image_from_file
from processing.file_parsing.get_data_from_file import get_transpose_data_from_file


import numpy as np


def create_graph_and_image(data_folder, side="left", create_graph=True):
    for file in os.listdir(f"{data_folder}/txts/{side}"):
        print(file)
        if file.endswith("left_1.txt"):

            name = file[:-4]
            input_txt_file = os.path.join(data_folder + f"/txts/{side}", file)
            flaw_detector_data = get_transpose_data_from_file(input_txt_file)  # shape: [15, N]

            image_folder = f"{data_folder}/union_images/{side}/{name}"
            graph_folder = f"{data_folder}/graphs/{side}/{name}"
            os.makedirs(image_folder, exist_ok=True)

            if create_graph:
                os.makedirs(graph_folder, exist_ok=True)

            for i in range(len(flaw_detector_data[0]) // 1000):
                values = flaw_detector_data[:, i * 1000: (i + 1) * 1000]

                output_img_file = f"{image_folder}/{i}.png"
                create_image_from_file(values, output_img_file)

                if create_graph:
                    output_graph_file = f"{graph_folder}/{i}.jpg"
                    create_graph_from_file(values, output_graph_file)


def main():

    # create_graph_and_image(
    #     "../../data/Исходные данные",
    #     "left",
    #     False
    # )

    pass


if __name__ == "__main__":
    main()
