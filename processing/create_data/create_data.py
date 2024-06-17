import os

from STATIC_DATA import IMAGE_GRAPH_INTERVAL
from processing.file_parsing.get_data_from_file import get_transposed_data_from_file
from processing.create_data.create_graph import create_graph_from_file
from processing.create_data.create_image import create_image_from_file


def create_graph_and_image():

    ###
    data_folder = "./data/Исходные данные"
    ###

    for file in os.listdir(f"{data_folder}/txts/right"):
        if file.endswith(".txt"):

            name = file[:-4]
            input_txt_file = os.path.join(data_folder + "/txts/right", file)
            flaw_detector_data = get_transposed_data_from_file(input_txt_file)

            image_folder = f"{data_folder}/images/right/{name}"
            graph_folder = f"{data_folder}/graphs/right/{name}"
            os.makedirs(image_folder, exist_ok=True)
            os.makedirs(graph_folder, exist_ok=True)

            for i in range(len(flaw_detector_data[0]) // IMAGE_GRAPH_INTERVAL):
                output_img_file = f"{image_folder}/{i}.jpg"
                output_graph_file = f"{graph_folder}/{i}.jpg"

                # create_graph_from_file(flaw_detector_data, output_graph_file, i)
                create_image_from_file(flaw_detector_data, output_img_file, i, color_mode="RGB")

