from processing.create_data.create_data import create_graph_and_image
from processing.file_parsing.get_data_from_file import get_uninterrupted_data_from_file
from processing.other.create_gisto import create_gisto


def main():
    create_graph_and_image()

    # data = get_uninterrupted_data_from_file("C:/Users/79301/OneDrive/Рабочий стол/Курсач/Project/data/tmp/railEnd.txt")
    # create_gisto(data)


if __name__ == "__main__":
    main()


