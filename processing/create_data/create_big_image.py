
from PIL import Image
import os


def main():

    input_folder = "F:/Курсач/Project/data/Исходные данные/images/right/right_1"
    final_image_size = (50000, 45)
    final_image = Image.new('RGB', final_image_size)

    current_x = 0

    for filename in sorted(os.listdir(input_folder), key=lambda s: int(s[:-4])):

        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):

            image_path = os.path.join(input_folder, filename)
            image = Image.open(image_path)

            final_image.paste(image, (current_x, 0))

            current_x += image.width

    # Сохраняем окончательное изображение
    final_image.save("F:/Курсач/Project/data/Исходные данные/big/right_1.jpg")
    print("Готово!")


if __name__ == "__main__":
    main()
