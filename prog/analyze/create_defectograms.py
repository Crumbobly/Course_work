from processing.create_raw_data.create_image import create_image_from_file


def create_defectograms_500(data):
    defectograms = []

    N = data.shape[1]
    i = 500
    while i < N - 500:
        crop = data[:, i - 500: i + 500]
        img = create_image_from_file(crop)
        defectograms.append(img)
        i += 500

    return defectograms


def create_defectograms_1000(data):
    defectograms = []

    N = data.shape[1]
    for i in range(N // 1000):
        crop = data[:, i * 1000: (i + 1) * 1000]
        img = create_image_from_file(crop)
        defectograms.append(img)

    return defectograms
