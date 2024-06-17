

def parse_string(input_string: str):

    # Разбиваем входную строку на отдельные части по пробелам
    parts = [x.strip() for x in input_string.split()]

    if len(parts) < 3:
        return None

    p1 = parts[0]
    p2 = parts[1]
    args = parts[2:]

    return p1, p2, args


if __name__ == "__main__":

    input_string = "name type subtype1 subtype2 subtypeN"
    transformed_strings = parse_string(input_string)
    for s in transformed_strings:
        print(s)
