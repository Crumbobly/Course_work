M = 0

for i in range(1, 200):
    f = open(f"../data/txts/left/left_{i}.txt")
    lines = [[int(j) for j in i.split()] for i in f.readlines()]
    f.close()

    m = 0
    for row in lines:
        for item in row:
            if abs(item) > m:
                m = abs(item)
                M = m

    print(m)

print("max: ", M)
