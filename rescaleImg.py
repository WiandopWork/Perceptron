import cv2

for i in range(11):
    for j in range(10):
        path = "data/" + str(i) + "/" + str(j) + ".bmp"

        src = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        scale_percent = 25

        width = int(src.shape[1] * scale_percent / 100) 
        height = int(src.shape[0] * scale_percent / 100)

        dsize = (width, height)
        output = cv2.resize(src, dsize)

        cv2.imwrite(path, output)
