import cv2
from pathlib import Path


paths = Path("outputs")
count = 0
for path in paths.iterdir():
    path = Path(str(path))
    for filename in path.iterdir():
        img = cv2.imread(str(filename))
        #img = cv2.resize(img, (368,368))
        img = cv2.resize(img, (368,368))
        fileame = "frame"
        count += 1
        fileame = "outputs_/frame" + str(count).zfill(5)+".png"
        cv2.imwrite(fileame, img)
