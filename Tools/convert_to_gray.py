import cv2
from pathlib import Path


img_path = Path('train2017_rgb')
for img_file in img_path.iterdir():
    img = cv2.imread(str(img_file))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('train2017/'+str(img_file).split('/')[-1], img_gray)

img_path = Path('train2017_rgb')
for img_file in img_path.iterdir():
    img = cv2.imread(str(img_file))
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('train2017/'+str(img_file).split('/')[-1], img_gray)
