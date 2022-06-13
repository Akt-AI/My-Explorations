import cv2
from pathlib import Path
import os
import matplotlib.pyplot as plt

input_path = "flu"
folders = os.listdir(input_path)
curr_dir = os.getcwd()

count=1
for folder, dir, filename in os.walk(input_path):
    for item in filename:
        print(dir, folder)
        count +=1
        img_path = curr_dir+os.path.sep+folder+os.path.sep+item
        print(img_path)
        img = cv2.imread(img_path)
        if count % 2==0:
            cv2.imwrite(img_path, img)
            os.remove(img_path)


