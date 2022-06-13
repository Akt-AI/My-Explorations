from pathlib import Path
import os
import shutil


json_path = Path("json_human")
imgs = os.listdir("images")

for json_file in json_path.iterdir():
    #print(str(json_file))
    img_name = str(json_file).split("/")[-1].split('.')[0].replace("_label", "")+'.png'
    if img_name in imgs:
        print(img_name)
        src = 'images/'+ img_name
        dest = 'out_images/'+ img_name
        shutil.copy(src, dest)
    
