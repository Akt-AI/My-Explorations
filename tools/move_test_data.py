import os
from pathlib import Path
import shutil
import random
random.seed(17)


train_data_path = Path("train_ir_data")
data_ = os.listdir("train_ir_data")

test_data_size = int(len(data_)*0.5)
test_data_size_ = int(test_data_size*0.3)

img_test_data_list = []
for i in range(test_data_size_):
    img_filename = random.choice(data_)
    if "jpg" in img_filename:
        img_test_data_list.append(img_filename)
    
path = 'test_ir_data'
if os.path.exists(path) is False:
    os.mkdir(path)
    
img_test_data_list = set(img_test_data_list)
for i in img_test_data_list:
    print(i.split(".")[0])
    img_src = "train_ir_data/" + i
    img_dest = "test_ir_data/" + i.split(".")[0] + ".jpg"
    shutil.move(img_src, img_dest)
    
    txt_src = "train_ir_data/" + i.split(".")[0] + ".txt"
    txt_dest = "test_ir_data/" + i.split(".")[0] + ".txt"
    shutil.move(txt_src, txt_dest)
    
print("Total data: ", int(len(data_)*0.5))
print("img_test_data_list", len(img_test_data_list)) 
