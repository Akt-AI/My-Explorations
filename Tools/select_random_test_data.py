import random
import os
import shutil


random.seed(10)
images = os.listdir("images")
total_no_of_images_to_select = int(len(images)*0.2)

selected_images = []
for _ in range(total_no_of_images_to_select):
    img = random.choice(images)
    selected_images.append(img)
    
selected_images = set(selected_images)
for img in selected_images:
    src_img = "images/"+img
    dest_img = "test_images/"+img
    src_json = "json/"+img.split('.')[0]+'_label.json'
    dest_json = "test_json/"+img.split('.')[0]+'_label.json'
    shutil.move(src_img, dest_img)
    shutil.move(src_json, dest_json)
    print(src_img, dest_img)
    print(src_json, dest_json)
 
print("Total No of Selected Images as test data: ", int(len(images)*0.2))

