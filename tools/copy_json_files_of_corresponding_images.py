from pathlib import Path
import os
import shutil


img_filenames = os.listdir('images')
print(img_filenames)

jsons_path = Path("json")

for json_file in jsons_path.iterdir():
	img_name = str(json_file).split('/')[-1].split('.')[0].replace("_label", "")+'.png'
	if img_name in img_filenames:
		src_json = str(json_file)
		dest_json = 'out_json/' + str(json_file).split('/')[-1] 
		shutil.copy(src_json, dest_json)
		print(src_json, dest_json)

print("Copy Completed............")

