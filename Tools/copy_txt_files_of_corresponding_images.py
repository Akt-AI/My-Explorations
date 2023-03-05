from pathlib import Path
import os
import shutil


img_filenames = os.listdir('images')
print(img_filenames)

#if os.path("out_txt") is not exists: 
    #os.mkdir('out_txt')
    
jsons_path = Path("ELM_txt")

for json_file in jsons_path.iterdir():
	img_name = str(json_file).split('/')[-1].split('.')[0].replace("_label", "")+'.jpg'
	if img_name in img_filenames:
		src_json = str(json_file)
		dest_json = 'out_txt/' + str(json_file).split('/')[-1] 
		shutil.copy(src_json, dest_json)
		print(src_json, dest_json)

print("Copy Completed............")

