from pathlib import Path
import os


img_files = []
path = Path("data")
for i in path.iterdir():
    filename = str(i)
    if ".jpg" in filename:
        img_files.append(filename.split("/")[-1])
        
print(len(img_files))
print(int(len(img_files)*0.2))
print(img_files)

test_split = int(len(img_files)*0.2)
os.chdir("data")
curr_dir = os.getcwd()
os.chdir("..")

for img_name in img_files[:-test_split]:
    with open("train.txt", 'a') as f:
        f.write(curr_dir + os.path.sep + img_name+"\n")
    
for img_name in img_files[-test_split:]:
    with open("test.txt", 'a') as f:
        f.write(curr_dir + os.path.sep + img_name+"\n")
            
        
