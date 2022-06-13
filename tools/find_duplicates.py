import json
from pathlib import Path
from collections import Counter
import os


json_path = Path('Others_client_Json')
for jsonfile in json_path.iterdir():
    #json_label = str(jsonfile).split('.')[0].replace("_label", "").split('/')[-1]
    
    KEYPOINTS_LIST =["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
                      "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
                      "left_knee","right_knee","left_ankle","right_ankle"]

    with open(str(jsonfile), "r") as read_file:
        data = json.load(read_file)
        
    json_label = os.path.basename(str(jsonfile))
    json_label = json_label.split('.')[0].replace("_label", "")
    anno = data['img'][json_label]['annotations']

    duplicate_serarch_list = []
    for i in anno:
        for k, v in i.items():
            if k=='label' and v in KEYPOINTS_LIST:
                label = i['polygon']['label']
                x = i['polygon']['points'][0]['x']
                y = i['polygon']['points'][0]['y']
                
                if x != 0 and y !=0:
                    duplicate_serarch_list.append(label)
                    
    counts = Counter(duplicate_serarch_list)
    for key, value in counts.items():
        if value == 2:
            print(key, value, jsonfile)     
            
            with open('falult.log', "a") as write_file:
                write_file.write(str(key) + "  " + str(value) + "  "+str(jsonfile) +'\n')
                            
