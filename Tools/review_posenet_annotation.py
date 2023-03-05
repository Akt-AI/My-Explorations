import cv2
import json as j
import numpy as np
import os
import shutil


IMG_FOLDER = "./images"
JSON_FOLDER = "./json"
REVIEW_FOLDER = "./review"

def validate_json_dir_and_files():
    if os.path.exists(JSON_FOLDER):
        json_list = os.listdir(JSON_FOLDER)
        if len(json_list) > 0:
            json_list.sort()
        else:
            print("[ERROR]: JSON folder is empty.")
            return False
    else:
        print("[ERROR]: JSON folder does not exist.")
        return False
    return True

def validate_img_dir_and_process():
    if os.path.exists(IMG_FOLDER):
        img_list = os.listdir(IMG_FOLDER)
        if len(img_list) > 0:
            img_list.sort()
        else:
            print("[ERROR]: Images folder is empty.")
            return None, False
    else:
        print("[ERROR]: Images folder does not exist.")
        return None, False
    return img_list, True
    
def validate_review_dir():
    if os.path.exists(REVIEW_FOLDER):
        shutil.rmtree(REVIEW_FOLDER)
        os.mkdir(REVIEW_FOLDER)
    else:
        os.mkdir(REVIEW_FOLDER)
    
        
def parse_json(json_filename, img, img_mask):
    with open(JSON_FOLDER + os.path.sep + json_filename) as outfile:
        d = j.load(outfile)
    outfile.close()
        
    for key, val in d.items():
        if 'img' in key:
            for img_key, img_val in val.items():
                if json_filename.split(".")[0].replace("_label", "") in img_key:
                    for inner_img_key, inner_img_val in img_val.items():
                        if 'annotations' in inner_img_key:
                            for i in range(len(inner_img_val)):
                                for anno_keys, anno_vals in inner_img_val[i].items():
                                    if 'bbox' in anno_keys:
                                        if anno_vals is not None:
                                            bbox_val = d['img'][img_key]['annotations'][i]['bbox']
                                            cv2.rectangle(img, (int(bbox_val['xmin']), int(bbox_val['ymin'])), 
                                                (int(bbox_val['xmax']), int(bbox_val['ymax'])), (0,255,0), 2)
                                            
                                    if 'polygon' in anno_keys:
                                        if anno_vals is not None:
                                            temp_polygon = d['img'][img_key]['annotations'][i]['polygon']['look']
                                            co_ords_val = d['img'][img_key]['annotations'][i]['polygon']['points']
                                            temp = d['img'][img_key]['annotations'][i]['polygon']['parameters'][0]['val']
                                            
                                            if temp_polygon == 'polygon':
                                                for k in range(len(co_ords_val)):
                                                    cv2.circle(img, (int(co_ords_val[k]['x']), int(co_ords_val[k]['y'])), 3, (0,0,255), 2)
                                            
                                            if temp_polygon == 'point':
                                                cv2.circle(img, (int(co_ords_val[0]['x']), int(co_ords_val[0]['y'])), 3, (255,0,0), 2)
    
    cv2.imwrite(REVIEW_FOLDER + os.path.sep + json_filename.replace("_label.json", ".png"), img)


def main():
    validate_review_dir()
    ret_json = validate_json_dir_and_files()
    if ret_json:
        list_images, ret_img = validate_img_dir_and_process()
        if ret_img:
            for filename in list_images:
                if filename is not None:
                    print("[INFO]: Processing file ", filename)
                    name, ext = filename.split(".")
                    img = cv2.imread(IMG_FOLDER + os.path.sep + filename)
                    h, w, c = img.shape
                    img_mask = np.zeros(img.shape, np.uint8)
                    json_filename = name + "_label.json"
                    parse_json(json_filename, img, img_mask)
        else:
            print("[INFO]: Terminating due to some error related to images folder and files.")
    else:
        print("[INFO]: Terminating due to some error related to JSON files and folder.")


if __name__ == "__main__":
    main()
