import skimage.io as io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import json
from pycocotools.coco import COCO


parent_dir = './'
img_dir = 'val2017'
data_dir = 'annotations_trainval2017'
data_file_name = 'instances_val2017.json'
annFile = os.path.join(data_dir, data_file_name)
coco=COCO(annFile)

# Opening annotation File
data_file = open(annFile,'r')
# loading json data from annotations file
data = json.load(data_file)

# Initializing images and annotations data
IMAGES = data['images']
ANNOTATIONS = data['annotations']

# COLORS
RED = (255, 0, 0)
BLUE = (0, 0, 225)
GREEN = (0, 255, 0)


def get_img_obj(index):
    return IMAGES[index]
    
def get_category(img_id):
    for cat in data['categories']:
        if cat['id'] == img_id:
            return cat

def get_img(path):
    return cv2.imread(path)

def get_img_data(id):
    for img_obj in ANNOTATIONS:
        if id == img_obj['image_id']:
            return img_obj

def draw_rectangle(img, bbox):
    start_point = (round(bbox[0]), round(bbox[1]))
    end_point =  (round(bbox[2])+round(bbox[0]), 
                  round(bbox[3])+round(bbox[1]))

    cv2.rectangle(img, start_point, end_point, color = BLUE, thickness=2)

def draw_segmentation(img, segmentations):
    i = 0
    while i < len(segmentations):
        x, y = round(segmentations[i]), round(segmentations[i+1])
        i += 2
        cv2.circle(img, (x,y), radius=0, color=RED, thickness=3)

def save_masking_img(img_data, img_obj, save_path):
    img_box_coord = img_data['bbox']
    img_seg_coords = img_data['segmentation']
    category = get_category(img_data['category_id'])
    catIds = img_data['category_id']
    I = io.imread(os.path.join(img_dir, img_obj['file_name']))
    plt.imshow(I);plt.axis('off')
    annIds = coco.getAnnIds(imgIds=img_obj['id'], catIds=catIds)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def get_cropped_img(img, bbox):
    x, y = round(bbox[0]), round(bbox[1])
    w, h =  round(bbox[2]), round(bbox[3])
    return img[y:y+h, x:x+w]


# file names for saving cropped and masked images
EDITED_IMG_DIR = './edited_img'
CROPPED_IMG_DIR = os.path.join(EDITED_IMG_DIR, 'cropped')
MASKED_IMG_DIR = os.path.join(EDITED_IMG_DIR, 'masked')

# Creating directory tree
if not os.path.exists(EDITED_IMG_DIR):
    os.mkdir(EDITED_IMG_DIR)
    
if not os.path.exists(CROPPED_IMG_DIR):
    os.mkdir(CROPPED_IMG_DIR)
    
if not os.path.exists(MASKED_IMG_DIR):
    os.mkdir(MASKED_IMG_DIR)


# No of images we want to mask and crop
NO_OF_IMAGES = 10

"""
MAIN
"""
def main():
    for i in range(NO_OF_IMAGES):
        # Getting image object
        img_obj = get_img_obj(i)
        img_file_name = img_obj['file_name']
        img_path = os.path.join(img_dir, img_file_name)
        img = get_img(img_path)
        img_id = img_obj['id']
        img_data = get_img_data(img_id)
        img_box_coord = img_data['bbox']
        img_seg_coords = img_data['segmentation']
        category = get_category(img_data['category_id'])

        ################# MASKED IMAGE #################
        # path for cropped images to be saved
        super_cat_dir = os.path.join(MASKED_IMG_DIR, category['supercategory'])
        sub_cat_dir = os.path.join(super_cat_dir, category['name'])

        # if super category does not exist
        if not os.path.exists(super_cat_dir):
            os.mkdir(super_cat_dir)
        # if sub category does not exist
        if not os.path.exists(sub_cat_dir):
            os.mkdir(sub_cat_dir)

        # Masking image
        masked_img_save_path = os.path.join(sub_cat_dir,img_file_name)
        save_masking_img(img_data, img_obj,masked_img_save_path) 

        ################# CROPPED IMAGE #################
        img = get_img(img_path)
        
        # Drawing rectangle
        draw_rectangle(img, img_box_coord)

        # Drawing segments points
        for coord_i, coord in enumerate(img_seg_coords):
            draw_segmentation(img, coord)

        # Cropping image
        crop_img = get_cropped_img(img, img_box_coord)
        
        super_cat_dir = os.path.join(CROPPED_IMG_DIR, category['supercategory'])
        sub_cat_dir = os.path.join(super_cat_dir, category['name'])

        # if super category does not exist
        if not os.path.exists(super_cat_dir):
            os.mkdir(super_cat_dir)
        # if sub category does not exist
        if not os.path.exists(sub_cat_dir):
            os.mkdir(sub_cat_dir)
            
        # Saving image
        img_save_path = os.path.join(sub_cat_dir , img_file_name)
        cv2.imwrite(img_save_path, crop_img)


if __name__ == '__main__':
    main()
