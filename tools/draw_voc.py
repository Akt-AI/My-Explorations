import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2


sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
classes = ["person", "other"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation():
    in_file = open('test.xml')
    out_file = open('test.txt', 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
        img = cv2.imread("img.jpg")
          
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        print("Bbox: ", b)
        img = cv2.rectangle(img, (int(b[0]), int(b[2])), (int(b[1]), int(b[3])), (255, 0,0), 2)
        img = cv2.resize(img, (512,512))
        # Displaying the image
        cv2.imshow("", img)
        cv2.waitKey(0)
        
    
    in_file.close()
    out_file.close()

convert_annotation()
