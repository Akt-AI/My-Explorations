import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np


"""<annotation>
 	<folder>cough_croppedsample_cough1</folder>
 	<filename>000.jpg</filename>
 	<path>/home/arun/Desktop/flu/cough_croppedsample_cough1/000.jpg</path>
 	<source>
 		<database>Unknown</database>
 	</source>
 	<size>
 		<width>224</width>
 		<height>224</height>
 		<depth>3</depth>
 	</size>
 	<segmented>0</segmented>
 	<object>
 		<name>person</name>
 		<pose>Unspecified</pose>
 		<truncated>0</truncated>
 		<difficult>0</difficult>
 		<bndbox>
 			<xmin>76</xmin>
 			<ymin>16</ymin>
 			<xmax>162</xmax>
 			<ymax>208</ymax>
 		</bndbox>
 	</object>
 </annotation>"""

def indent(elem, level=0):
  i = "\n" + level*"     "
  if len(elem):
    if not elem.text or not elem.text.strip():
      elem.text = i + "     "
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
    for elem in elem:
      indent(elem, level+1)
    if not elem.tail or not elem.tail.strip():
      elem.tail = i
  else:
    if level and (not elem.tail or not elem.tail.strip()):
      elem.tail = i

def generate_xml(preds, img_path, output_path_xml):
    img = cv2.imread(img_path)
    shape = []
    width, height, depth = img.shape[0], img.shape[1], img.shape[2]
    print(width, height, depth)
    shape.append(width)
    shape.append(height)
    shape.append(depth)
    bboxes = preds
    print(bboxes)
    annotation = ET.Element('annotation')
    folder = ET.SubElement(annotation, 'folder')
    filename = ET.SubElement(annotation, 'filename')
    path = ET.SubElement(annotation, 'path')
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    height = ET.SubElement(size, 'height')
    depth = ET.SubElement(size, 'depth')
    segmented = ET.SubElement(annotation, 'segmented')
    object = ET.SubElement(annotation, 'object')
    pose = ET.SubElement(object, 'pose')
    truncated = ET.SubElement(object, 'truncated')
    difficult = ET.SubElement(object, 'difficult')
    
    folder.text = 'output_'
    filename.text = img_path.split("/")[-1]
    path.text = img_path
    database.text = "Unknown"
    width.text = str(shape[0])
    height.text = str(shape[1])
    depth.text = str(shape[2])
    pose.text = "Unspecified"
    segmented.text = "0"
    difficult.text = "0"
    for i in range(len(bboxes)):
    	name = ET.SubElement(object, 'name')
    	bndbox = ET.SubElement(object, 'bndbox')
    	xmin = ET.SubElement(bndbox, 'xmin')
    	ymin = ET.SubElement(bndbox, 'ymin')
    	xmax = ET.SubElement(bndbox, 'xmax')
    	ymax = ET.SubElement(bndbox, 'ymax')
    	name.text = 'Fixed'
    	#name.text = bboxes[i][0]
    	ymin.text = str(bboxes[i][1])
    	xmin.text = str(bboxes[i][2])
    	xmax.text = str(bboxes[i][3])
    	ymax.text = str(bboxes[i][4])

    indent(annotation)
    ET.ElementTree(annotation).write(output_path_xml)

"""
if __name__=="__main__":
	#bboxes = [["person","2","3","4","5"], ["hourse","2","3","4","5"]]
	generate_xml(preds, img_path, output_path_xml)"""





