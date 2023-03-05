import json
from pathlib import Path
from collections import OrderedDict
import os
import xml.etree.ElementTree as ET


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

def generate_xml(preds, img_file_name, shape, output_dir, output_label, output_path_xml):
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
    
    folder.text = output_dir.split("_")[0]
    filename.text = img_file_name
    path.text = os.getcwd()+os.path.sep+img_file_name
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
    	#name.text = output_label
    	name.text = bboxes[i][0]
    	xmin.text = str(bboxes[i][1])
    	ymin.text = str(bboxes[i][2])
    	xmax.text = str(bboxes[i][3])
    	ymax.text = str(bboxes[i][4])

    indent(annotation)
    ET.ElementTree(annotation).write(output_path_xml)
    

def find_bb_in_nat_json(output_dir, output_label, json_path):
    error_log = []
    output_dir = output_dir + "_xml"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    for jsonfile in json_path.iterdir():
        json_label = str(jsonfile).split('.json')[0].replace("_label", "").split('/')[-1]
        
        img_file_name = json_label + ".jpg"
        with open(str(jsonfile), "r") as read_file:
            data = json.load(read_file)
            
        json_label = os.path.basename(str(jsonfile))
        json_label = json_label.split('.json')[0].replace("_label", "")
        
        anno = data['img'][json_label]['annotations']

        error_dict = {}
        annotated_label_list = []
        errors = []
        error_in_file = []
        bboxes = []
        for i in anno:
            for k, v in i.items():
                if k is not None and v is not None:
                    if k=='bbox':
                        if "xmin" and "ymin" and "xmax" and "ymax" in v.keys():
                            person_anno = ["person", v["xmin"], v["ymin"], v["xmax"], v["ymax"]] 
                            bboxes.append(person_anno)
                 
        width = data['width']
        height = data['height'] 
        shape = [width, height, 3]
        
        output_path_xml = output_dir + os.path.sep + img_file_name.split(".")[0]+ '.xml'
        generate_xml(bboxes, img_file_name, shape, output_dir, output_label, output_path_xml)
                
                
if __name__ == "__main__":
    input_dir = "json_all_single_class"
    output_dir = input_dir.split("_")[0]
    output_label = output_dir+"_person"
    print(output_dir, output_label)
    json_path = Path(input_dir)
    find_bb_in_nat_json(output_dir, output_label, json_path)
    print("Conversion Completed....")
