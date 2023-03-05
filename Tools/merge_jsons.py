import json


def merge_coco_json():    
    json_file_path1 = 'Final.json'
    with open(json_file_path1) as json_file1:
        data1 = json.load(json_file1)
        print(data1.keys())
        anno1 = data1['annotations']
        images1 = data1['images']
            
    json_file_path2 = 'person_keypoints_train2017.json'
    with open(json_file_path2) as json_file2:
        data2 = json.load(json_file2)
        print(data2.keys())
        anno2 = data2['annotations']
        images2 = data2['images']
        
    anno = anno1 + anno2
    images = images1 + images2
    
    data1['annotations'] = anno 
    data1['images'] = images
    
    return data1
    
def write_JSON_file(json_data):
    json_object = json.dumps(json_data, indent = 4) 
    with open("Merged.json", "w") as outfile: 
        outfile.write(json_object)
    
if __name__ == "__main__":
    json_data = merge_coco_json()
    write_JSON_file(json_data)

