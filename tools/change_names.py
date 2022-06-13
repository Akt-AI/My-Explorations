import json


def merge_coco_json():    
    json_file_path1 = 'person_keypoints_train2017.json'
    with open(json_file_path1) as json_file1:
        data1 = json.load(json_file1)
        print(data1.keys())
        anno1 = data1['annotations']
        images1 = data1['images']
        for i in images1:
            print(i['file_name'])

    print(anno1[0].keys())
    print(anno1[0]['image_id'])
    
    return data1
    
def write_JSON_file(json_data):
    json_object = json.dumps(json_data, indent = 4) 
    with open("Merged.json", "w") as outfile: 
        outfile.write(json_object)
    
if __name__ == "__main__":
    json_data = merge_coco_json()
    #write_JSON_file(json_data)

