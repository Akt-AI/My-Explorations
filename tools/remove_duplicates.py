import os
import json

KEYPOINTS_LIST = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder","right_shoulder",
                  "left_elbow","right_elbow","left_wrist","right_wrist","left_hip","right_hip",
                  "left_knee","right_knee","left_ankle","right_ankle"]

def get_random_string(t):
    return "random_staing"

def createEmptyDict(label_name):
    anno_dict = dict()
    anno_dict["bbox"] = None
    id = get_random_string(10)
    anno_dict["id"] = id
    anno_dict["imageTag"] = None
    anno_dict["label"] = label_name
    
    anno_dict["polygon"] = dict()
    anno_dict["polygon"]["annoId"] = id
    anno_dict["polygon"]["fill"] = "#fd4f74"
    anno_dict["polygon"]["id"] = get_random_string(10)
    anno_dict["polygon"]["label"] = label_name
    anno_dict["polygon"]["look"] = "point"
    anno_dict["polygon"]["opacity"] = "0.5"
    anno_dict["polygon"]["parameters"] = []
    
    anno_param_dict = dict()
    anno_param_dict["nam"] = "Id"
    anno_param_dict["val"] = 0
    
    anno_dict["polygon"]["parameters"].append(anno_param_dict)
    anno_dict["polygon"]["points"] = []
    
    anno_points_dict = dict()
    anno_points_dict["id"] = get_random_string(10)
    anno_points_dict["x"] = 0
    anno_points_dict["y"] = 0
    anno_dict["polygon"]["points"].append(anno_points_dict)
    anno_dict["polygon"]["score"] = 0
    return anno_dict
      


def correctJson(json_file_path, corrected_file_path):
    
    with open(json_file_path) as jsonfile:
        json_data = json.load(jsonfile)
        jsonfile.close()
    item = os.path.basename(json_file_path)

    default_dict = dict()
    non_default_dict = dict()
    duplicate_dict = dict()

    # Check which labels default labels and which are not.
    for key, val in json_data.items():
            if 'img' in key:
                for img_key, img_val in val.items():
                    if item.split(".")[0].replace("_label", "") in img_key:
                        for inner_img_key, inner_img_val in img_val.items():
                            if 'annotations' in inner_img_key:
                                for i in range(len(inner_img_val)):
                                    current_dict = inner_img_val[i]
                                    current_dict_keys = inner_img_val[i].keys()
                                    if 'label' not in current_dict_keys:
                                        continue
                                    current_label = current_dict['label']

                                    if 'polygon' not in current_dict_keys:
                                        continue
                                    polygon = current_dict['polygon']
                                    if polygon is None:
                                        continue

                                    if 'points' not in polygon:
                                        continue
                                    points = polygon['points']
                                    
                                    is_default = False
                                    if len(points) == 1:
                                        point = points[0]
                                        if point['x'] == 0 and point['y'] == 0:
                                            default_dict[current_label] = current_dict
                                            is_default = True

                                    if not is_default:
                                        non_default_dict[current_label] = current_dict
                                    pass

    # Check which labels are duplicates.
    for default_lable in default_dict.keys():
        if default_lable in non_default_dict.keys():
            duplicate_dict[default_lable] = default_dict[default_lable]

    # Removing duplicate lables.
    for key, val in json_data.items():
            if 'img' in key:
                for img_key, img_val in val.items():
                    if item.split(".")[0].replace("_label", "") in img_key:
                        for inner_img_key, inner_img_val in img_val.items():
                            if 'annotations' in inner_img_key:
                                for duplicate_label in duplicate_dict:
                                    dublicate_data = duplicate_dict[duplicate_label]
                                    if dublicate_data in inner_img_val:
                                        inner_img_val.remove(dublicate_data)

    # Checking finally which lables are there.
    final_lables = list()
    for key, val in json_data.items():
            if 'img' in key:
                for img_key, img_val in val.items():
                    if item.split(".")[0].replace("_label", "") in img_key:
                        for inner_img_key, inner_img_val in img_val.items():
                            if 'annotations' in inner_img_key:
                                for i in range(len(inner_img_val)):
                                    current_dict = inner_img_val[i]
                                    current_dict_keys = inner_img_val[i].keys()
                                    if 'label' not in current_dict_keys:
                                        continue
                                    current_label = current_dict['label']
                                    final_lables.append(current_label)

    # Checking which labels are missing.
    missing_label = list()
    for label in KEYPOINTS_LIST:
        if label not in final_lables:
            missing_label.append(label)
    
    extra_label = list()
    for label in final_lables:
        if label not in KEYPOINTS_LIST:
            extra_label.append(label)
    if len(extra_label) > 0:
        print('json contains extra labels : ',extra_label)
    delete_extra_labels = True
    
    # Adding empty dict for the missing labels.
    for key, val in json_data.items():
            if 'img' in key:
                for img_key, img_val in val.items():
                    if item.split(".")[0].replace("_label", "") in img_key:
                        for inner_img_key, inner_img_val in img_val.items():
                            if 'annotations' in inner_img_key:
                                
                                for label in missing_label:
                                    empty_dict = createEmptyDict(label)
                                    inner_img_val.append(empty_dict)
                                to_delete = list()
                                if delete_extra_labels is True:
                                    for i in range(len(inner_img_val)):
                                        current_dict = inner_img_val[i]
                                        current_dict_keys = inner_img_val[i].keys()
                                        if 'label' not in current_dict_keys:
                                            continue
                                        current_label = current_dict['label']
                                        if current_label in extra_label:
                                            to_delete.append(current_dict)
                                    for extra in to_delete:
                                        if extra in inner_img_val:
                                            inner_img_val.remove(extra)
    
    # Saving corrected json data in file.
    with open(corrected_file_path, 'w') as outfile:
        json.dump(json_data, outfile,indent=4)
    pass

if __name__ == "__main__":
    output_dir = "CorrectedJsons"
    if(os.path.isdir(output_dir) is False):
        os.mkdir(output_dir)
    json_path = Path('json')
    for json_file in json_path.iterdir():
        faulty_json_path = str(json_file)
        correctJson(faulty_json_path,os.path.join(output_dir,os.path.basename(faulty_json_path)))
        pass
