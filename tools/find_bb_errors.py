import json
from pathlib import Path
from collections import OrderedDict
import os


def find_errors_in_bb_labels(json_path):
    error_log = []
    for jsonfile in json_path.iterdir():
        print(jsonfile)
        json_label = str(jsonfile).split('.json')[0].replace("_label", "").split('/')[-1]
        
        #print(jsonfile)
        with open(str(jsonfile), "r") as read_file:
            data = json.load(read_file)
            
            
        json_label = os.path.basename(str(jsonfile))
        print(json_label)
        json_label = json_label.split('.json')[0].replace("_label", "")
        
        anno = data['img'][json_label]['annotations']

        error_dict = {}
        annotated_label_list = []
        errors = []
        error_in_file = []
        for i in anno:
            for k, v in i.items():
                if k is not None and v is not None:
                    if k=='bbox':
                        if v['parameters'][0]['val'] !=0:
                            errors.append(("val", v['parameters'][0]['val']))
                            error_in_file.append(str(jsonfile))
                        
        if len(error_in_file) > 0:   
           error_dict["ID Error in File"] = str(set(error_in_file))
           error_dict["Total No of Errors"] = len(error_in_file)
           error_dict["Errors"] = str(errors)
           error_log.append(error_dict)
           
    with open("error_log.json", 'a') as error_file:
        json.dump(error_log, error_file, indent=4)
            

if __name__ == "__main__":
    json_path = Path('input_json')
    annotated_labels = find_errors_in_bb_labels(json_path)
    print("Log generated.")
