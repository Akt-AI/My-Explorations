from pathlib import Path
import os


img_path = Path('images_val2017_format')
json_path = Path('json_val2017_format')
def main():    
    file_path = 'rmlist.txt'
    with open(file_path) as file:
        data = file.readlines()
        print(data)
    for name in data:
        json_file = name.strip('\n') + '_label.json'
        img_file = name.strip('\n') + '.png'
        print(json_file, img_file)
        
        for json in json_path.iterdir():
            if str(json) == 'json_val2017_format/'+json_file:
                os.remove(str(json))
            
        for img in img_path.iterdir():
            if str(img) == 'images_val2017_format/'+img_file:
                os.remove(str(img))
if __name__=="__main__":
    main()
    
