from pathlib import Path
import os


path = Path('images_val2017_format')
rmlist = [579,
			582,
			637,
			610,
			592,
			622,
			494,
			495,
			504,
			367,
			314,
			315]
    
for item in path.iterdir():
    item = str(item).split('/')[-1]
    image_id = item.split('_')[-1]
    image_id_new = int(image_id[0:image_id.rfind('.')])
    #print(image_id_new)
    
    if image_id_new in rmlist:
    	#print(image_id_new)
    	if str(image_id_new) in str(item):
    		print(str(item))
    		#os.remove(str(item))
