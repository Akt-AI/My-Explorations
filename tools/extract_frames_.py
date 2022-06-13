import cv2
from pathlib import Path


path = Path("video")
for filename in path.iterdir():
    filename = str(filename.absolute())
    print(filename)    
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      if image is not None:
          cv2.imwrite("frames/frame%d.jpg" % count, image)     # save frame as JPEG file
          #if cv2.waitKey(10) == 27:                     # exit if Escape is hit
              #break
          count += 1
          continue

print("Done......")
