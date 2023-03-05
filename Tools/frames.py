import sys
import argparse
import os
from pathlib import Path
import cv2
print(cv2.__version__)


#pathIn = "/home/arun/Desktop/cough_sneeze/data/video/S001_M_COUG_STD_FCE.avi"

def extractImages(pathIn, pathOut, count):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    #count = 0
    success = True
    while success:
      count += 1
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      if image is not None:
        #sys.exit()
        label = str(pathIn).split(".")[-2]
        label = label.split("/")[-1] 
        print(label)
        path = pathOut + os.path.sep +label+"_frame%d.jpg" % count
        print(path)
        cv2.imwrite(path, image)
        #cv2.imwrite(pathOut +os.path.sep +"frame%d.jpg" % count, image)     # save frame as JPEG file
        #count += 1
      else:
        #sys.exit() 
        continue

if __name__=="__main__":
    print("aba")
    pathOut = "out"
    """
    a = argparse.ArgumentParser()
    a.add_argument("--pathIn", help="path to video")
    a.add_argument("--pathOut", help="path to images")
    args = a.parse_args()
    print(args)
    """
    #extractImages(args.pathIn, args.pathOut)
    count = 0 
    path = Path("video")
    for filename in path.iterdir():
        count += 1
        print(count)
        filename = str(filename.absolute())
        print(filename)    
        pathIn = filename
        extractImages(pathIn, pathOut, count)
