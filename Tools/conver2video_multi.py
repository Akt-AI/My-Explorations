import cv2
import numpy as np
import os
from pathlib import Path 
from os.path import isfile, join
 
 
def convert_frames_to_video(pathIn, pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
 
    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        #print(filename)
        img = cv2.imread(filename)
        #print(img)
        #img = cv2.resize(img, (368,368))
        img = cv2.resize(img, (368,368))
        height, width, layers = img.shape
        size = (width,height)
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
 
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()
 
def main():
    #pathIn= 'outputs/'
    pathIn = Path("outputs")
    for path in pathIn.iterdir():
        path = str(path) 
        #print(path)
        pathOut = path + ".mp4"
        fps = 30.0
        path = path+os.path.sep
        print(pathOut)
        convert_frames_to_video(path, pathOut, fps)
 
if __name__=="__main__":
    main()

