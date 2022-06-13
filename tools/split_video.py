import cv2
import numpy as np
import os
import queue
from pathlib import Path
from os.path import isfile, join


def make_clips(input_path, output_path):
    fps = 32
    vidcap = cv2.VideoCapture(input_path)
    count = 0
    folder_count = 0
    success = True
    frame_list = []
    while success:
        _,image = vidcap.read()
        if image is not None:
            height, width, layers = image.shape
            size = (width,height)
            frame_list.append(image)
            if cv2.waitKey(10) == 27:
              break
            count += 1
            if count%200==0:
                out = cv2.VideoWriter(output_path + os.path.sep + str(folder_count) + "clip.mp4", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                folder_count +=1
                for i in range(len(frame_list)):
                    out.write(frame_list[i])
                frame_list.clear()
                out.release()
        else:
            break
    vidcap.release()            
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    video_list = Path("input_for_split")
    for video in video_list.iterdir():
        print(video)
        input_path = str(video)
        os.mkdir("output1/" + str(video).split("/")[-1].split(".")[0])
        output_path = "output1/" + str(video).split("/")[-1].split(".")[0]
        print(output_path)
        make_clips(input_path, output_path)

