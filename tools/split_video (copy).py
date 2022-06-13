import cv2
import numpy as np
import os
import queue
from os.path import isfile, join


def main(fps=32):
    vidcap = cv2.VideoCapture("sneezing-loudly-in-public.mp4")
    count = 0
    folder_count = 0
    success = True
    frame_list = []
    while success:
        success,image = vidcap.read()
        if image is not None:
            height, width, layers = image.shape
            size = (width,height)
            frame_list.append(image)
            if cv2.waitKey(10) == 27:
              break
            count += 1
            if count%300==0:
                out = cv2.VideoWriter("out/" + str(folder_count) + "clip.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
                folder_count +=1
                for i in range(len(frame_list)):
                    out.write(frame_list[i])
                frame_list.clear()
                out.release()
                print(len(frame_list))

    print(len(frame_list))
if __name__=="__main__":
    main()

