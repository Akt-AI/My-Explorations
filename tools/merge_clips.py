import cv2
from pathlib import Path
import numpy as np


path = Path("input_for_merge")
fps =32
resolution = (640, 360)
# Create a new video
video = cv2.VideoWriter("merged.mp4", cv2.VideoWriter_fourcc(*"MPEG"), fps, resolution)

# Write all the frames sequentially to the new video
for filename in path.iterdir():
    curr_v = cv2.VideoCapture(str(filename))
    while curr_v.isOpened():
        r, frame = curr_v.read()    # Get return value and curr frame of curr video
        if frame is not None:
            frame = cv2.resize(frame, resolution)
            #img = np.zeros([frame.shape[0],frame.shape[1],3],dtype=np.uint8)
            #img.fill(0) # or img[:] = 255
            #frame1 = img + frame
            cv2.imshow("", frame)
            cv2.waitKey(1)
            video.write(frame)
        else:
            break
        
video.release()

