import cv2
import os
from pathlib import Path


def rotate(image):
    #img=cv2.imread("/home/arun/Desktop/cough_sneeze/frames/tosend/output_1/frame0.jpg")
    
    # rotate ccw
    out=cv2.transpose(image)
    out=cv2.flip(out,flipCode=0)

    # rotate cw
    out=cv2.transpose(image)
    image=cv2.flip(out,flipCode=1)

    #cv2.imwrite("rotated.jpg", out)
    #cv2.imshow("", img)
    #cv2.waitKey(0)
    return image

def extract_frames(filename, folder_name, output_path):
    vidcap = cv2.VideoCapture(filename)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
      success,image = vidcap.read()
      #image = rotate(image)
      if image is not None:
          cv2.imwrite(output_path + os.path.sep +folder_name + "/frame%d.jpg" % count, image)     # save frame as JPEG file
          if cv2.waitKey(10) == 27:                     # exit if Escape is hit
              break
          count += 1
          #if count==32:
              #break
              
def main():
    input_path = "input"
    output_path = "output"

    #folder_names = os.listdir("input").split(".")[0]
    path = Path("input")
    for item in path.iterdir():
        filename = str(item)
        #print(filename)
        folder_name = filename.split("/")[-1].split(".")[0]
        os.mkdir(output_path + os.path.sep + folder_name)
        extract_frames(filename, folder_name, output_path)
    print("Done")
        

if __name__=="__main__":   
    main()


