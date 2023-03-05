import cv2


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

vidcap = cv2.VideoCapture('/home/arun/workspace/backup/video_classification/example_clips/VIDEO-2020-04-23-18-33-34.mp4')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  image = rotate(image)
  if image is not None:
      cv2.imwrite("test/frame%d.jpg" % count, image)     # save frame as JPEG file
      if cv2.waitKey(10) == 27:                     # exit if Escape is hit
          break
      count += 1
      
      
print("Done")


