import cv2
import time
import os

def convert_to_ir_to_gray_video(filename):
	cap = cv2.VideoCapture(filename)
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	output_file = 'out/'+ filename.split('/')[-1] +'_gray.avi'
	print(output_file)
	out = cv2.VideoWriter(output_file ,fourcc, 30.0, (368,368), isColor=False)
	while True:    
		ret, frame = cap.read()
		#time.sleep(0.1)
		
		if frame is not None:
		    frame = cv2.resize(frame, (368,368))
		    #cv2.imshow('frame1',frame)
		    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		    out.write(frame)
		    cv2.imwrite('img.jpg',frame)
		    #cv2.imshow('frame',frame)
		    if cv2.waitKey(1) & 0xFF == ord('q'):
		        break
		if frame is None:
		    break
	cap.release()
	out.release()
	cv2.destroyAllWindows()
	
if __name__=="__main__":
	filename = 'input/IR.mp4'
	convert_to_ir_to_gray_video(filename)
	print("Done.....")
   
