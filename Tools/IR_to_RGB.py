import cv2
import numpy as np
from pathlib import Path


def ir_to_bgr(imgfile):
	# read image
	img = cv2.imread('ir.jpg')

	# convert to gray
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# make color channels
	red = gray.copy()
	green = gray.copy()
	blue = gray.copy()

	# set weights
	R = .642
	G = .532
	B = .44

	MWIR = 4.5

	# get sum of weights and normalize them by the sum
	R = R**4
	G = G**4
	B = B**4
	sum = R + G + B
	R = R/sum
	G = G/sum
	B = B/sum
	#print(R,G,B)

	# combine channels with weights
	red = (R*red)
	green = (G*green)
	blue = (B*blue)
	result = cv2.merge([red,green,blue])

	# scale by ratio of 255/max to increase to fully dynamic range
	max=np.amax(result)
	result = ((255/max)*result).clip(0,255).astype(np.uint8)

	# write result to disk
	result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
	cv2.imwrite('out/'+ imgfile.split('/')[-1], result)
	# display it
	#cv2.imshow("RESULT", result)
	#cv2.waitKey(0)
	
ir_img_path = Path("ir_out") 
for imgfile in ir_img_path.iterdir():
	ir_to_bgr(str(imgfile))
	print(imgfile)

	
