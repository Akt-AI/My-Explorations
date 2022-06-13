import numpy as np
import cv2
import time
import pandas as pd

def motion_vectors(img, flow, bbox, orig_next, step=1):
    x1, y1,x2, y2 = bbox
    h, w = img.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]  #Motion 2d -vectors in catesian cordinates
    mag, ang = cv2.cartToPolar(flow[:,:, 0], flow[:,:, 1], angleInDegrees=True) #Motion 2d -vectors in polar cordinates
    
    '''cv2.imwrite("fx.png", fx)
    cv2.imwrite("fy.png", fy)
    cv2.imwrite("mag.png", mag)
    cv2.imwrite("ang.png", ang)
    np.savetxt('fx.csv', fx, delimiter=',')
    np.savetxt('fy.csv', fy, delimiter=',')
    np.savetxt('ang.csv', ang, delimiter=',')
    np.savetxt('mag.csv', mag, delimiter=',')'''
    
    
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    mag1, ang1 = cv2.cartToPolar(fx, fy, angleInDegrees=True) 
    print(fx.shape, fy.shape)
    print(mag1.shape, ang1.shape)
    
    lines1 = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines2 = np.vstack([x+x1, y+y1, x+fx+x1, y+fy+y1]).T.reshape(-1, 2, 2)
    lines1 = np.int32(lines1 + 0.5)
    lines2 = np.int32(lines2 + 0.5)
    
    vis1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.polylines(vis1, lines1, 0, (0, 255, 0))
    cv2.imshow("", vis1)
    cv2.waitKey(0)
    
    vis2 = cv2.polylines(orig_next, lines2, 0, (0, 255, 0))
    cv2.imshow("", vis2)
    cv2.waitKey(0)
    
    cv2.imwrite("fx.png", fx)
    cv2.imwrite("fy.png", fy)
    cv2.imwrite("mag.png", mag)
    cv2.imwrite("ang.png", ang)
    np.savetxt('fx.csv', fx, delimiter=',')
    np.savetxt('fy.csv', fy, delimiter=',')
    np.savetxt('ang.csv', ang1, delimiter=',')
    np.savetxt('mag.csv', mag1, delimiter=',')
    
    
    with open("grids.csv", "a") as f:
        for i, j in zip(x, y):
            f.write(str(i)+ "," + str(j)+"\n") 

    with open("cords.csv", "a") as f:
        for i, j in zip(fx, fy):
            f.write(str(i)+ "," + str(j)+"\n") 
        
    with open("cords_in_polar.csv", "a") as f1:    
        for k, l in zip(mag1, ang1):
            print(k[0], l[0])
            k = k[0]
            l = l[0]
            f1.write(str(k)+ "," + str(l)+"\n")
            
    #df = pd.DataFrame({"fx": fx}, {"fy": fy})
    #print(df.head())
    
if __name__ == '__main__':
    prev = cv2.imread("/home/arun/workspace/object_tracking/Motion_Vector/motion_vectors/initial.png")
    next = cv2.imread("/home/arun/workspace/object_tracking/Motion_Vector/motion_vectors/tracked_after15_frames.png")
    
    orig_next = next.copy()
    #prev = cv2.imread("initial.png")
    bbox = cv2.selectROI(prev, True)
    x1, y1,x2, y2 = bbox
    prev = prev[y1:y2+y1, x1:x2+x1]
    #cv2.imwrite("roi.png", prev)
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
    next = next[y1:y2+y1, x1:x2+x1]
    img = next.copy()
    flow = cv2.calcOpticalFlowFarneback(prevgray,next,None,0.5,5,15,3,5,1.1,cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    motion_vectors(img, flow, bbox, orig_next)

               
