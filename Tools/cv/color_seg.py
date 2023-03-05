import numpy as np
import cv2

img = cv2.imread('car.jpg')
Z = np.float32(img.reshape((-1,3)))

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 5
_,labels,centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
labels = labels.reshape((img.shape[:-1]))
reduced = np.uint8(centers)[labels]

result = [np.hstack([img, reduced])]
for i, c in enumerate(centers):
    mask = cv2.inRange(labels, i, i)
    mask = np.dstack([mask]*3) # Make it 3 channel
    ex_img = cv2.bitwise_and(img, mask)
    ex_reduced = cv2.bitwise_and(reduced, mask)
    result.append(np.hstack([ex_img, ex_reduced]))

cv2.imwrite('watermelon_out.jpg', np.vstack(result))
img = np.vstack(result)
#img = cv2.resize(img, (640, 420))
cv2.namedWindow('finalImg', cv2.WINDOW_NORMAL)
cv2.imshow("finalImg", Z)
cv2.waitKey(0)
