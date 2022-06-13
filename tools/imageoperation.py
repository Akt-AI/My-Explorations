import cv2 as cv
import numpy as np

# reading an image and displaying it
image = cv.imread('/home/manoj/Desktop/AI/Deep-learning/tree.jpg')
cv.imshow("test", image)
cv.waitKey(0)
cv.destroyAllWindows()

# changing an image to gray
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

cv.waitKey(0)
cv.destroyAllWindows()

# resizing an image
resized = cv.resize(image, (500,500))
cv.imshow('Resized', resized)

cv.waitKey(0)
cv.destroyAllWindows()

# rotating an image
rotated_image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
cv.imshow('Rotated Image', rotated_image)

cv.waitKey(0)
cv.destroyAllWindows()

# #manually rotating image
# h,w,c = image.shape

# empty_img = np.zeros([h,w,c], dtype=np.uint8)

# for i in range(h):
#     for j in range(w):
#         empty_img[i,j] = image[h-i-1,w-j-1]
#         empty_img = empty_img[0:h,0:w]

# cv.imshow('Roated Image',empty_img)
# cv.waitKey(0)
# cv.destroyAllWindows()


# binary data of image/for gray only two argument
h, w, c = resized.shape

empty_img = np.zeros([h, w, c], dtype=np.uint8)

print(empty_img)


# saving an operated image
cv.imwrite('gray_version.jpg',gray)