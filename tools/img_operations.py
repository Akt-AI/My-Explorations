import cv2 as cv


def imgRead(path):
    img = cv.imread(path)
    return img

def imgShow(label,img):
    img_show = cv.imshow(label,img)
    cv.waitKey(0)
    return img_show
def imgSave(path,img_var):
    img_save = cv.imwrite(path,img_var)
    print("Image Has been saved Successfully !!!")
def resizeImg(img,width,height):
    resized_img = cv.resize(img,(width,height))
    print("Image has been Resized Succesfully !!!")
    return resized_img

def rotateImg180(img):
    rotated_Img = cv.rotate(img,cv.ROTATE_180)
    return rotated_Img

def rotateImg90(img):
    rotated_Img = cv.rotate(img,cv.ROTATE_90_CLOCKWISE)
    return rotated_Img

def RGBToGray(img):
    gray_img = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    print("Image has been Converted to Grayscale Successfully !!!")
    return gray_img

def grayToRGB(img):
    gray_img = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    print("Image has been Converted to RGB Successfully !!!")
    return gray_img

def cropImg(img,start_x,start_y,width,height):
    cropped_img = img[start_x:width,start_y:height]
    return cropped_img

#Reading Image
img_run = imgRead('Photos/run01.jpg')

#Displaying Original Image
imgShow("Original",img_run)

#Resizing Image and Displaying
resized_img = resizeImg(img_run,200,240)
imgShow('Resized Image',resized_img)

#Rotating Image by 180 
rotate_img = rotateImg180(img_run)
imgShow("Rotated Image",rotate_img)

#Converting RGB Image to Grayscale
gray_img = RGBToGray(img_run)
imgShow("GrayScale Image",gray_img)

#Cropping the Image
img_crop = cropImg(img_run,100,300,240,400)
imgShow('Cropped Image',img_crop)
