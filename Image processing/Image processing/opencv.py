import cv2 as cv
######################################################### resize dataset
def rescale(frame, scale=0.5):
    dimension = (int(frame.shape[1]*scale),int(frame.shape[0]*scale))
    return cv.resize(frame, dimension,interpolation=cv.INTER_AREA)
image = cv.imread('dataset/jambu_biji/003.jpg')
cv.imshow('jambu_biji', rescale(cv.imread('dataset/jambu_biji/003.jpg')))
cv.waitKey(0)
########################################################## open image berupa array
print(cv.imread('dataset/jambu_biji/003.jpg'))

########################################################## crop
# im_crop = image[300:600,300:600]
# cv.imshow('crop',im_crop)
# cv.waitKey(0)
# cv.destroyAllWindows()
# print(im_crop.shape)

########################################################## BGR
# (b,g,r) = image[20,100]
# print("blue=",b)
# print("green=",g)
# print("red=",r)

########################################################### copy image
# cp_image = image.copy()
# print(cp_image.shape)

########################################################## adjust image contrast
# import numpy as np
# im_adjusted = cv.addWeighted(image, 2.5,np.zeros(image.shape,image.dtype),0,-100)
#
# cv.imshow('original image',rescale(image))
# cv.imshow('adjusted image',rescale(im_adjusted))
# cv.waitKey(0)
# cv.destroyAllWindows()

# ######################################################### detect edges
# im_edges = cv.Canny(image,100,200)
# cv.imshow('Original image',rescale(image))
# cv.imshow('Original image',rescale(im_edges))
# cv.waitKey(0)
# cv.destroyAllWindows()
######################################################## convert image to grayscale
# im_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
# cv.imshow('Grayscale',rescale(im_gray))
# cv.waitKey(0)
# cv.destroyAllWindows()
####################################################### beruntun kontras
import glob
import numpy as np
imdir = 'dataset/jambu_biji/'
ext = ['png','jpg','gif'] #add image formats
files = []
[files.extend(glob.glob(imdir + '*.'+e)) for e in ext]
images = [cv.imread(file) for file in files]
#adjust contrast
i = 1
for img in images :
    im_kon = cv.addWeighted(img,1.5,np.zeros(img.shape,img.dtype),0,-100)
    im_name = "dataset/jambu_biji_kontras/" + str(i) + ".png"
    cv.imwrite(im_name, im_kon)
    i+=1
################################################################# beruntun  Greyscale
import glob
imdir = 'dataset/jambu_biji/'
ext = ['png','jpg','gif'] #add image formats
files = []
[files.extend(glob.glob(imdir + '*.'+e)) for e in ext]
images = [cv.imread(file) for file in files]
#adjust contrast
i = 1
for img in images :
    im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    im_name1 = "dataset/jambu_biji_grey/" + str(i) + ".png"
    cv.imwrite(im_name1, im_gray)
    i+=1
################################################################# beruntun edge
import glob
imdir = 'dataset/jambu_biji/'
ext = ['png','jpg','gif'] #add image formats
files = []
[files.extend(glob.glob(imdir + '*.'+e)) for e in ext]
images = [cv.imread(file) for file in files]
#adjust contrast
i = 1
for img in images :
    im_edges = cv.Canny(img,100,200)
    im_name2 = "dataset/jambu_biji_edge/" + str(i) + ".png"
    cv.imwrite(im_name2, im_edges)
    i+=1
