import cv2 as cv
#### resize dataset
def rescale(frame, scale=0.75):
    dimension = (int(frame.shape[1]*scale),int(frame.shape[0]*scale))
    return cv.resize(frame, dimension,interpolation=cv.INTER_AREA)
image = cv.imread('dataset/jambu/001.jpg')
cv.imshow('jambu', rescale(cv.imread('dataset/jambu/001.jpg')))
cv.waitKey(0)
########################################################## open array
print(cv.imread('dataset/jambu/001.jpg'))

#### crop
# cv.imshow('crop',im_crop)
# cv.waitKey(0)
# cv.destroyAllWindows()
# print(im_crop.shape)

### BGR
(b,g,r) = image [20,100]
print("blue=",b)
print("green=",g)
print("red=",r)

####################copy image####################
# cp_image = image.copy()
# print(cp_image.shape)

####################ajust image contrast####################
import numpy as np
im_adjusted = cv.addWeighted(image, 1.5,np.zeros(image.shape,image.dtype),0,-100)

cv.imshow('original image',rescale(image))
cv.imshow('adjusted image',rescale(im_adjusted))
cv.waitKey(0)
cv.destroyAllWindows()

####################detect adges####################
im_adges = cv. Canny(im_adjusted,100,200)
cv.imshow('Original image',rescale(image))
cv.imshow('Original image',rescale(im_adges))
cv.waitKey(0)
cv.destroyAllWindows()

######################################################## convert image to grayscale
im_gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale',rescale(im_gray))
cv.waitKey(0)
cv.destroyAllWindows()
#######################################################
import glob
imdir = 'dataset/jambu/'
ext = ['png','jpg','gif'] #add image formats
files = []
[files.extend(glob.glob(imdir + '*.'+e)) for e in ext]
images = [cv.imread(file) for file in files]
#adjust contrast
i = 1
for img in images :
    im_kon = cv.addWeighted(img,1.5,np.zeros(img.shape,img.dtype),0,-100)
    im_name = "dataset/jambu_kon/" + str(i) + ".png"
    cv.imwrite(im_name, im_kon)
    i+=1
    ################################################################# beruntun  Greyscale
    import glob

    imdir = 'dataset/Jambu/'
    ext = ['png', 'jpg', 'gif']  # add image formats
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    images = [cv.imread(file) for file in files]
    # adjust contrast
    i = 1
    for img in images:
        im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        im_name1 = "dataset/jambu_grey/" + str(i) + ".png"
        cv.imwrite(im_name1, im_gray)
        i += 1
    ################################################################# beruntun edge
    import glob

    imdir = 'dataset/jambu/'
    ext = ['png', 'jpg', 'gif']  # add image formats
    files = []
    [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]
    images = [cv.imread(file) for file in files]
    # adjust contrast
    i = 1
    for img in images:
        im_edges = cv.Canny(img, 100, 200)
        im_name2 = "dataset/jambu_edge/" + str(i) + ".png"
        cv.imwrite(im_name2, im_edges)
        i += 1