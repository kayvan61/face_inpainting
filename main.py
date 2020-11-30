import cv2 as cv 
from matplotlib import pyplot as plt
import numpy as np
from model import load_model

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(image, lower, upper)
    # return the edged image
    return edged

filename = "IMG_0146.jpg"

img = cv.imread(filename, cv.COLOR_RGB2HSV)
#img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#img = cv.GaussianBlur(img, (3,3), 0)
#edges = auto_canny(img)

print(img.shape)

img[:,:,1] = 255
img = cv.bilateralFilter(img,15,100,75)
filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
img = cv.filter2D(img,-1,filter)
edges = cv.Canny(img,100,200)

# Apply Laplacian operator in some higher datatype
#blur = cv.GaussianBlur(img,(5,5),0)
#edges = cv.Laplacian(blur, ddepth=cv.CV_16S)
#edges = cv.convertScaleAbs(edges)

kernel_s = (5,5)
print(kernel_s)
kernel = np.ones(kernel_s,np.uint8)
edges = cv.dilate(edges,kernel,iterations = 3)

print("done")

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
#plt.show()

cv.imshow('a7',edges)
cv.waitKey(0)

contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
#final = cv.drawContours(img, contours, -1, (0,255,0), 3)

contours = np.array(contours, dtype=object)
hier = np.array(hierarchy)
print(contours.shape)
print(hier.shape)
len_h = int(contours.shape[0] * .5)
hier_ind = hier[:,:,2] >= 0
print(hier_ind.shape)

final = cv.drawContours(img, contours[:len_h], -1, (255,0,0), 3)

# create hull array for convex hull points
hull = []

# calculate points for each contour
for i in range(len(contours)):
    # creating convex hull object for each contour
    hull.append(cv.convexHull(contours[i], False))

# create an empty black image
drawing = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

# draw contours and hull points
for i in range(len(contours)):
    color_contours = (0, 255, 0) # green - color for contours
    color = (255, 0, 0) # blue - color for convex hull
    # draw ith contour
    cv.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
    # draw ith convex hull object
    cv.drawContours(drawing, hull, i, color, 1, 8)

cv.imshow('a7', final)
cv.waitKey(0)

cv.imshow('p', drawing)
cv.waitKey(0)

for i in range(len(hull)):
    original =  cv.imread(filename) 
    mask = np.ones(original.shape)

    np_h = np.array(hull[i], dtype='int32')
    fill = np_h.reshape((np_h.shape[0], np_h.shape[2]))
    fill[:,0] = fill[:,0] * original.shape[0] / final.shape[0]
    fill[:,1] = fill[:,1] * original.shape[1] / final.shape[1]
    
    mask = cv.fillConvexPoly(mask, fill, color=(0,0,0))

    mask = mask != (0,0,0)
    mask = np.where(mask, 1, 0)
    mask = mask.astype('uint8')

    kernel_s = (31,31)
    print(kernel_s)
    kernel = np.ones(kernel_s,np.uint8)
    mask = cv.erode(mask,kernel,iterations = 3)

    cv.imshow(" ", mask*255)
    cv.waitKey(0)
    cv.imshow("", cv.imread(filename) * (mask))
    cv.waitKey(0)
