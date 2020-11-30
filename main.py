import cv2 as cv 
from matplotlib import pyplot as plt
import numpy as np
from model import load_model, get_device
import torch

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
model_name = "partConvModelCheckpoint.pt"

img = cv.imread(filename, cv.COLOR_RGB2HSV)

print("loaded image with shape:", img.shape)

print("creating edge map using canny")
img[:,:,1] = 255
img = cv.bilateralFilter(img,15,100,75)
filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
img = cv.filter2D(img,-1,filter)
edges = cv.Canny(img,100,200)

kernel_s = (5,5)
kernel = np.ones(kernel_s,np.uint8)
edges = cv.dilate(edges,kernel,iterations = 3)

print("done creating edge map")

print("creating masks")
contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

contours = np.array(contours, dtype=object)
hier = np.array(hierarchy)
len_h = int(contours.shape[0] * .5)
hier_ind = hier[:,:,2] >= 0

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

masks = []
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
    kernel = np.ones(kernel_s,np.uint8)
    mask = cv.erode(mask,kernel,iterations = 3)

    masks.append(mask)
    
images = []
for i in range(len(masks)):
    images.append(cv.imread(filename))
masks = np.array(masks).transpose((0, 3, 1, 2))
images = np.array(images).transpose((0, 3, 1, 2))
mask_tens = torch.from_numpy(masks)
image_tens = torch.from_numpy(images)
print("done creating masks")

print("loading model....")
dev = get_device()
model = load_model(model_name)
applied_images = image_tens * mask_tens
applied_images = applied_images
mask_tens = mask_tens
print("done loading model from file")

print("running model....")
total = len(applied_images)
for i in range(len(applied_images)):
    ind = i + 1
    print(F"copying image {ind}/{total} to device...")
    ap_im = applied_images[i].to(dev)
    print("copied. Running...")
    res = model(applied_images, mask_tens)
print("done running model")

print(res.shape)