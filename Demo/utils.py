# input is an image
# output is a list of the contours found in the image's edgemap
def getContours(image, method):
    pass

# input is an image. This uses getContours and fillContours
# returns a list of masks
def getMasks(image):
    pass
    
# input is an image. 
# returns number of faces detected
def countFaces(image):
    pass
    
# input is a list of contours and image dimensions
# returns a binary image of 1 with 0 on the inside of the contour.
def fillContours(contours, width, height):
    pass