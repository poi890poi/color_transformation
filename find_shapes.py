import numpy as np
import cv2
import imutils


PATH_IN = './images/20200213_103455.jpg'
PATH_IN = './images/20200213_103542.jpg'
WIDTH_OUT = 640

class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        bounding = cv2.boundingRect(approx)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            x, y, w, h = bounding
            ar = w / float(h)
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape, approx, bounding


# load the image
image = cv2.imread(PATH_IN)
resized = imutils.resize(image, width=WIDTH_OUT)
src_width, src_height, src_channels = image.shape
dst_width, dst_height, *_ = resized.shape
ratio = src_width / float(dst_width)
cv2.imshow('Source', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert the resized image to grayscale, blur it slightly,
# and threshold it
lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
gray, a, b = cv2.split(lab)

# Use different thresholding to get contours to deal with the problem
# that different patches have different contrast.
CASCADES = [
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 7, 2),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 7, 2),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 4),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 4),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 17, 2),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 17, 2),
]
contours = list()
for c in CASCADES:
    method, blockSize, C, *_ = c
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, method,
                cv2.THRESH_BINARY, blockSize, C)
    #cv2.imshow('Binary', thresh)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # find contours in the thresholded image and initialize the
    # shape detector
    clist = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
        cv2.CHAIN_APPROX_TC89_KCOS)
    clist = imutils.grab_contours(clist)
    contours += clist
#cv2.drawContours(resized, contours, -1, (0, 255, 0), 3)
#cv2.imshow('Contours', resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

sd = ShapeDetector()

plist = list() # List of the bounding box of the color patch

# loop over the contours
for c in contours:
    # compute the center of the contour, then detect the name of the
    # shape using only the contour
    M = cv2.moments(c)
    if M['m00'] == 0: continue
    cX = int((M["m10"] / M["m00"]) * ratio)
    cY = int((M["m01"] / M["m00"]) * ratio)
    shape, poly, bounding = sd.detect(c)
    # multiply the contour (x, y)-coordinates by the resize ratio,
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    if shape != 'square': continue
    x, y, w, h = bounding
    area = w * h / float(dst_width) / float(dst_height)
    if area < 0.005 or area > 0.025: continue
    plist.append(bounding)
    #cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    #cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
    #    0.5, (255, 255, 255), 2)
    # show the output image
    #cv2.imshow("Image", image)

def rectIoU(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x = max(x1, x2)
    y = max(y1, y2)
    w = min(x1 + w1, x2 + w2) - x
    if w <= 0: return 0
    h = min(y1 + h1, y2 + h2) - y
    if h <= 0: return 0
    iou = w * h / float(w1 * h1 + w2 * h2 - w * h)
    return iou

for i in range(len(plist)):
    for j in range(i+1, len(plist)):
        iou = rectIoU(plist[i], plist[j])
        if iou > 0.6:
            plist[i] = (0, 0, 0, 0)
for p in plist:
    if p is not (0, 0, 0, 0):
        bbox = np.array(p, dtype=np.float) * ratio
        print(bbox.astype(np.uint))
        pt1 = tuple(bbox[:2].astype(np.uint))
        pt2 = tuple((pt1 + bbox[-2:]).astype(np.uint))
        print(pt1, pt2)
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
cv2.imshow("Image", image)

cv2.waitKey(0)
