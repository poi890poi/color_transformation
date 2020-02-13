import cv2
import imutils


PATH_IN = './images/20200213_103455.jpg'
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
print(ratio)
cv2.imshow('Source', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# convert the resized image to grayscale, blur it slightly,
# and threshold it
lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
gray, a, b = cv2.split(lab)
blurred = cv2.GaussianBlur(gray, (9, 9), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Binary', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find contours in the thresholded image and initialize the
# shape detector
contours = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
    cv2.CHAIN_APPROX_TC89_KCOS)
contours = imutils.grab_contours(contours)
#cv2.drawContours(resized, contours, -1, (0, 255, 0), 3)
#cv2.imshow('Contours', resized)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

sd = ShapeDetector()

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
    area = w * h / float(src_width) / float(src_height)
    if area < 0.001: continue
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
    cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2)
    # show the output image
    cv2.imshow("Image", image)

cv2.waitKey(0)
