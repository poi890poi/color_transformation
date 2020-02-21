import sys
import math
import operator
from pprint import pprint

import numpy as np
import cv2
import imutils
from scipy.optimize import minimize, least_squares
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans

# colormath is slower than my own implementation
#from colormath.color_objects import LabColor
#from colormath.color_diff import delta_e_cie2000


PATH_IN = './images/20200213_103455.jpg'
PATH_IN = './images/20200213_103542.jpg'
PATH_IN = './images/color_checker.jpg'
PATH_IN = './images/20200220_120820.jpg'
PATH_IN = './images/20200220_120829.jpg'
WIDTH_OUT = 800

COLOR_SPACE = 'lab'
REF_POINTS = {
    'lab': [
        (1, 'dark skin', 37.986, 13.555, 14.059),
        (2, 'light skin', 65.711, 18.13, 17.81),
        (3, 'blue sky', 49.927, -4.88, -21.925),
        (4, 'foliage', 43.139, -13.095, 21.905),
        (5, 'blue flower', 55.112, 8.844, -25.399),
        (6, 'bluish green', 70.719, -33.397, -0.199),
        (7, 'orange', 62.661, 36.067, 57.096),
        (8, 'purplish blue', 40.02, 10.41, -45.964),
        (9, 'moderate red', 51.124, 48.239, 16.248),
        (10, 'purple', 30.325, 22.976, -21.587),
        (11, 'yellow green', 72.532, -23.709, 57.255),
        (12, 'orange yellow', 71.941, 19.363, 67.857),
        (13, 'blue', 28.778, 14.179, -50.297),
        (14, 'green', 55.261, -38.342, 31.37),
        (15, 'red', 42.101, 53.378, 28.19),
        (16, 'yellow', 81.733, 4.039, 79.819),
        (17, 'magenta', 51.935, 49.986, -14.574),
        (18, 'cyan', 51.038, -28.631, -28.638),
        (19, 'white (.05*)', 96.539, -0.425, 1.186),
        (20, 'neutral 8 (.23*)', 81.257, -0.638, -0.335),
        (21, 'neutral 6.5 (.44*)', 66.766, -0.734, -0.504),
        (22, 'neutral 5 (.70*)', 50.867, -0.153, -0.27),
        (23, 'neutral 3.5 (1.05*)', 35.656, -0.421, -1.231),
        (24, 'black (1.50*)', 20.461, -0.079, -0.973),
    ],
}

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
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 5, 2),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 5, 2),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 5, 6),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 5, 6),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 7, 2),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 7, 2),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 4),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 4),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 13, 2),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 13, 2),
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 17, 2),
    (cv2.ADAPTIVE_THRESH_MEAN_C, 17, 2),
]
CASCADES = [
    (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
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
cv2.drawContours(resized, contours, -1, (0, 255, 0), 3)
cv2.imshow('Contours', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

sd = ShapeDetector()

plist = list() # List of the bounding box of the color patch
anchors = list()

# loop over the contours
image_contours = np.copy(image)
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
    if area < 0.005 or area > 0.035: continue
    #anchors.append(poly)
    anchors.append(np.mean(poly, axis=0))
    plist.append(bounding)
    cv2.drawContours(image_contours, [c], -1, (0, 255, 0), 2)
    cv2.putText(image_contours, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 2)
# show the output image
cv2.imshow("Contours", image_contours)
cv2.waitKey(0)

def rect_iou(bbox1, bbox2):
    '''
    Calculate intersection-over-union of 2 rectangles supplied in the format 
    of (x, y, w, h).
    '''
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

# Merge overlapped contours
for i in range(len(plist)):
    for j in range(i+1, len(plist)):
        iou = rect_iou(plist[i], plist[j])
        if iou > 0.6:
            plist[i] = (0, 0, 0, 0)
new_list = list()
for p in plist:
    if p is not (0, 0, 0, 0):
        new_list.append(p)
plist = new_list
del new_list

# Sort contours
plist = sorted(plist, key=operator.itemgetter(1))
new_list = list()
row_begin = 0
while True:
    row = plist[row_begin:row_begin+6]
    if not row: break
    row = sorted(row, key=operator.itemgetter(0))
    print(row)
    new_list += row
    row_begin += 6
plist = new_list
del new_list
pprint(plist)

def grids_integrity(params, vectors):
    z = 1
    tmatrix = params.reshape(3, 3)
    vectors = vectors.reshape((-1, 2))
    vectors = np.concatenate((vectors, 
        np.full((vectors.shape[0], 1), z)), axis=1)
    transformed = np.einsum('ij,kj->ik', vectors, tmatrix)
    #print(left_top, z, tmatrix, vectors, transformed, unity)
    vectors = transformed[:, 0:2]
    dev = 0.0
    x = np.sort(vectors[:, 0:1], axis=0)
    y = np.sort(vectors[:, 1:2], axis=0)
    kmx = KMeans(n_clusters=6).fit(x)
    kmy = KMeans(n_clusters=4).fit(y)
    d = kmx.inertia_ + kmy.inertia_
    #print(d)
    return d

anchors = (np.array(anchors, dtype=np.float).reshape(-1, 2)
    / thresh.shape[1])
print('anchors', anchors)
params = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float)
grids_integrity(params, anchors.flatten())
res_lsq = least_squares(grids_integrity, params, args=(
    anchors.flatten(),), jac='3-point', loss='soft_l1', tr_solver='exact')
tmatrix = res_lsq.x.reshape((3, 3))
tmatrix_inv = np.linalg.inv(tmatrix)
print(tmatrix_inv)
z = 1
anchors = np.concatenate((anchors, 
    np.full((anchors.shape[0], 1), z)), axis=1)
anchors = anchors.dot(tmatrix.T)
#grids = (grids * thresh.shape[1]).astype(np.uint)
print('grids aligned', anchors)
x = np.sort(anchors[:, 0:1], axis=0)
y = np.sort(anchors[:, 1:2], axis=0)
kmx = KMeans(n_clusters=6).fit(x)
kmy = KMeans(n_clusters=4).fit(y)
print(kmx.labels_, kmy.labels_, kmx.cluster_centers_, kmy.cluster_centers_)
grids_3d = np.ndarray(shape=(24, 3), dtype=np.float)
centersx = np.sort(kmx.cluster_centers_, axis=0).ravel()
centersy = np.sort(kmy.cluster_centers_, axis=0)
grids_3d[:, 0] = np.tile(centersx, 4)
grids_3d[:, 1] = np.tile(centersy, 6).ravel()
grids_3d[:, 2] = np.average(anchors[:, 2:3])
print(grids_3d)
plist = np.array(plist)
print(plist)
gw = int(np.average(plist[:, 2:3]) / 3)
gh = int(np.average(plist[:, 2:4]) / 3)
diag = np.array((gw, gh + 1))
print(gw, gh)
grids = (grids_3d.dot(tmatrix_inv.T) * thresh.shape[1])[:, 0:2]

image_grids = np.copy(image)
for p in np.rint(grids * ratio).astype(np.uint):
    #cv2.circle(image_grids, tuple(p), 30, (0, 255, 0), 2)
    pt1 = tuple((p - diag * ratio).astype(np.uint))
    pt2 = tuple((p + diag * ratio).astype(np.uint))
    print(pt1, pt2)
    cv2.rectangle(image_grids, pt1, pt2, (0, 255, 0), 2)
grids -= diag
grids = np.concatenate((grids, 
    np.full((grids.shape[0], 1), gw * 2)), axis=1)
grids = np.concatenate((grids, 
    np.full((grids.shape[0], 1), gh * 2)), axis=1)
plist = grids.astype(np.uint)
print(plist)
# show the output image
cv2.imshow("Grids", image_grids)
cv2.waitKey(0)

# Convert to target color space
if COLOR_SPACE == 'xyz':
    cie = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    cie_x, cie_y, cie_z = cv2.split(cie)
elif COLOR_SPACE == 'lab':
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_l, lab_a, lab_b = cv2.split(lab)
else:
    raise ValueError('Unknown color space')

def color_distance(color_a, color_b):
    l1, a1, b1 = color_a
    l2, a2, b2 = color_b
    c1 = math.sqrt(a1**2 + b1**2)
    c2 = math.sqrt(a2**2 + b2**2)
    l_delta = l2 - l1
    l_mean = (l1 + l2) / 2
    c_mean = (c1 + c2) / 2
    c_7 = c_mean**7
    c_25 = math.sqrt(c_7 / (c_7 + 25**7))
    param_a_prime = 1 + (1 - c_25) / 2
    a1_prime = a1 * param_a_prime
    a2_prime = a2 * param_a_prime
    c1_prime = math.sqrt(a1_prime**2 + b1**2)
    c2_prime = math.sqrt(a2_prime**2 + b2**2)
    c_prime_delta = c2_prime - c1_prime
    c_prime_mean = (c1_prime + c2_prime) / 2
    h1_prime = math.degrees(math.atan2(b1, a1_prime))
    h2_prime = math.degrees(math.atan2(b2, a2_prime))
    h_range = abs(h1_prime - h2_prime)
    if h_range <= 180:
        h_prime_delta = h2_prime - h1_prime
    elif h_range > 180 and h2_prime <= h1_prime:
        h_prime_delta = h2_prime - h1_prime + 360
    elif h_range > 180 and h2_prime > h1_prime:
        h_prime_delta = h2_prime - h1_prime - 360
    else:
        raise ValueError('Invalid value for H prime of color difference')
    h_delta = (2 * math.sqrt(c1_prime * c2_prime) 
        * math.sin(math.radians(h_prime_delta / 2)))
    if h_range <= 180:
        h_mean = (h1_prime + h2_prime) / 2
    elif h_range > 180 and (h1_prime + h2_prime) < 360:
        h_mean = (h1_prime + h2_prime + 360) / 2
    elif h_range > 180 and (h1_prime + h2_prime) >= 360:
        h_mean = (h1_prime + h2_prime - 360) / 2
    else:
        raise ValueError('Invalid value for H prime of color difference')
    T = (1 - 0.17 * math.cos(math.radians(h_mean - 30))
        + 0.24 * math.cos(math.radians(h_mean * 2))
        + 0.32 * math.cos(math.radians(h_mean * 3 + 6))
        - 0.20 * math.cos(math.radians(h_mean * 4 - 63)))
    L50 = (l_mean - 50)**2
    SL = 1 + 0.015 * L50 / math.sqrt(20 + L50)
    SC = 1 + 0.045 * c_mean
    SH = 1 + 0.015 * c_mean * T
    rtr = math.radians(60 * math.exp(-(((h_mean - 275) / 25)**2)))
    RT = -2 * math.sqrt(c_25) * math.sin(rtr)
    kL = 1.0
    kC = 1.0
    kH = 1.0
    component_c = c_prime_delta / kC / SC
    component_h = h_delta / kH / SH
    E00 = ((l_delta / kL / SL)**2 
        + component_c**2 + component_h**2 
        + RT * component_c * component_h)
    '''delta_l = (ref_l - l) ** 2
    delta_a = (ref_a - a) ** 2
    delta_b = (ref_b - b) ** 2
    d = math.sqrt(delta_l + delta_a + delta_b)'''
    return E00

def color_distance_sum(colors_a, colors_b):
    l1, a1, b1 = np.split(colors_a, 3, axis=1)
    l2, a2, b2 = np.split(colors_b, 3, axis=1)
    np.split(colors_b, 3, axis=1)
    c1 = np.sqrt(a1**2 + b1**2)
    c2 = np.sqrt(a2**2 + b2**2)
    l_delta = l2 - l1
    l_mean = (l1 + l2) / 2
    c_mean = (c1 + c2) / 2
    c_7 = c_mean**7
    c_25 = np.sqrt(c_7 / (c_7 + 25**7))
    param_a_prime = 1 + (1 - c_25) / 2
    a1_prime = a1 * param_a_prime
    a2_prime = a2 * param_a_prime
    c1_prime = np.sqrt(a1_prime**2 + b1**2)
    c2_prime = np.sqrt(a2_prime**2 + b2**2)
    c_prime_delta = c2_prime - c1_prime
    c_prime_mean = (c1_prime + c2_prime) / 2
    h1_prime = np.degrees(np.arctan2(b1, a1_prime))
    h2_prime = np.degrees(np.arctan2(b2, a2_prime))
    h_range = np.absolute(h1_prime - h2_prime)
    h_prime_delta = np.copy(h_range)
    h_prime_delta[h_range <= 180] = \
        h2_prime[h_range <= 180] - h1_prime[h_range <= 180]
    h_prime_delta[(h_range > 180) & (h2_prime <= h1_prime)] = (
        h2_prime[(h_range > 180) & (h2_prime <= h1_prime)] 
        - h1_prime[(h_range > 180) & (h2_prime <= h1_prime)] + 360)
    h_prime_delta[(h_range > 180) & (h2_prime > h1_prime)] = (
        h2_prime[(h_range > 180) & (h2_prime > h1_prime)] 
        - h1_prime[(h_range > 180) & (h2_prime > h1_prime)] - 360)
    h_delta = (2 * np.sqrt(c1_prime * c2_prime) 
        * np.sin(np.radians(h_prime_delta / 2)))
    h_mean = np.copy(h_range)
    h_mean[h_range <= 180] = \
        (h1_prime[h_range <= 180] + h2_prime[h_range <= 180]) / 2
    h_mean[(h_range > 180) & ((h1_prime + h2_prime) < 360)] = ((
        h1_prime[(h_range > 180) & ((h1_prime + h2_prime) < 360)] 
        - h2_prime[(h_range > 180) & ((h1_prime + h2_prime) < 360)] + 360) / 2)
    h_mean[(h_range > 180) & ((h1_prime + h2_prime) >= 360)] = ((
        h1_prime[(h_range > 180) & ((h1_prime + h2_prime) >= 360)] 
        - h2_prime[(h_range > 180) & ((h1_prime + h2_prime) >= 360)] - 360) / 2)
    T = (1 - 0.17 * np.cos(np.radians(h_mean - 30))
        + 0.24 * np.cos(np.radians(h_mean * 2))
        + 0.32 * np.cos(np.radians(h_mean * 3 + 6))
        - 0.20 * np.cos(np.radians(h_mean * 4 - 63)))
    L50 = (l_mean - 50)**2
    SL = 1 + 0.015 * L50 / np.sqrt(20 + L50)
    SC = 1 + 0.045 * c_mean
    SH = 1 + 0.015 * c_mean * T
    rtr = np.radians(60 * np.exp(-(((h_mean - 275) / 25)**2)))
    RT = -2 * np.sqrt(c_25) * np.sin(rtr)
    kL = 1.0
    kC = 1.0
    kH = 1.0
    component_c = c_prime_delta / kC / SC
    component_h = h_delta / kH / SH
    E00 = np.sqrt((l_delta / kL / SL)**2 
        + component_c**2 + component_h**2 
        + RT * component_c * component_h)
    return np.sum(E00)

def nearest_color(sample):
    d_min = sys.float_info.max
    for ref in REF_POINTS['lab']:
        index, name, ref_l, ref_a, ref_b = ref
        l, a, b = sample
        delta_l = (ref_l - l) ** 2
        delta_a = (ref_a - a) ** 2
        delta_b = (ref_b - b) ** 2
        d = math.sqrt(delta_l + delta_a + delta_b)
        #print(sample, ref, d)
        if d < d_min:
            d_min = d
            nearest = ref
    return nearest, d

samples = np.ndarray(shape=(24, 3), dtype=np.float)
ref_color_index = 0
padding = 0
for p in plist:
    bbox = np.array(p, dtype=np.float) * ratio
    x, y, w, h = bbox.astype(np.uint)
    r = (int(y+padding), int(y+h-padding*2-1), 
        int(x+padding), int(x+w-padding*2-1))
    if r[0] >= r[1] or r[2] >= r[3]:
        r = (int(y), int(y+h-1), 
            int(x), int(x+w-1))
    ''' # Use histogram instead of mean to calculate the XY values
    hist, xbins, ybins = np.histogram2d(
        cie_x[r[0]:r[1], r[2]:r[3]].ravel(),
        cie_y[r[0]:r[1], r[2]:r[3]].ravel(), [256, 256])
    major = np.unravel_index(np.argmax(hist, axis=None), hist.shape)'''

    if COLOR_SPACE == 'xyz':
        x = np.average(cie_x[r[0]:r[1], r[2]:r[3]]) / 255
        y = np.average(cie_y[r[0]:r[1], r[2]:r[3]]) / 255
    elif COLOR_SPACE == 'lab':
        x = np.average(lab_a[r[0]:r[1], r[2]:r[3]]) - 128
        y = np.average(lab_b[r[0]:r[1], r[2]:r[3]]) - 128
        z = np.average(lab_l[r[0]:r[1], r[2]:r[3]]) * 100 / 255
    else:
        raise ValueError('Unknown color space')
    cvalues = '%.2f, %.2f' % (x, y)
    samples[ref_color_index] = (z, x, y)
    print(ref_color_index, (z, x, y), samples[ref_color_index])
    #nearest, d = nearest_color((z, x, y))
    index, name, *ref_color = REF_POINTS['lab'][ref_color_index]
    d = color_distance(ref_color, (z, x, y))
    #print(x, y, cvalues)
    #print()
    # Print color values
    pt1 = tuple((bbox[:2] + [padding, padding]).astype(np.uint))
    pt2 = tuple((pt1 + bbox[-2:] - [padding*2-1, padding*2-1]).astype(np.uint))
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(image, cvalues, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)
    # Print ref color
    pt1 = tuple((bbox[:2] + [padding, padding + 16]).astype(np.uint))
    cv2.putText(image, name, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)
    # Print distance
    distance = 'd=%.4f' % (d,)
    pt1 = tuple((bbox[:2] + [padding, padding + 32]).astype(np.uint))
    cv2.putText(image, distance, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)

    ref_color_index += 1

def color_transform(tmatrix, colors_train, colors_target):
    colors_train = colors_train.reshape((24, 3))
    #matrices = matrices.reshape((24, 3, 3))
    tmatrix = tmatrix.reshape((3, 3))
    # einsum subscriptor for N vectors dot N matrices 'ij,ikj->ik'
    transformed = np.einsum('ij,kj->ik', colors_train, tmatrix)
    #diff = transformed.flatten() - REF_MATRICES['lab'].flatten()
    #diff = transformed.flatten() - colors_target
    #d = np.linalg.norm(diff)
    d = 0.0
    target = colors_target.reshape((24, 3))
    #for i in range(24):
    #    d += color_distance(transformed[i], target[i])
        #d += delta_e_cie2000(
        #    LabColor(*transformed[i]), LabColor(*target[i]))
    d = color_distance_sum(transformed, target)
    return d

# Use least square to find optimal color transformation matrix
#color_transform(samples.flatten(), 1, 0, 0, 0, 1, 0, 0, 0, 1)
ref_colors = np.array([list(ref)[2:5] for ref in REF_POINTS['lab']], dtype=np.float)
ref_colors[:, 0:1] *= 255 / 100
ref_colors[:, 1:3] += 128
samples[:, 0:1] *= 255 / 100
samples[:, 1:3] += 128
tmatrix = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float)
res_lsq = least_squares(color_transform, tmatrix, args=(
    samples.flatten(), ref_colors.flatten()), jac='3-point', loss='soft_l1',
    tr_solver='exact')
tmatrix = res_lsq.x.reshape(3, 3)
#tmatrix = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float).reshape(3, 3)
print(res_lsq)
#res = minimize(color_transform, samples.flatten(), method='Nelder-Mead', tol=1e-6)
#print(res)

cv2.imwrite('./images/output.jpg', image)
cv2.imshow("Image", image)
cv2.waitKey(0)

'''image = image.astype(np.float)
image[:, 0:1] *= 100 / 255
image[:, 1:3] -= 128'''
img_trans = np.rint(image.dot(tmatrix.T))
'''img_trans[:, 0:1] *= 255 / 100
img_trans[:, 1:3] += 128'''
cv2.imshow("Image Transformed", img_trans.astype('uint8'))
cv2.waitKey(0)
