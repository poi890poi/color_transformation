import sys
import argparse
import math
import operator
from pprint import pprint

import numpy as np
import cv2
import imutils
from scipy.optimize import minimize, least_squares
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans

# colormath is slower than my own implementation
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000


PATH_IN = './images/20200213_103455.jpg'
PATH_IN = './images/20200213_103542.jpg'
#PATH_IN = './images/color_checker.jpg'
#PATH_IN = './images/20200220_120829.jpg'
#PATH_IN = './images/20200220_120820.jpg'
#PATH_IN = './images/20200303_114801.jpg'
#PATH_IN = './images/20200303_114806.jpg'
#PATH_IN = './images/20200303_114812.jpg' # poor
#PATH_IN = './images/20200303_114815.jpg' # Need to fix Z
PATH_IN = './images/20200303_114817.jpg' # Need to fix Z


PI = math.pi
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

parser = argparse.ArgumentParser(description='Auto color correction.')
parser.add_argument('--debug', action='store_true',
                    help='Display images for debug.')
args = parser.parse_args()
DEBUG = args.debug
lsq_verbose = 1
if DEBUG:
    lsq_verbose = 2


class ImageProcessing:
    
    DISPLAY_WIDTH = 1600
    WORKING_WIDTH = 800

    SHAPE_UNIDENTIFIED = 0
    SHAPE_TRIANGLE = 1
    SHAPE_SQUARE = 2
    SHAPE_QUADRILATERAL = 3

    def __init__(self, filename):
        '''
        Open source image from file.
        '''
        source = cv2.imread(filename)
        source_height, source_width, *_ = source.shape
        display = imutils.resize(source, width=self.DISPLAY_WIDTH)
        display_height, display_width, *_ = display.shape
        self.__source_width = source_width
        self.__display_width = display_width
        self.__img_source = display

    def detect_shape(self, c):
        # initialize the shape name and approximate the contour
        shape = self.SHAPE_UNIDENTIFIED
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        bounding = cv2.boundingRect(approx)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = self.SHAPE_TRIANGLE
        elif cv2.isContourConvex(approx):
            # if the shape has 4 vertices, it is either a square or
            # a rectangle
            if len(approx) == 4:
                # compute the bounding box of the contour and use the
                # bounding box to compute the aspect ratio
                polygon = np.array(approx[:, 0], dtype=np.float)
                polygon = np.vstack((polygon, polygon[0]))
                sides = np.linalg.norm(np.diff(polygon, axis=0), axis=1)
                sides *= self.__length_factor
                print(sides)
                sides_dev = np.std(sides)
                print('sides_dev={}'.format(sides_dev))
                shape = (self.SHAPE_SQUARE 
                    if sides_dev < 0.008 else self.SHAPE_QUADRILATERAL)
        # return the name of the shape
        return shape, approx, bounding

    def normalize_source(self):
        '''
        Resize the source image to a normalized dimension so parameters for 
        future algorithms can be fixed. This is crucial for filters with 
        kernel matrices that are size-sentitive.
        '''
        source = self.__img_source
        resized = imutils.resize(source, width=self.WORKING_WIDTH)
        src_height, src_width, src_channels = source.shape
        resized_height, resized_width, *_ = resized.shape
        self.__source_factor = src_width / resized_width
        self.__working_width = resized_width
        self.__length_factor = 1 / resized_width
        self.__area_factor = self.__length_factor ** 2
        self.__img_resized = resized
        cv2.imshow('Source', resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @classmethod
    def display(cls, caption, image):
        cv2.imshow(caption, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def render_contours(self, image, contours):
        for c in contours:
            M = cv2.moments(c)
            if M['m00'] == 0: continue
            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    def get_contours(self):
        '''
        1. Convert image to gray scale.
        2. Apply threshold to get binary image.
        3. Find contours.
        '''
        lab = cv2.cvtColor(self.__img_resized, cv2.COLOR_BGR2LAB)
        gray, a, b = cv2.split(lab)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        if DEBUG: self.display('Blurred', blurred)
        method, blockSize, C, *_ = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 7, 2)
        binary = cv2.adaptiveThreshold(blurred, 255, method,
                    cv2.THRESH_BINARY, blockSize, C)
        if DEBUG: self.display('Binary', binary)

        # Apply erosion to remove noises connected to the square.
        kernel = np.ones((9,9),np.uint8)
        binary = cv2.erode(binary, kernel)
        if DEBUG: self.display('Eroded', binary)

        contours = cv2.findContours(binary.copy(), cv2.RETR_LIST,
            cv2.CHAIN_APPROX_TC89_KCOS)
        contours = imutils.grab_contours(contours)
        if DEBUG:
            img_ = self.__img_resized.copy()
            self.render_contours(img_, contours)
            self.display('Contours', img_)
        self.__contours = contours
        return (contours, lab, binary)

    def approximate_contours(self):
        img_ = self.__img_resized.copy()
        centers = list()
        polygons = list()
        for c in self.__contours:
            M = cv2.moments(c)
            if M['m00'] == 0: continue
            shape, approx, bounding = self.detect_shape(c)
            if shape != self.SHAPE_SQUARE: continue
            A = cv2.contourArea(approx) * self.__area_factor
            print('area', A)
            if A > 0.016 or A < 0.001: continue
            polygons.append(approx)
            M = cv2.moments(approx)
            center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            centers.append(center)
            approx = approx.astype("int")
            if DEBUG:
                cv2.circle(img_, tuple(np.rint(center).astype(np.uint)), 
                    5, (255, 255, 0), -1)
                cv2.drawContours(img_, [approx], -1, (0, 255, 0), 2)
        self.display('Squares', img_)
        self.__shapes = np.array(polygons, dtype=np.float).reshape((-1, 4, 2))
        self.__shape_centers = np.array(centers, dtype=np.float)

    @classmethod
    def render_vectors(cls, image, vectors):
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0))
        for v in vectors.astype(np.uint):
            label, vx, vy, cx, cy = v
            cv2.arrowedLine(image, (cx, cy), (cx+vx, cy+vy), colors[label], 
                2, tipLength=0.2)

    def align_grids(self, columns, rows):
        N = columns * 2 + rows * 2
        N = 4
        vertices = self.shapes.reshape(-1, 8)
        vectors = np.concatenate((vertices, vertices[:, 0:2]), axis=1)
        vectors = vectors.reshape(-1, 5, 2)
        vectors = np.diff(vectors, axis=1)
        vectors = vectors.reshape(-1, 2)
        vertices = vertices.reshape(-1, 2)
        directions = np.arctan2(vectors[:, 0], vectors[:, 1])
        pprint(directions)
        #kernel = gaussian_kde(directions)
        #print(kernel.d, kernel.n, kernel.neff, kernel.factor, kernel.covariance)
        directions = directions.reshape(-1, 1)
        print('directions', directions[:, 0])
        km = KMeans(n_clusters=N).fit(np.concatenate((directions, directions), axis=1))
        print('clustered directions', km.cluster_centers_[:, 0])
        print('labels', km.labels_)
        vectors_clustered = np.ndarray((km.cluster_centers_.shape[0], 2))
        vectors_clustered[:, 0] = np.cos(km.cluster_centers_[:, 0])
        vectors_clustered[:, 1] = np.sin(km.cluster_centers_[:, 0])
        vectors = np.concatenate((km.labels_.reshape(-1, 1), vectors, vertices), axis=1)
        '''
        Output grid vectors are in the format of 
            [[label, vx, vy, cx, cy],... ],
        where (cx, cy) is the starting point and (vx, vy) is the vector.
        '''
        #vectors = np.sort(vectors, axis=0)
        directions = directions.ravel()
        pprint(vectors)
        print(directions.shape, vectors.shape)

        down = vectors[(directions <= PI / 4) & (directions > -PI / 4)]
        up = vectors[(directions <= -PI * 3/4) | (directions > PI *3/4)]
        left = vectors[(directions <= -PI / 4) & (directions > -PI * 3/4)]
        right = vectors[(directions <= PI * 3/4) & (directions > PI / 4)]

        '''names = 'label, vx, vy, cx, cy'
        formats = 'u4, f8, f8, f8, f8'
        down = np.core.records.fromarrays(down.transpose(), 
            names=names, formats=formats)
        up = np.core.records.fromarrays(up.transpose(), 
            names=names, formats=formats)
        left = np.core.records.fromarrays(left.transpose(), 
            names=names, formats=formats)
        right = np.core.records.fromarrays(right.transpose(), 
            names=names, formats=formats)
        np.sort(down, axis=0, order='cx')
        np.argsort(up, axis=0, order='cx')
        np.sort(left, axis=0, order='cy')
        np.argsort(right, axis=0, order='cy')'''

        if DEBUG:
            img_ = self.__img_resized.copy()
            self.render_vectors(img_, vectors)
            self.display('Vectors', img_)

        OR_VERTICAL = 0
        OR_HORIZONTAL = 1

        def fit_line(vectors, group_n, orientation):
            s_lines = list()
            group_index = 3 if orientation==OR_VERTICAL else 4
            c_ = vectors[:, group_index].reshape(-1, 1)
            km = KMeans(n_clusters=group_n).fit(np.concatenate((c_, c_), axis=1))
            for c in range(group_n):
                d_ = vectors[km.labels_==c]
                x = np.concatenate((d_[:, 3], d_[:, 3] + d_[:, 1]), axis=0)
                y = np.concatenate((d_[:, 4], d_[:, 4] + d_[:, 2]), axis=0)
                p_ = np.polyfit(x, y, 1)
                s_lines.append((p_, x, y))
                print(np.poly1d(p_), p_)
            return s_lines
            
        p_down = fit_line(down, columns, OR_VERTICAL)
        p_up = fit_line(up, columns, OR_VERTICAL)
        p_left = fit_line(left, rows, OR_HORIZONTAL)
        p_right = fit_line(right, rows, OR_HORIZONTAL)
        assert (len(p_down)==columns and len(p_up)==columns 
            and len(p_left)==rows and len(p_right)==rows), 'Grids lines mismatch'

        def get_s_line(points):
            '''
            Use points to find a straight line as polynomial function.

            Input vectors are in the format of 
                [[label, vx, vy, cx, cy],... ],
            where (cx, cy) is the starting point and (vx, vy) is the vector.
            '''
            d_ = points.reshape(1, -1)
            x = np.concatenate((d_[:, 3], d_[:, 3] + d_[:, 1]), axis=0)
            y = np.concatenate((d_[:, 4], d_[:, 4] + d_[:, 2]), axis=0)
            p_ = np.polyfit(x, y, 1)
            return p_

        vertical = np.vstack((down, up))
        s_left = get_s_line(vertical[np.argmin(vertical[:, 3])])
        s_right = get_s_line(vertical[np.argmax(vertical[:, 3])])
        horizontal = np.vstack((left, right))
        s_top = get_s_line(horizontal[np.argmin(horizontal[:, 4])])
        s_bottom = get_s_line(horizontal[np.argmax(horizontal[:, 4])])

        def s_line_intersect(p1, p2):
            '''
            Take polynomial coefficients of 2 straight lines and find the 
            intersection.
            '''
            a1, b1 = p1
            a2, b2 = p2
            x = -(b1 - b2) / (a1 - a2)
            y = a1 * x + b1
            return np.array((x, y))

        pt1 = s_line_intersect(s_left, s_top)
        pt2 = s_line_intersect(s_right, s_top)
        pt3 = s_line_intersect(s_right, s_bottom)
        pt4 = s_line_intersect(s_left, s_bottom)
        corners = np.vstack((pt1, pt2, pt3, pt4))

        def draw_straight_line(image, poly):
            a, b = poly
            h, w, *_ = image.shape
            x = 0
            pt1 = (x, int(a * x + b))
            x = w - 1
            pt2 = (x, int(a * x + b))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
            y = 0
            pt1 = (int((y - b) / a), y)
            y = h - 1
            pt2 = (int((y - b) / a), y)
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        if DEBUG:
            img_ = self.__img_resized.copy()
            draw_straight_line(img_, s_left)
            draw_straight_line(img_, s_right)
            draw_straight_line(img_, s_top)
            draw_straight_line(img_, s_bottom)
            for c in corners:
                cv2.circle(img_, tuple(c.astype(np.uint)), 8, (255, 255, 0), 2)
            self.display('Grids', img_)

        '''params = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float)
        #grids_integrity(params, anchors.flatten())
        res_lsq = least_squares(grids_integrity, params, args=(
            anchors.flatten(),), jac='3-point', loss='soft_l1', tr_solver='exact',
            verbose=lsq_verbose)'''

        raise

        if DEBUG:
            img_ = self.__img_resized.copy()
            h, w, *_ = img_.shape
            '''for p, sx, sy in p_down:
                for i in range(len(sx)):
                    cv2.circle(img_, (int(sx[i]), int(sy[i])), 4, (255, 255, 0), 2)
                a, b = p
                y = 0
                pt1 = (int((y - b) / a), y)
                y = h - 1
                pt2 = (int((y - b) / a), y)
                cv2.line(img_, pt1, pt2, (0, 255, 0), 2)
            for p, sx, sy in p_up:
                for i in range(len(sx)):
                    cv2.circle(img_, (int(sx[i]), int(sy[i])), 4, (255, 255, 0), 2)
                a, b = p
                y = 0
                pt1 = (int((y - b) / a), y)
                y = h - 1
                pt2 = (int((y - b) / a), y)
                cv2.line(img_, pt1, pt2, (0, 255, 0), 2)'''
            for p, sx, sy in p_left[0:1]:
                for i in range(len(sx)):
                    cv2.circle(img_, (int(sx[i]), int(sy[i])), 4, (255, 255, 0), 2)
                a, b = p
                x = 0
                pt1 = (x, int(a * x + b))
                x = w - 1
                pt2 = (x, int(a * x + b))
                cv2.line(img_, pt1, pt2, (0, 255, 0), 2)
            '''for p, sx, sy in p_right:
                for i in range(len(sx)):
                    cv2.circle(img_, (int(sx[i]), int(sy[i])), 4, (255, 255, 0), 2)
                a, b = p
                x = 0
                pt1 = (x, int(a * x + b))
                x = w - 1
                pt2 = (x, int(a * x + b))
                cv2.line(img_, pt1, pt2, (0, 255, 0), 2)'''
            self.display('Grids', img_)

        #print('clustered directions', km.cluster_centers_[:, 0])
        #print('labels', km.labels_)
        raise

        grids = np.ndarray((N, 4), dtype=np.float)
        for i in range(N):
            grids[i] = np.mean(vectors[vectors[:, 0]==i], axis=0)[1:]
            print(i, grids[i])

        def vanishing_points(params, grids):
            grids = np.copy(grids.reshape(-1, 4))
            centers = grids[:, :2].reshape(-1, 2)
            vectors = grids[:, 2:4].reshape(-1, 2)
            vectors[:, 0] *= params
            vectors[:, 1] *= params
            vanishing_points = centers + vectors
            km = KMeans(n_clusters=2).fit(vanishing_points)
            print(km.inertia_, vanishing_points)
            return km.inertia_

        params = np.ones(shape=(N,), dtype=np.float)
        res_lsq = least_squares(vanishing_points, params, args=(
            grids.flatten(),), jac='3-point', loss='soft_l1', 
            tr_solver='exact', max_nfev=9000, xtol=None, verbose=1)
        print(res_lsq.x)
        vanishing_points(res_lsq.x, grids)

        raise

    @property
    def shapes(self):
        return self.__shapes

    @property
    def shape_centers(self):
        return self.__shape_centers

    @property
    def length_factor(self):
        return self.__length_factor

    @property
    def working_width(self):
        return self.__working_width

    @property
    def display_width(self):
        return self.__display_width

    @property
    def source_width(self):
        return self.__source_width

    @property
    def area_factor(self):
        return self.__area_factor

    @property
    def image(self):
        return self.__img_source

# load the source image
imgp = ImageProcessing(PATH_IN)
imgp.normalize_source()

# convert the resized image to grayscale, blur it slightly,
# and threshold it
imgp.get_contours()

imgp.approximate_contours()
imgp.align_grids(6, 4)

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

anchors = imgp.shape_centers.reshape(-1, 2) * imgp.length_factor
print('anchors', anchors)

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

rows = 4
columns = 6
params = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float)
#grids_integrity(params, anchors.flatten())
res_lsq = least_squares(grids_integrity, params, args=(
    anchors.flatten(),), jac='3-point', loss='soft_l1', tr_solver='exact',
    verbose=lsq_verbose)
tmatrix = res_lsq.x.reshape((3, 3))
#tmatrix = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float).reshape((3, 3))
tmatrix_inv = np.linalg.inv(tmatrix)
print('tmatrix_inv', tmatrix_inv)
z = 1
plist = imgp.shapes.reshape(-1, 2) * imgp.length_factor
plist = np.concatenate((plist, 
    np.full((plist.shape[0], 1), z)), axis=1)
plist = plist.dot(tmatrix.T)
#grids = (grids * thresh.shape[1]).astype(np.uint)
print('grids aligned', plist)
x = np.sort(plist[:, 0:1], axis=0)
y = np.sort(plist[:, 1:2], axis=0)
#kmx = KMeans(n_clusters=6).fit(x)
#kmy = KMeans(n_clusters=4).fit(y)
kmx = KMeans(n_clusters=columns*2).fit(x)
kmy = KMeans(n_clusters=rows*2).fit(y)
x_coords = np.sort(kmx.cluster_centers_, axis=0).reshape((-1, 2))
y_coords = np.sort(kmy.cluster_centers_, axis=0).reshape((-1, 2))
print('KMeans', y_coords, x_coords)

'''grids_3d = np.ndarray(shape=(24, 3), dtype=np.float)
centersx = np.sort(kmx.cluster_centers_, axis=0).ravel()
centersy = np.sort(kmy.cluster_centers_, axis=0)
grids_3d[:, 0] = np.tile(centersx, 4)
grids_3d[:, 1] = np.tile(centersy, 6).ravel()
grids_3d[:, 2] = np.average(anchors[:, 2:3])'''
grids_3d = np.ndarray(shape=(columns*rows, 4, 3), dtype=np.float)
x_coords = np.repeat(np.tile(x_coords, 2).reshape(1, -1, 2), rows, axis=0)
y_coords = np.tile(np.repeat(y_coords, 2).reshape(1, -1, 4), columns)
grids_3d[:, :, 0] = x_coords.reshape(-1, 4)
grids_3d[:, :, 1] = y_coords.reshape(-1, 4)
grids_3d[:, :, 2] = np.average(plist[:, 2:3])
'''plist = np.array(plist)
gw = int(np.average(plist[:, 2:3]) / 3)
gh = int(np.average(plist[:, 2:4]) / 3)
diag = np.array((gw, gh + 1))
print(gw, gh)'''
print('grids_3d', grids_3d)
grids = (grids_3d.reshape(-1, 3).dot(tmatrix_inv.T))[:, 0:2]
grids = grids.reshape(-1, 4, 2)
print('grids', grids)

color_patches = list()
image_grids = np.copy(imgp.image)
image_mask = np.zeros(imgp.image.shape, np.uint8)
for p in np.rint(grids * imgp.display_width).astype(np.uint):
    p = p.reshape(-1, 1, 2)
    _ = p[2].copy()
    p[2] = p[3].copy()
    p[3] = _
    color_patches.append(p)
    cv2.drawContours(image_grids, [p], -1, (0, 255, 0), 2)
    cv2.fillConvexPoly(image_mask, p, (255, 255, 255))
    '''#cv2.circle(image_grids, tuple(p), 30, (0, 255, 0), 2)
    pt1 = tuple((p - diag * imgp.length_factor).astype(np.uint))
    pt2 = tuple((p + diag * imgp.length_factor).astype(np.uint))
    print(pt1, pt2)
    cv2.rectangle(image_grids, pt1, pt2, (0, 255, 0), 2)'''
'''grids -= diag
grids = np.concatenate((grids, 
    np.full((grids.shape[0], 1), gw * 2)), axis=1)
grids = np.concatenate((grids, 
    np.full((grids.shape[0], 1), gh * 2)), axis=1)
plist = grids.astype(np.uint)
print(plist)'''
# show the output image
cv2.imshow("Grids", image_grids)
cv2.waitKey(0)
cv2.imshow("Mask", image_mask)
cv2.waitKey(0)

def color_distance_(color_a_, color_b_):
    l1, a1, b1 = color_a_
    l2, a2, b2 = color_b_
    color_a = LabColor(l1/100, a1, b1)
    color_b = LabColor(l2/100, a2, b2)
    return delta_e_cie2000(color_a, color_b)

def color_distance(color_a_, color_b_):
    l1, a1, b1 = color_a_
    l2, a2, b2 = color_b_
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
    E00 = np.sqrt((l_delta / kL / SL)**2 
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
    E00 = ((l_delta / kL / SL)**2 
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

# Convert to target color space
image = np.copy(imgp.image)
if COLOR_SPACE == 'xyz':
    cie = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
    cie_x, cie_y, cie_z = cv2.split(cie)
elif COLOR_SPACE == 'lab':
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_l, lab_a, lab_b = cv2.split(lab)
else:
    raise ValueError('Unknown color space')

padding = 0
image = np.copy(imgp.image)
samples = np.ndarray(shape=(24, 3), dtype=np.float)
ref_color_index = 0
for p in color_patches:
    image_mask = np.zeros(imgp.image.shape[:2], np.uint8)
    cv2.fillConvexPoly(image_mask, p, (255,))
    ''' # Use histogram instead of mean to calculate the XY values
    hist, xbins, ybins = np.histogram2d(
        cie_x[r[0]:r[1], r[2]:r[3]].ravel(),
        cie_y[r[0]:r[1], r[2]:r[3]].ravel(), [256, 256])
    major = np.unravel_index(np.argmax(hist, axis=None), hist.shape)'''

    if COLOR_SPACE == 'xyz':
        x = np.average(cie_x[r[0]:r[1], r[2]:r[3]]) / 255
        y = np.average(cie_y[r[0]:r[1], r[2]:r[3]]) / 255
    elif COLOR_SPACE == 'lab':
        color_mean = cv2.mean(lab, image_mask)
        print('mean', lab.shape, lab[20][20], color_mean)
        z = color_mean[0] * 100 / 255
        x = color_mean[1] - 128
        y = color_mean[2] - 128
    else:
        raise ValueError('Unknown color space')
    cvalues = '%d, %.2f, %.2f' % (z, x, y)
    samples[ref_color_index] = (z, x, y)
    print(ref_color_index, (z, x, y), samples[ref_color_index])
    #nearest, d = nearest_color((z, x, y))
    index, name, *ref_color = REF_POINTS['lab'][ref_color_index]
    d = color_distance(ref_color, (z, x, y))
    #print(x, y, cvalues)
    #print()
    # Print color values
    cv2.drawContours(image_grids, [p], -1, (0, 255, 0), 2)
    pt1 = tuple((p[0].ravel() + [padding, padding + 16]).astype(np.uint))
    cv2.putText(image, cvalues, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)
    # Print ref color
    pt1 = tuple((p[0].ravel() + [padding, padding + 32]).astype(np.uint))
    cv2.putText(image, name, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)
    # Print reference color values
    ref_values = '%.2f, %.2f' % (ref_color[1], ref_color[2])
    pt1 = tuple((p[0].ravel() + [padding, padding + 48]).astype(np.uint))
    cv2.putText(image, ref_values, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)
    # Print distance
    distance = 'd=%.4f' % (d,)
    pt1 = tuple((p[0].ravel() + [padding, padding + 64]).astype(np.uint))
    cv2.putText(image, distance, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)

    ref_color_index += 1

def color_transform(tmatrix, colors_train, colors_target):
    colors_train = colors_train.reshape((24, 3))
    #colors_train = np.concatenate((colors_train, 
    #    np.full((colors_train.shape[0], 1), 1)), axis=1)
    tmatrix = tmatrix.reshape((3, 3))
    # einsum subscriptor for N vectors dot N matrices 'ij,ikj->ik'
    transformed = np.einsum('ij,kj->ik', colors_train, tmatrix)
    #diff = transformed.flatten() - REF_MATRICES['lab'].flatten()
    #diff = transformed.flatten() - colors_target
    #d = np.linalg.norm(diff)
    d = 0.0
    d_check = 0.0
    transformed = transformed[:, 0:3]
    target = colors_target.reshape((24, 3))
    #for i in range(24):
    #    d += color_distance(transformed[i], target[i])
        #d += delta_e_cie2000(
        #    LabColor(*transformed[i]), LabColor(*target[i]))
    d = color_distance_sum(transformed, target)
    '''for i in range(target.shape[0]):
        d_check += color_distance(transformed[i], target[i])
        my_d = color_distance(transformed[i], target[i])
        color_a = LabColor(transformed[i][0]/100, transformed[i][1], transformed[i][2])
        color_b = LabColor(target[i][0]/100, target[i][1], target[i][2])
        color_math_d = delta_e_cie2000(color_a, color_b)
        print('mine vs color_math', my_d, color_math_d)
    print('color_distance_sum', d, d_check)'''
    print('color_distance_sum', d)
    return d
    d = 0.0
    for i in range(target.shape[0]):
        color_a = LabColor(transformed[i][0]/100, transformed[i][1], transformed[i][2])
        color_b = LabColor(target[i][0]/100, target[i][1], target[i][2])
        d += delta_e_cie2000(color_a, color_b)
    return d

# Use least square to find optimal color transformation matrix
#color_transform(samples.flatten(), 1, 0, 0, 0, 1, 0, 0, 0, 1)
ref_colors = np.array([list(ref)[2:5] for ref in REF_POINTS['lab']], dtype=np.float)
'''ref_colors[:, 0:1] *= 255 / 100
ref_colors[:, 1:3] += 128
samples[:, 0:1] *= 255 / 100
samples[:, 1:3] += 128'''

'''print('ref_colors', ref_colors)
print('samples', samples)
ref_colors[:, 0:1] *= 255 / 100 
ref_colors[:, 0:1] -= 128
ref_colors[:, :] /= 128
samples[:, 0:1] *= 255 / 100 
samples[:, 0:1] -= 128
samples[:, :] /= 128
print('ref_colors', ref_colors)
print('samples', samples)'''

tmatrix = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float)
#tmatrix = np.array((1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1), dtype=np.float)
res_lsq = least_squares(color_transform, tmatrix, args=(
    samples.flatten(), ref_colors.flatten()), jac='3-point', loss='soft_l1',
    tr_solver='exact', ftol=None, xtol=1e-12, gtol=None, max_nfev=9000,
    verbose=lsq_verbose)
tmatrix = res_lsq.x.reshape(3, 3)
#tmatrix = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float).reshape(3, 3)
print(res_lsq)
#raise
#res = minimize(color_transform, samples.flatten(), method='Nelder-Mead', tol=1e-6)
#print(res)

cv2.imwrite('./images/output.jpg', image)
cv2.imshow("Image", image)
cv2.waitKey(0)

'''image = image.astype(np.float)
image[:, 0:1] *= 100 / 255
image[:, 1:3] -= 128'''
image = np.copy(imgp.image)#.astype(np.float)
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float)

lab[:, :, 0] *= 100 / 255
lab[:, :, 1:3] -= 128
print('pre_dot', image[100][100])
lab = lab.dot(tmatrix.T)
lab[:, :, 1:3] += 128
lab[:, :, 0] *= 255 / 100
print('post_dot', image[100][100])

image = cv2.cvtColor(np.rint(lab).astype(np.uint8), cv2.COLOR_LAB2BGR)

'''image[:, :, 0:1] *= 100 / 255
image[:, :, 0:1] -= 128
image[:, :, :] /= 128
print('pre_dot', image[100][100])
image = image.dot(tmatrix.T)
image[:, :, :] *= 128
image[:, :] += 128
image[:, :, 0:1] *= 255 / 100
print('post_dot', image[100][100])'''

ref_color_index = 0
#image = np.rint(image)
for p in color_patches:
    image_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(image_mask, p, (255,))
    if COLOR_SPACE == 'xyz':
        x = np.average(cie_x[r[0]:r[1], r[2]:r[3]]) / 255
        y = np.average(cie_y[r[0]:r[1], r[2]:r[3]]) / 255
    elif COLOR_SPACE == 'lab':
        color_mean = cv2.mean(lab, image_mask)
        print('mean', lab.shape, lab[20][20], color_mean)
        z = color_mean[0] * 100 / 255
        x = color_mean[1] - 128
        y = color_mean[2] - 128
    else:
        raise ValueError('Unknown color space')
    cvalues = '%d, %.2f, %.2f' % (z, x, y)
    samples[ref_color_index] = (z, x, y)
    print(ref_color_index, (z, x, y), samples[ref_color_index])
    #nearest, d = nearest_color((z, x, y))
    index, name, *ref_color = REF_POINTS['lab'][ref_color_index]
    d = color_distance(ref_color, (z, x, y))
    #print(x, y, cvalues)
    #print()
    # Print color values
    cv2.drawContours(image_grids, [p], -1, (0, 255, 0), 2)
    pt1 = tuple((p[0].ravel() + [padding, padding + 16]).astype(np.uint))
    cv2.putText(image, cvalues, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)
    # Print ref color
    pt1 = tuple((p[0].ravel() + [padding, padding + 32]).astype(np.uint))
    cv2.putText(image, name, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)
    # Print distance
    distance = 'd=%.4f' % (d,)
    pt1 = tuple((p[0].ravel() + [padding, padding + 48]).astype(np.uint))
    cv2.putText(image, distance, pt1, cv2.FONT_HERSHEY_PLAIN,
        1, (255, 255, 0), 1)

    ref_color_index += 1

'''img_trans[:, 0:1] *= 255 / 100
img_trans[:, 1:3] += 128'''
cv2.imshow("Image Transformed", image.astype('uint8'))
cv2.waitKey(0)
