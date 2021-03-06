import sys
import argparse
import math
import operator
import pickle
import json
import time
import os
from pprint import pprint

import numpy as np
import cv2
import imutils
from scipy.optimize import minimize, least_squares
from scipy.stats import gaussian_kde
from scipy.special import comb
from sklearn.neighbors import KernelDensity
from sklearn.cluster import KMeans
import skimage.transform as sk_xform

# colormath is slower than my own implementation
from colormath.color_objects import LabColor
from colormath.color_diff import delta_e_cie2000


PATH_IN = './images/20200213_103455.jpg'
PATH_IN = './images/20200213_103542.jpg' # noise rectangles
PATH_IN = './images/color_checker.jpg'
PATH_IN = './images/20200220_120829.jpg' # yellowish
#PATH_IN = './images/20200220_120820.jpg' # blueish
#PATH_IN = './images/20200303_114801.jpg'
#PATH_IN = './images/20200303_114806.jpg'
#PATH_IN = './images/20200303_114812.jpg' # poor
#PATH_IN = './images/20200303_114815.jpg' # Need to fix Z
#PATH_IN = './images/20200303_114817.jpg' # Need to fix Z


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
parser.add_argument('--headless', action='store_true',
                    help='Run silently in the background.')
parser.add_argument('--filename', type=str,
                    help='Filename of the image to perform the task.')
parser.add_argument('--prefix', type=str, default='output',
                    help='Directory name to output the result of this color calibration task.')
parser.add_argument('--test', action='store_true',
                    help='Run with less iterations for testing.')
parser.add_argument('--max_nfev', type=int, default=4096,
                    help='Max iterations for least squares optimization.')
parser.add_argument('--poly_degree', type=int, default=4,
                    help='Degree of polynomial function for color transformation.')
parser.add_argument('--method', type=str, default='poly',
                    help='Max iterations for least squares optimization.')
args = parser.parse_args()
DEBUG = args.debug
HEADLESS = args.headless
TEST = args.test
lsq_verbose = 1
if DEBUG:
    lsq_verbose = 2

COLOR_XFORM_MATRIX = 0
COLOR_POLYNOMIAL = 1
COLOR_CALIB_METHOD = COLOR_POLYNOMIAL
if args.method != 'poly':
    COLOR_CALIB_METHOD = COLOR_XFORM_MATRIX
MAX_NFEV = args.max_nfev
POLY_DEGREE = args.poly_degree
PATH_IN = './images/{}'.format(args.filename)
PREFIX = args.prefix
p_ = './images/output/{}'.format(PREFIX)
if not os.path.isdir(p_):
    os.mkdir(p_)

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
        if not HEADLESS:
            cv2.imshow('Source', resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    @classmethod
    def display(cls, caption, image):
        if not HEADLESS:
            cv2.imshow(caption, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def output_image_file(self, name, image):
        try:
            self.__step_output += 1
        except AttributeError:
            self.__step_output = 1
        cv2.imwrite('./images/output/{}/{:02d}_{}.png'.format(
            PREFIX, self.__step_output, name), image)

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
        self.output_image_file('gaussian_blur', blurred)

        method, blockSize, C, *_ = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 7, 2)
        binary = cv2.adaptiveThreshold(blurred, 255, method,
                    cv2.THRESH_BINARY, blockSize, C)
        if DEBUG: self.display('Binary', binary)
        self.output_image_file('adaptive_threshold', binary)

        # Apply erosion to remove noises connected to the square.
        kernel = np.ones((9,9),np.uint8)
        binary = cv2.erode(binary, kernel)
        if DEBUG: self.display('Eroded', binary)
        self.output_image_file('erosion', binary)

        contours = cv2.findContours(binary.copy(), cv2.RETR_LIST,
            cv2.CHAIN_APPROX_TC89_KCOS)
        contours = imutils.grab_contours(contours)
        
        img_ = self.__img_resized.copy()
        self.render_contours(img_, contours)
        if DEBUG:
            self.display('Contours', img_)
        self.output_image_file('contours', img_)

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
            if A > 0.02 or A < 0.001: continue
            polygons.append(approx)
            M = cv2.moments(approx)
            center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
            centers.append(center)
            approx = approx.astype("int")
            cv2.circle(img_, tuple(np.rint(center).astype(np.uint)), 
                5, (255, 255, 0), -1)
            cv2.drawContours(img_, [approx], -1, (0, 255, 0), 2)
        if DEBUG: self.display('Squares', img_)
        self.output_image_file('shapes', img_)
        self.__shapes = np.array(polygons, dtype=np.float).reshape((-1, 4, 2))
        self.__shape_centers = np.array(centers, dtype=np.float)

    @classmethod
    def render_vectors(cls, image, vectors):
        colors = ((255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0))
        for v in vectors.astype(np.uint):
            label, vx, vy, cx, cy = v
            cv2.arrowedLine(image, (cx, cy), (cx+vx, cy+vy), colors[label], 
                2, tipLength=0.2)

    @classmethod
    def draw_straight_line(cls, image, poly):
        a, b = poly
        h, w, *_ = image.shape
        if b is None:
            pt1 = (int(a), 0)
            pt2 = (int(a), h - 1)
        else:
            x = 0
            pt1 = (x, int(a * x + b))
            x = w - 1
            pt2 = (x, int(a * x + b))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
            y = 0
            pt1 = (int((y - b) / a), y)
            y = h - 1
            pt2 = (int((y - b) / a), y)
        try:
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
        except OverflowError:
            print(image.shape, a, b, pt1, pt2)

    def detect_grids(self, columns, rows):
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
        km = KMeans(n_clusters=4).fit(vectors[:, 0:2])
        print('clustered directions', km.cluster_centers_[:, 0])
        print('labels', km.labels_)
        vectors_clustered = np.ndarray((km.cluster_centers_.shape[0], 2))
        vectors_clustered[:, 0] = np.cos(km.cluster_centers_[:, 0])
        vectors_clustered[:, 1] = np.sin(km.cluster_centers_[:, 0])
        vectors = np.concatenate((km.labels_.reshape(-1, 1), vectors, vertices), axis=1)
        print(vectors.shape)

        for l in range(4):
            v = vectors[vectors[:, 0]==l]
            print(l)
            print(np.mean(vectors[vectors[:, 0]==l], axis=0)[1:3])
            direction = np.arctan2(*np.mean(vectors[vectors[:, 0]==l], axis=0)[1:3])
            print(direction)
            if (direction <= PI / 4) and (direction > -PI / 4):
                down = l
            elif (direction <= -PI * 3/4) or (direction > PI * 3/4):
                up = l
            elif (direction <= -PI / 4) and (direction > -PI * 3/4):
                left = l
            elif (direction <= PI * 3/4) or (direction > PI / 4):
                right = l

        '''
        Output grid vectors are in the format of 
            [[label, vx, vy, cx, cy],... ],
        where (cx, cy) is the starting point and (vx, vy) is the vector.
        '''
        #vectors = np.sort(vectors, axis=0)
        directions = directions.ravel()
        pprint(vectors)
        print(directions.shape, vectors.shape)

        down = vectors[vectors[:, 0]==down]
        up = vectors[vectors[:, 0]==up]
        left = vectors[vectors[:, 0]==left]
        right = vectors[vectors[:, 0]==right]
        print('down', down)
        print('up', up)
        print('left', left)
        print('right', right)

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

        img_ = self.__img_resized.copy()
        self.render_vectors(img_, vectors)
        if DEBUG: self.display('Vectors', img_)
        self.output_image_file('sides_sparse', img_)

        """p_down = fit_line(down, columns, OR_VERTICAL)
        p_up = fit_line(up, columns, OR_VERTICAL)
        p_left = fit_line(left, rows, OR_HORIZONTAL)
        p_right = fit_line(right, rows, OR_HORIZONTAL)
        assert (len(p_down)==columns and len(p_up)==columns 
            and len(p_left)==rows and len(p_right)==rows), 'Grids lines mismatch'
        """

        def get_straight(points):
            '''
            Use points to find a straight line as polynomial function.

            Input vectors are in the format of 
                [[label, vx, vy, cx, cy],... ],
            where (cx, cy) is the starting point and (vx, vy) is the vector.
            '''
            d_ = points.reshape(1, -1)
            x = np.concatenate((d_[:, 3], d_[:, 3] + d_[:, 1]), axis=0)
            y = np.concatenate((d_[:, 4], d_[:, 4] + d_[:, 2]), axis=0)
            print('X DEV', x, np.std(x))
            if np.std(x) < 1:
                p_ = (np.mean(x), None)
            else:
                p_ = np.polyfit(x, y, 1)
            return p_

        vertical = np.vstack((down, up))
        s_left = get_straight(vertical[np.argmin(vertical[:, 3])])
        s_right = get_straight(vertical[np.argmax(vertical[:, 3])])
        horizontal = np.vstack((left, right))
        s_top = get_straight(horizontal[np.argmin(horizontal[:, 4])])
        s_bottom = get_straight(horizontal[np.argmax(horizontal[:, 4])])

        def s_line_intersect(p1, p2):
            '''
            Take polynomial coefficients of 2 straight lines and find the 
            intersection.
            '''
            a1, b1 = p1
            a2, b2 = p2
            if b1 is None and b2 is None:
                assert False, 'Parallel lines do not intersect, {}, {}'.format(p1, p2)
            elif b1 is None:
                x = a1
                y = a2 * x + b2
            elif b2 is None:
                x = a2
                y = a1 * x + b1
            else:
                x = -(b1 - b2) / (a1 - a2)
                y = a1 * x + b1
            return np.array((x, y))

        pt1 = s_line_intersect(s_left, s_top)
        pt2 = s_line_intersect(s_right, s_top)
        pt3 = s_line_intersect(s_right, s_bottom)
        pt4 = s_line_intersect(s_left, s_bottom)
        corners = np.vstack((pt1, pt2, pt3, pt4))
        print('CORNERS', corners)

        img_ = self.__img_resized.copy()
        self.draw_straight_line(img_, s_left)
        self.draw_straight_line(img_, s_right)
        self.draw_straight_line(img_, s_top)
        self.draw_straight_line(img_, s_bottom)
        for c in corners:
            try:
                cv2.circle(img_, tuple(c.astype(np.uint)), 8, (255, 255, 0), 2)
            except OverflowError:
                pass
        if DEBUG:
            self.display('Grids', img_)
        self.output_image_file('vectors_aligned', img_)

        def fit_rect_3d(params, corners):
            obj_points = np.array(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
            tmatrix = params.reshape(3, 3)
            vectors = np.copy(corners).reshape((-1, 2))
            vectors = np.hstack((vectors, 
                np.full((vectors.shape[0], 1), 0)))
            transformed = np.einsum('ij,kj->ik', vectors, tmatrix)
            #print(left_top, z, tmatrix, vectors, transformed, unity)
            d = np.linalg.norm(transformed - obj_points)
            print(transformed, d)
            return d

        obj_points = np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32).reshape(1, -1, 3)
        h, w, *_ = self.__img_resized.shape
        obj_points = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float)
        camera_matrix = np.zeros((3, 3), dtype=np.float)
        #, flags=cv2.CV_CALIB_USE_INTRINSIC_GUESS
        '''img_points = np.hstack((corners, 
            np.full((corners.shape[0], 1), 0))).astype(np.float32)'''
        print(obj_points.shape, corners)
        xform = sk_xform.ProjectiveTransform()
        ret = xform.estimate(corners, obj_points)
        tmatrix = xform.params
        tmatrix_inv = np.linalg.inv(tmatrix)
        print(ret, tmatrix)

        '''img_points = np.hstack((corners, 
            np.full((corners.shape[0], 1), 0))).astype(np.float32)
        for c in img_points:
            print(c, c.dot(tmatrix.T))
        raise'''

        GRID_MERIDIAN = 0
        GRID_PARALLEL = 1

        def fit_grid_lines(vectors, group_data, group_n, orientation, tmatrix):
            '''
            Use aligned points in 2D to group vectors into clusters of grid 
            lines.

            vectors - Vectors of shape (N, 5) in the format of (label, vx, vy, cx, cy)
            group_data - Data for clustering of shape (N, 2) as points in 2D
            group_n - Number of grid lines to fit
            orientation - Meridian or parallel
            tmatrix - Projective transformation matrix to align group_data

            Return grid_lines as polynomial coefficients of the grid lines.
            '''
            grid_lines = list()
            aligned_vectors = list()
            group_index = 0 if orientation==GRID_MERIDIAN else 1

            c_ = group_data
            c_ = np.hstack((c_, 
                np.full((c_.shape[0], 1), 0)))
            c_ = np.dot(c_, tmatrix.T)
            c_ = c_[:, group_index].reshape(-1, 1)

            km = KMeans(n_clusters=group_n).fit(np.concatenate((c_, c_), axis=1))
            for c in range(group_n):
                d_ = vectors[km.labels_==c]
                x = np.concatenate((d_[:, 3], d_[:, 3] + d_[:, 1]), axis=0)
                y = np.concatenate((d_[:, 4], d_[:, 4] + d_[:, 2]), axis=0)
                print('X DEV', x, np.std(x))
                if np.std(x) < 1:
                    p_ = (np.mean(x), None)
                else:
                    p_ = np.polyfit(x, y, 1)
                grid_lines.append(p_)
                aligned_vectors.append(d_)
                try:
                    print(np.poly1d(p_), p_)
                except TypeError:
                    print(p_)

                if False:
                    img_ = self.__img_resized.copy()
                    self.draw_straight_line(img_, p_)
                    self.render_vectors(img_, d_)
                    self.display('Grid Line', img_)

            return grid_lines, aligned_vectors

        vertices = np.hstack((down[:, -2:], 
            np.full((down.shape[0], 1), 0)))
        vertices = np.dot(vertices, tmatrix.T)
        print(vertices)
        #down[:, -2:] = vertices[:, :2]
        p_down, v = fit_grid_lines(down, down[:, 3:5], columns, GRID_MERIDIAN, tmatrix)
        print('p_down', p_down)
        pprint(v)
        p_up, v = fit_grid_lines(up, up[:, 3:5], columns, GRID_MERIDIAN, tmatrix)
        print('p_up', p_up)
        pprint(v)
        p_left, v = fit_grid_lines(left, left[:, 3:5], rows, GRID_PARALLEL, tmatrix)
        p_right, v = fit_grid_lines(right, right[:, 3:5], rows, GRID_PARALLEL, tmatrix)
        assert (len(p_down)==columns and len(p_up)==columns 
            and len(p_left)==rows and len(p_right)==rows), 'Error fitting grids lines'

        meridians = np.vstack((p_down, p_up))
        parallels = np.vstack((p_left, p_right))
        meridians_sort = np.copy(meridians)
        m_ = meridians_sort[meridians_sort[:, 1]==None]
        meridians_sort[meridians_sort[:, 1]==None] = np.flip(meridians_sort[meridians_sort[:, 1]==None], axis=1)
        meridians_sort[meridians_sort==None] = -1
        meridians = meridians[np.argsort(meridians_sort[:, 0]/meridians_sort[:, 1])]
        parallels = parallels[np.argsort(parallels[:, 1])]
        intersects = list()
        for p in parallels:
            for m in meridians:
                intersects.append(s_line_intersect(p, m))
        intersects = np.rint(intersects).astype(np.uint)

        '''params = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float)
        res_lsq = least_squares(fit_rect_3d, params, args=(
            corners.flatten() / self.__working_width,), jac='3-point', loss='soft_l1', tr_solver='exact',
            verbose=1)
        tmatrix = res_lsq.x.reshape((3, 3))
        print('tmatrix', tmatrix)
        tmatrix_inv = np.linalg.inv(tmatrix)
        print('tmatrix_inv', tmatrix_inv)
        print(corners, np.ones((corners.shape[0], 1)))
        c_ = np.hstack((corners, 
            np.full((corners.shape[0], 1), 100)))
        print('corners', corners.astype(np.uint))
        print('transformed', c_.dot(tmatrix.T))'''

        img_ = self.__img_resized.copy()
        h, w, *_ = img_.shape
        for p in meridians:
            a, b = p
            if b is None:
                x = int(a)
                pt1 = (x, 0)
                pt2 = (x, h - 1)
            else:
                y = 0
                pt1 = (int((y - b) / a), y)
                y = h - 1
                pt2 = (int((y - b) / a), y)
            cv2.line(img_, pt1, pt2, (0, 255, 0), 2)
        for p in parallels:
            a, b = p
            if b is None:
                x = int(a)
                pt1 = (x, 0)
                pt2 = (x, h - 1)
            else:
                x = 0
                pt1 = (x, int(a * x + b))
                x = w - 1
                pt2 = (x, int(a * x + b))
            cv2.line(img_, pt1, pt2, (0, 255, 0), 2)
        i = 0
        for intersect in intersects:
            try:
                cv2.circle(img_, tuple(intersect), 4, (0, 255, 255), 2)
                cv2.putText(img_, '{}'.format(i), tuple(intersect), 
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            except OverflowError:
                pass
            i += 1
        if DEBUG:
            self.display('Grid Lines', img_)
        self.output_image_file('grid_lines', img_)

        intersects = intersects.reshape((rows*2, columns, 4))
        #print(intersects[::2], intersects[1::2, :, :2])
        self.__grids = np.hstack((intersects[::2].reshape(-1, 4), 
            intersects[1::2, :, -2:].reshape(-1, 2), 
            intersects[1::2, :, :2].reshape(-1, 2))).reshape(-1, 4, 2)

        img_ = self.__img_resized.copy()
        i = 0
        for p in self.__grids:
            cv2.drawContours(img_, [p], -1, (0, 255, 0), 2)
            cv2.putText(img_, '{}'.format(i), tuple(np.mean(p, axis=0).astype(np.uint)), 
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            i += 1
        if DEBUG:
            self.display('Grids', img_)
        self.output_image_file('grids', img_)

    @property
    def grids(self):
        return self.__grids

    @property
    def shapes(self):
        return self.__shapes

    @property
    def shapes(self):
        return self.__shapes

    @property
    def shape_centers(self):
        return self.__shape_centers

    @property
    def source_factor(self):
        return self.__source_factor

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


class Geometry:

    @classmethod
    def rect_iou(cls, bbox1, bbox2):
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


class ColorMath:

    @classmethod
    def color_distance_(cls, color_a_, color_b_):
        l1, a1, b1 = color_a_
        l2, a2, b2 = color_b_
        color_a = LabColor(l1/100, a1, b1)
        color_b = LabColor(l2/100, a2, b2)
        return delta_e_cie2000(color_a, color_b)

    @classmethod
    def color_distance(cls, color_a_, color_b_):
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

    @classmethod
    def color_distance_sum(cls, colors_a, colors_b):
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

    @classmethod
    def nearest_color(cls, sample):
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

    @classmethod
    def color_transform_poly(cls, params, degree, colors_train, colors_target):
        colors_train = colors_train.reshape((24, 3))
        coeffs = Polynomial.coeffs_1d_to_3d(params, degree)
        transformed = np.copy(colors_train)
        for c in range(3):
            l = colors_train[:, 0]
            a = colors_train[:, 1]
            b = colors_train[:, 2]
            transformed[:, c] = np.polynomial.polynomial.polyval3d(l, a, b, coeffs[c])
        target = colors_target.reshape((24, 3))
        # Errors encountered in scipy leaset squares optimization with CIEDE2000 calculation 
        #d = color_distance_sum(transformed, target)
        d = np.linalg.norm(transformed - target)
        return d

    @classmethod
    def color_transform(cls, tmatrix, colors_train, colors_target):
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
        d = cls.color_distance_sum(transformed, target)
        '''for i in range(target.shape[0]):
            d_check += color_distance(transformed[i], target[i])
            my_d = color_distance(transformed[i], target[i])
            color_a = LabColor(transformed[i][0]/100, transformed[i][1], transformed[i][2])
            color_b = LabColor(target[i][0]/100, target[i][1], target[i][2])
            color_math_d = delta_e_cie2000(color_a, color_b)
            print('mine vs color_math', my_d, color_math_d)
        print('color_distance_sum', d, d_check)'''
        return d
        d = 0.0
        for i in range(target.shape[0]):
            color_a = LabColor(transformed[i][0]/100, transformed[i][1], transformed[i][2])
            color_b = LabColor(target[i][0]/100, target[i][1], target[i][2])
            d += delta_e_cie2000(color_a, color_b)
        return d

    @classmethod
    def get_color_xform_matrix(cls, samples, ref_colors, initial_matrix, max_nfev=3200):
        '''
        Use source color samples and reference color values to optimize 3D
        transformation matrix for color transformation.
        '''
        tmatrix = initial_matrix
        res_lsq = least_squares(cls.color_transform, tmatrix, args=(
            samples.flatten(), ref_colors.flatten()), jac='3-point', loss='soft_l1',
            tr_solver='exact', ftol=None, xtol=1e-15, gtol=None, max_nfev=max_nfev,
            verbose=lsq_verbose)
        tmatrix = res_lsq.x.reshape(3, 3)
        return tmatrix, res_lsq

    @classmethod
    def get_color_polynomial(cls, samples, ref_colors, degree, initial_coeffs, max_nfev=3200):
        '''
        Use source color samples and reference color values to optimize polynomial 
        coefficients for color transformation.
        '''
        lsq_verbose = 2
        D = degree + 1
        params = Polynomial.coeffs_3d_to_1d(initial_coeffs, degree)
        res_lsq = least_squares(cls.color_transform_poly, params.ravel(), args=(
            degree, samples.flatten(), ref_colors.flatten()), jac='3-point', loss='soft_l1',
            tr_solver='exact', ftol=1e-15, xtol=None, gtol=None, max_nfev=max_nfev,
            verbose=lsq_verbose)
        coeffs = Polynomial.coeffs_1d_to_3d(res_lsq.x, degree)
        return coeffs, res_lsq


class Polynomial:

    @classmethod
    def coeffs_1d_to_3d(cls, params, degree, N=3):
        '''
        Some polynomial functions use array of N*D to store coefficients while
        acutal number of coefficients for the polynomial function is comb(N+D, D).
        This function takes array of shape (N, D) and return 1D array of the 
        coefficients.
        '''
        d = degree
        D = degree + 1
        params = params.reshape(N, -1)
        coeffs = np.zeros((N, D, D, D), dtype=params.dtype)
        for c in range(N):
            p = 0
            for z in range(D):
                for y in range(D-z):
                    l = D - y - z
                    coeffs[c, 0:l, y, z] = params[c, p:p+l]
                    p += l
        return coeffs

    @classmethod
    def coeffs_3d_to_1d(cls, coeffs, degree, N=3):
        '''
        Some polynomial functions use array of N*D to store coefficients while
        acutal number of coefficients for the polynomial function is comb(N+D, D).
        This function takes 1D array of the coefficients and return array of 
        shape (N, D).
        '''
        d = degree
        D = degree + 1
        params = np.zeros((N, int(comb(d + N, d))), dtype=coeffs.dtype)
        for c in range(N):
            p = 0
            for z in range(D):
                for y in range(D-z):
                    l = D - y - z
                    params[c, p:p+l] = coeffs[c, 0:l, y, z]
                    p += l
        return params

    @classmethod
    def poly_extend(cls, coeffs):
        '''
        Increase degree of polynomial by 1 and return resized coefficients array.
        '''
        variables, D, *_ = coeffs.shape
        new = np.zeros((variables, D+1, D+1, D+1), dtype=coeffs.dtype)
        new[:, :D, :D, :D] = coeffs
        return new

        # Fill axis 3 with zeros of shape (1,)
        c_ = coeffs.reshape(-1, D)
        s_ = np.zeros((variables*D**2, 1))
        coeffs = np.hstack((c_, s_)).reshape(-1, D, D+1)

        # Fill axis 2 with zeros of shape (1, 3)
        c_ = coeffs.reshape(-1, D, D+1)
        s_ = np.zeros((variables*D, 1, D+1))
        coeffs = np.hstack((c_, s_)).reshape(-1, D, D+1, D+1)

        # Fill axis 1 with zeros of shape (1, 3, 3)
        c_ = coeffs.reshape(-1, D, D+1, D+1)
        s_ = np.zeros((variables, 1, D+1, D+1))
        coeffs = np.hstack((c_, s_)).reshape(-1, D+1, D+1, D+1)

        return coeffs


if __name__ == "__main__":
    # load the source image
    imgp = ImageProcessing(PATH_IN)
    imgp.normalize_source()

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    imgp.get_contours()

    imgp.approximate_contours()
    imgp.detect_grids(6, 4)


    grids = imgp.grids

    color_patches = list()
    image_grids = np.copy(imgp.image)
    image_mask = np.zeros(imgp.image.shape, np.uint8)
    for p in np.rint(grids * imgp.source_factor).astype(np.uint):
        p = p.reshape(-1, 1, 2)
        color_patches.append(p)
        cv2.drawContours(image_grids, [p], -1, (0, 255, 0), 2)
        cv2.fillConvexPoly(image_mask, p, (255, 255, 255))

    # show the output image
    if DEBUG:
        cv2.imshow("Grids", image_grids)
        cv2.waitKey(0)
        cv2.imshow("Mask", image_mask)
        cv2.waitKey(0)
    imgp.output_image_file('mask', image_mask)

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
    color_distances_source = list()
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
        d = ColorMath.color_distance(ref_color, (z, x, y))
        color_distances_source.append(d)
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

    color_distances_source = np.array(color_distances_source)
    source_samples = np.copy(samples)

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

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float)
    imgp.output_image_file('source_labeled', image)
    if not HEADLESS:
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    if COLOR_CALIB_METHOD == COLOR_XFORM_MATRIX:
        initial_matrix = np.array((1, 0, 0, 0, 1, 0, 0, 0, 1), dtype=np.float)
        fit_iterations = list()
        t_ = time.perf_counter()
        tmatrix, res_lsq = ColorMath.get_color_xform_matrix(samples, ref_colors, initial_matrix)
        time_fit = time.perf_counter() - t_
        fit_iterations.append(res_lsq.nfev)

        image = np.copy(imgp.image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float)

        lab[:, :, 0] *= 100 / 255
        lab[:, :, 1:3] -= 128
        t_ = time.perf_counter()
        lab = lab.dot(tmatrix.T)
        time_transform = time.perf_counter() - t_
        lab[:, :, 1:3] += 128
        lab[:, :, 0] *= 255 / 100

        coeffs = None
        n_coeffs = None

    else:
        degree = 1
        D = degree + 1
        coeffs = np.zeros((3, D, D, D), dtype=np.float)
        coeffs[0][1][0][0] = 1
        coeffs[1][0][1][0] = 1
        coeffs[2][0][0][1] = 1

        t_ = time.perf_counter()
        if TEST:
            max_nfev = 4
        else:
            max_nfev = MAX_NFEV
        fit_iterations = list()
        coeffs, res_lsq = ColorMath.get_color_polynomial(samples, ref_colors, degree, coeffs, max_nfev)
        fit_iterations.append(res_lsq.nfev)
        for i in range(POLY_DEGREE - 1):
            coeffs = Polynomial.poly_extend(coeffs)
            degree += 1
            coeffs, res_lsq = ColorMath.get_color_polynomial(samples, ref_colors, degree, coeffs, max_nfev)
            fit_iterations.append(res_lsq.nfev)
        n_coeffs = len(res_lsq.x)
        time_fit = time.perf_counter() - t_

        image = np.copy(imgp.image)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float)
        lab[:, :, 0] *= 100 / 255
        lab[:, :, 1:3] -= 128
        transformed = np.copy(lab)
        for c in range(3):
            l = lab[:, :, 0]
            a = lab[:, :, 1]
            b = lab[:, :, 2]
            t_ = time.perf_counter()
            transformed[:, :, c] = np.polynomial.polynomial.polyval3d(l, a, b, coeffs[c])
            time_transform = time.perf_counter() - t_
        transformed[:, :, 1:3] += 128
        transformed[:, :, 0] *= 255 / 100
        lab = transformed

    lab = np.clip(lab, 0, 255)
    lab = np.rint(lab).astype(np.uint8)

    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    ref_color_index = 0
    #image = np.rint(image)
    color_distances_calib = list()
    for p in color_patches:
        image_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(image_mask, p, (255,))
        if COLOR_SPACE == 'xyz':
            x = np.average(cie_x[r[0]:r[1], r[2]:r[3]]) / 255
            y = np.average(cie_y[r[0]:r[1], r[2]:r[3]]) / 255
        elif COLOR_SPACE == 'lab':
            color_mean = cv2.mean(lab, image_mask)
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
        d = ColorMath.color_distance(ref_color, (z, x, y))
        color_distances_calib.append(d)
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
        
    color_distances_calib = np.array(color_distances_calib)

    if COLOR_CALIB_METHOD == COLOR_XFORM_MATRIX:
        method = 'projective_transformation'
    else:
        method = '{} degree polynomial'.format(degree)
    fit_summary = {
        'method': method,
        'fit_max_nfev': fit_iterations,
        'image_dimension': image.shape,
        'color_distances_source': color_distances_source,
        'color_distances_calib': color_distances_calib,
        'distance_sum_source': np.sum(color_distances_source),
        'distance_max_source': np.max(color_distances_source),
        'distance_sum_calibrated': np.sum(color_distances_calib),
        'distance_max_calibrated': np.max(color_distances_calib),
        'coeffs': coeffs,
        'num_coeffs': n_coeffs,
        'time_fit': time_fit,
        'time_transform': time_transform,
        'source_samples': source_samples,
        'calib_samples': samples,
    }

    with open('./images/output/{}/fit_summary.pkl'.format(PREFIX), 'wb') as fp:
        pickle.dump(fit_summary, fp)
    with open('./images/output/{}/fit_summary.json'.format(PREFIX), 'w') as fp:
        fit_summary.pop('coeffs', None)
        fit_summary.pop('source_samples', None)
        fit_summary.pop('calib_samples', None)
        fit_summary.pop('color_distances_source', None)
        fit_summary.pop('color_distances_calib', None)
        pprint(fit_summary)
        json.dump(fit_summary, fp, indent=4)

    '''img_trans[:, 0:1] *= 255 / 100
    img_trans[:, 1:3] += 128'''
    # Transformation matrix, sum of color distances: 249.69263337647757
    # 3 degrees polynomial with 320000 iterations, sum of color distances: 213.58460800749998
    # 3 pass polynomial fit with 3200, 3200, 3200 iterations, sum of color distances: 210.72933187615357
    # Sum of color distances: 152.23432845231142
    image = image.astype('uint8')
    imgp.output_image_file('calibrated_labeled', image)
    if not HEADLESS:
        cv2.imshow("Image Transformed", image)
        cv2.waitKey(0)

    # Render color chart
    lab = np.zeros((600, 600, 3), dtype=np.float)
    l = 128
    for y in range(600):
        for x in range(600):
            a = x / 600 * 256
            b = 256 - (y / 600 * 256)
            lab[y, x, 0] = l
            lab[y, x, 1] = a
            lab[y, x, 2] = b

    with open('./images/output/{}/fit_summary.pkl'.format(PREFIX), 'rb') as fp:
        fit_summary = pickle.load(fp)

    coeffs = fit_summary['coeffs']

    image = cv2.cvtColor(lab.astype('uint8'), cv2.COLOR_LAB2BGR)
    for c in REF_POINTS['lab'][:19]:
        i, name, l, a, b = c
        a += 128
        b += 128
        x = int(a / 256 * 600)
        y = int((256 - b) / 256 * 600)
        pt0 = (x, y)

        l, a, b = fit_summary['source_samples'][i - 1]
        a += 128
        b += 128
        x = int(a / 256 * 600)
        y = int((256 - b) / 256 * 600)
        pt1 = (x, y)

        l, a, b = fit_summary['calib_samples'][i - 1]
        a += 128
        b += 128
        x = int(a / 256 * 600)
        y = int((256 - b) / 256 * 600)
        pt2 = (x, y)

        cv2.circle(image, pt1, 4, (255, 255, 0), 1)
        cv2.arrowedLine(image, pt1, pt2, (255, 255, 0), 1, tipLength=0.1)
        cv2.arrowedLine(image, pt2, pt0, (0, 0, 255), 1, tipLength=0.1)

        pt1 = tuple(((np.array(pt1) + np.array(pt2)) / 2).astype(np.uint))
        #cv2.putText(image, name, pt1, cv2.FONT_HERSHEY_PLAIN,
        #    1, (255, 255, 0), 1)

    cv2.imwrite('./images/output/{}/color_chart.png'.format(PREFIX), image)
    if not HEADLESS:
        cv2.imshow("Color Space", image)
        cv2.waitKey(0)