import pickle

import numpy as np
import cv2

from color_calib import REF_POINTS


lab = np.zeros((600, 600, 3), dtype=np.float)
l = 128
for y in range(600):
    for x in range(600):
        a = 256 - x / 600 * 256
        b = y / 600 * 256
        lab[x, y, 0] = l
        lab[x, y, 1] = a
        lab[x, y, 2] = b

with open('fit_summary.pkl', 'rb') as fp:
    fit_summary = pickle.load(fp)
    print(fit_summary)

coeffs = fit_summary['coeffs']
transformed = np.copy(np.array(REF_POINTS['lab'])[:, 2:5]).astype(np.float)
for c in range(3):
    l = transformed[:, 0]
    a = transformed[:, 1]
    b = transformed[:, 2]
    transformed[:, c] = np.polynomial.polynomial.polyval3d(l, a, b, coeffs[c])
transformed[:, 1:3] += 128
transformed[:, 0] *= 255 / 100

image = cv2.cvtColor(lab.astype('uint8'), cv2.COLOR_LAB2BGR)
for c in REF_POINTS['lab']:
    i, name, l, a, b = c
    a += 128
    b += 128
    x = int((256 - a) / 256 * 600)
    y = int(b /256 * 600)
    pt1 = (x, y)
    cv2.circle(image, pt1, 8, (255, 255, 0), 1)
    print(l, a, b, pt1)

    l, a, b = transformed[i - 1]
    x = int((256 - a) / 256 * 600)
    y = int(b /256 * 600)
    pt2 = (x, y)
    print(l, a, b, pt2)
    cv2.arrowedLine(image, pt1, pt2, (255, 255, 0), 1, tipLength=0.1)

cv2.imshow("Color Space", image)
cv2.waitKey(0)