import pickle

import numpy as np
import cv2

from color_calib import REF_POINTS


lab = np.zeros((600, 600, 3), dtype=np.float)
l = 128
for y in range(600):
    for x in range(600):
        a = x / 600 * 256
        b = 256 - (y / 600 * 256)
        lab[y, x, 0] = l
        lab[y, x, 1] = a
        lab[y, x, 2] = b

with open('./images/output/fit_summary.pkl', 'rb') as fp:
    fit_summary = pickle.load(fp)
    print(fit_summary)

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
    print(i, name)
    print(l, a, b, pt1)

    l, a, b = fit_summary['calib_samples'][i - 1]
    a += 128
    b += 128
    x = int(a / 256 * 600)
    y = int((256 - b) / 256 * 600)
    pt2 = (x, y)
    print(l, a, b, pt2)

    cv2.circle(image, pt1, 4, (255, 255, 0), 1)
    cv2.arrowedLine(image, pt1, pt2, (255, 255, 0), 1, tipLength=0.1)
    cv2.arrowedLine(image, pt2, pt0, (0, 0, 255), 1, tipLength=0.1)

    pt1 = tuple(((np.array(pt1) + np.array(pt2)) / 2).astype(np.uint))
    #cv2.putText(image, name, pt1, cv2.FONT_HERSHEY_PLAIN,
    #    1, (255, 255, 0), 1)

cv2.imshow("Color Space", image)
cv2.imwrite('./images/output/color_chart.jpg', image)
cv2.waitKey(0)