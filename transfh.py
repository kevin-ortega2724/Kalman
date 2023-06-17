"""
Transformada Hough
"""

import math

import cv2
import numpy as np


def non_max_supress(linesP):
    unitvectors = list()
    maxpair = [0, 1]
    delta = math.inf

    if linesP is not None:
        # take each line
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            # compute the norm
            norma = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** (1.0 / 2.0)
            # compute its unit vector
            unitvectors.append([(x2 - x1) / norma, (y2 - y1) / norma])

    # number of unit vectors:
    nv = unitvectors.__len__()
    if nv > 1:
        for i in range(0, nv - 1):
            for j in range(i + 1, nv):
                punto = unitvectors[i][0] * unitvectors[j][0] + unitvectors[i][1] * unitvectors[j][1]
                print(i, j, unitvectors[i], unitvectors[j], punto)
                if abs(punto) < delta:
                    # this is the new pair
                    delta = abs(punto)
                    maxpair = [i, j]

    return maxpair[0], maxpair[1]


def encontrarlineas(img):
    dst = cv2.Canny(img, 50, 200, None, 3)
    # cdstP = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(image=dst, rho=1, theta=np.pi / 180, threshold=10, minLineLength=20, maxLineGap=5)

    if linesP is not None:
        dim = len(linesP)
        # print('dim=', dim)
        i, j = non_max_supress(linesP)
    else:
        print('dim=', 0)
        i, j = 0, 0

    return linesP, i, j