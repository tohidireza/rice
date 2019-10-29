import os

import cv2
import glob

import numpy as np
import xlsxwriter
from skimage import feature

classes = {'basmati', 'daneboland'}

# iterate on images of each class
for c in classes:
    features_list = []
    #   make xlsx file to store each class features
    workbook = xlsxwriter.Workbook('{}.xlsx'.format(c))
    worksheet = workbook.add_worksheet()
    for filename in glob.glob('{}/*.png'.format(c)):
        # load, convert to greyscale and resize image
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (20, 5))

        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        eps = 1e-7
        numPoints = 16
        radius = 3
        lbp = feature.local_binary_pattern(gray, numPoints,
                                           radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, numPoints + 3),
                                 range=(0, numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        print(hist)

        row = os.path.basename(filename).split('.')[0]
        # iterating through values of hist
        for column, value in enumerate(hist):
            # write lbp hist in xlsx file
            worksheet.write(int(row), column, str(value))

    workbook.close()

'''
References:
    - https://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
'''
