import numpy as np
import cv2
import os


# crop rectangle returned by cv2.minAreaRect
def crop_minAreaRect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    # correct cropped image rotation
    height, width = img_crop.shape[0], img_crop.shape[1]
    if height > width:
        img_crop = np.rot90(img_crop)

    return img_crop



# Load daneboland and basmati image in BGR
daneboland = cv2.imread('daneboland01.jpg')
basmati = cv2.imread('basmati02.jpg')

# resize images to make their size similar
daneboland = cv2.resize(daneboland, (425, 298))
basmati = cv2.resize(basmati, (425, 336))

# covert images to greyscale for thresholding
daneboland_gray = cv2.cvtColor(daneboland, cv2.COLOR_BGR2GRAY)
basmati_gray = cv2.cvtColor(basmati, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur to denoise images
daneboland_gray = cv2.GaussianBlur(daneboland_gray, (7, 7), 0)
basmati_gray = cv2.GaussianBlur(basmati_gray, (7, 7), 0)

# apply Otsu's thresholding for finding contours
daneboland_thresh = cv2.threshold(daneboland_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
basmati_thresh = cv2.threshold(basmati_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# find contours to segment parts
daneboland_contours = cv2.findContours(daneboland_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
basmati_contours = cv2.findContours(basmati_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

# apply mask to remove background
daneboland = cv2.bitwise_and(daneboland, daneboland, mask=daneboland_thresh)
basmati = cv2.bitwise_and(basmati, basmati, mask=basmati_thresh)

# make folders to save croped datasets of each image
daneboland_dir = 'daneboland'
basmati_dir = 'basmati'
if not os.path.exists(daneboland_dir):
    os.makedirs(daneboland_dir)
if not os.path.exists(basmati_dir):
    os.makedirs(basmati_dir)

# crop and save each segment
for i, cnt in enumerate(daneboland_contours):
    rect = cv2.minAreaRect(cnt)
    cropped = crop_minAreaRect(daneboland, rect)
    cv2.imwrite(os.path.join(daneboland_dir, '{}.png'.format(i)), cropped)
for i, cnt in enumerate(basmati_contours):
    rect = cv2.minAreaRect(cnt)
    cropped = crop_minAreaRect(basmati, rect)
    cv2.imwrite(os.path.join(basmati_dir, '{}.png'.format(i)), cropped)

daneboland = cv2.resize(daneboland, None, fy=2, fx=2)
basmati = cv2.resize(basmati, None, fy=2, fx=2)

cv2.imshow("daneboland", daneboland)
cv2.imshow("basmati", basmati)

cv2.waitKey(0)

'''
References:
    - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    - https://jdhao.github.io/2019/02/23/crop_rotated_rectangle_opencv/
'''
