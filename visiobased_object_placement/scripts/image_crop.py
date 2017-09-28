#! /usr/bin/python

import rospy
import cv2
import numpy as np


def crop(inputImg, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """

    image = cv2.imread(inputImg)
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image
    
    #img = cv2.imread(image)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    #contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cnt = contours[0]
    #x,y,w,h = cv2.boundingRect(cnt)
    #crop = img[y:y+h,x:x+w]
    #return crop
    #cv2.imwrite('sofwinres.png',crop)