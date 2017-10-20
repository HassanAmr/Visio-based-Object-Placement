#! /usr/bin/python

import rospy
import cv2
import numpy as np


def crop(image1, image2, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """

    inv_image1 = cv2.bitwise_not(image1)
    inv_image2 = cv2.bitwise_not(image2)

    #image = cv2.imread(inputImg)
    if len(inv_image1.shape) == 3:
        flatImage1 = np.max(inv_image1, 2)
        flatImage2 = np.max(inv_image2, 2)
    else:
        flatImage1 = inv_image1
        flatImage2 = inv_image2
    assert len(flatImage1.shape) == 2

    rows = np.where(np.max(flatImage1, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage1, 1) > threshold)[0]
        inv_image1 = inv_image1[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
        inv_image2 = inv_image2[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        inv_image1 = inv_image1[:1, :1]
        inv_image2 = inv_image2[:1, :1]

    return [cv2.bitwise_not(inv_image1),cv2.bitwise_not(inv_image2)]
    
    #img = cv2.imread(image)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #_,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    #contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #cnt = contours[0]
    #x,y,w,h = cv2.boundingRect(cnt)
    #crop = img[y:y+h,x:x+w]
    #return crop
    #cv2.imwrite('sofwinres.png',crop)