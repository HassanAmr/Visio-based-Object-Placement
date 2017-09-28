#!/usr/bin/env python

import rospy
import numpy as np
import cv2
import errno
import csv
from os import listdir
from os import makedirs
from os.path import join
from os.path import basename


def find_rotation_matrix(query_path, in_dir, out_dir, log_dir):


    try:
        makedirs(out_dir)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        pass

    MIN_MATCH_COUNT = 10

    img1 = cv2.imread(query_path,0)          # queryImage

    data_dir_test = in_dir
    datalist_test = [join(data_dir_test, f) for f in listdir(data_dir_test)]

    if datalist_test > 0:
        fp = open(log_dir+"/"+ "rotation_log.csv", 'w')
        fp.truncate()
        fpWriter = csv.writer(fp, delimiter='\t')
        fpWriter.writerow(["File", "Matches"])
        
        sift = cv2.xfeatures2d.SIFT_create()

    for j in datalist_test:

        img2 = cv2.imread(j,0) # trainImage

        # Initiate SIFT detector

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        fpWriter.writerow([j, len(good)])

        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            h,w = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img2 = cv2.polylines(cv2.imread(j),[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)
        else:
            print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)

        img3 = cv2.drawMatches(cv2.imread(query_path),kp1,img2,kp2,good,None,**draw_params)

        cv2.imwrite(out_dir + "/" + basename(j),img3)
    
    if datalist_test > 0:
        fp.close()