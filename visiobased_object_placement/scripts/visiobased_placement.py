#!/usr/bin/env python

import rospy
import rosparam
import rospkg
import errno
import os
import io
import argparse
import numpy as np
import datetime
import time
import csv
import image_retrieval
import classify_image
import download_images
import image_rotation
import image_crop
import detect_image
import ros_image_listener
import tensorflow as tf
from sensor_msgs import msg
from cv_bridge import CvBridge, CvBridgeError # ROS Image message -> OpenCV2 image converter
import cv2 # OpenCV2 for saving an image
from PIL import Image 

parser = argparse.ArgumentParser()


parser.add_argument("--download_images", help="A flag to re-download images from the web", action="store_true")

parser.add_argument("--from_disk", help="run pipeline from disk")

args = parser.parse_args()



bridge = CvBridge()

def image_callback(msg):

    CACHED_QUERY_FILE_NAME = rospy.get_param("/visiobased_placement/CACHED_QUERY_FILE_NAME")
    CACHE_THRESHOLD = rospy.get_param("/visiobased_placement/CACHE_THRESHOLD")
    DIST_TYPE = rospy.get_param("/visiobased_placement/DIST_TYPE")
    GRAPH_PATH = rospy.get_param("/visiobased_placement/GRAPH_PATH")
    LABELS_PATH = rospy.get_param("/visiobased_placement/LABELS_PATH")
    LOG_PATH = rospy.get_param("/visiobased_placement/LOG_PATH")
    N = rospy.get_param("/visiobased_placement/N")
    RETRIEVED_PATH = rospy.get_param("/visiobased_placement/RETRIEVED_PATH")
    ROTATION_PATH = rospy.get_param("/visiobased_placement/ROTATION_PATH")
    SEARCH_DESINATION_DIR = rospy.get_param("/visiobased_placement/SEARCH_DESINATION_DIR")
    UPRIGHT_PATH = rospy.get_param("/visiobased_placement/UPRIGHT_PATH")
    WEIGHTS_PATH = rospy.get_param("/visiobased_placement/WEIGHTS_PATH")
    download_limit = rospy.get_param("/visiobased_placement/download_limit")
    keywords = rospy.get_param("/visiobased_placement/keywords")
    #search_keyword = rospy.get_param("/visiobased_placement/search_keyword") #TODO: This should be removed once API is online

    try:
        # Convert your ROS Image message to OpenCV2
        bg_subtracted_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        sub_once.unregister()
    except CvBridgeError, e:
        print(e)
    #else:
        # Save your OpenCV2 image as a jpeg 
        #cv2.imwrite('camera_image.jpeg', cv2_img)

    t0_cache = time.time()

    
    #bg_subtracted_path = "bg_subtracted_image.jpg"
    #scene_path = "scene_image.jpg"
    #bg_subtracted_image = cv2.imread(bg_subtracted_path)
    #scene_image = cv2.imread(scene_path)
    cv_query_image = image_crop.crop(bg_subtracted_image)
    cv2.imwrite("query_image.jpg", cv_query_image)
    QUERY_IMG = np.asarray( cv2.cvtColor(cv_query_image[:,:], cv2.COLOR_BGR2RGB) )

    # Setup Session
    # This will be used for the chaching  and retrieve_nsmallest_dist steps
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = image_retrieval.vgg16(imgs, WEIGHTS_PATH, sess)


    cwd = os.getcwd()
    dirs_list = [ x for x in os.listdir(cwd) if os.path.isdir(x) ]


    min_dist = CACHE_THRESHOLD
    found_cache = False
    for curr_dir in dirs_list:
        #for the list above check query_image.jpeg against curr image in the following
        #test_image = cwd + "/"+ curr_dir + CACHED_QUERY_FILE_NAME
        test_image_path = curr_dir + "/" + CACHED_QUERY_FILE_NAME
        curr_dist = image_retrieval.retrieve_dist(QUERY_IMG, test_image_path, DIST_TYPE, vgg, sess)
        print(curr_dist)
        if curr_dist < min_dist:
            min_dist = curr_dist
            curr_dir_session = curr_dir
            found_cache = True

    print("\nMin dist: %f\n" % min_dist)
    #True is the default case for all steps unless..
    run_cloud_api_step = True
    run_image_download_step = True

    if found_cache:
        print(curr_dir_session)
        # we set it from the cached query_image since we want all the stats to correspond to the same exact image. 
        cv_image = cv2.imread(curr_dir_session + "/" + CACHED_QUERY_FILE_NAME)
        QUERY_IMG = np.asarray( cv2.cvtColor(cv_image[:,:], cv2.COLOR_BGR2RGB) )
        #QUERY_IMG=curr_dir_session + CACHED_QUERY_FILE_NAME 
        f = open(curr_dir_session + "/checklist.csv", 'rb')
        reader = csv.reader(f)
        for row in reader:
            print (row)
        #TODO: set keywords from keyword.txt
        #TODO: Check checklist and set the following accordingly
        run_cloud_api_step = False
        run_image_download_step = False

        f.close()
    else:
        curr_dir_session = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        try:
            os.makedirs(curr_dir_session)
        except OSError, e:
            raise  # Here we raise even when folder exists because it should not exist.

        cv2.imwrite(curr_dir_session + "/query_image.jpg", cv_query_image)
        cv2.imwrite(curr_dir_session + "/bg_subtracted_image.jpg", bg_subtracted_image)
        #cv2.imwrite(curr_dir_session + "/scene_image.jpg", scene_image)

    #rospy.set_param('bool_True', "true")

    LOG_PATH = curr_dir_session + "/" + LOG_PATH
    SEARCH_DESINATION_DIR = curr_dir_session + "/" + SEARCH_DESINATION_DIR
    RETRIEVED_PATH= curr_dir_session + "/" + RETRIEVED_PATH
    UPRIGHT_PATH= curr_dir_session + "/" + UPRIGHT_PATH
    ROTATION_PATH= curr_dir_session + "/" + ROTATION_PATH
    try:
        os.makedirs(LOG_PATH)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        pass
    
    t0 = time.time()
    cache_time = t0-t0_cache

    fp = open(LOG_PATH+"/"+ datetime.datetime.now().strftime('%d%m%Y%H%M%S') + ".csv", 'w')
    ch = open(curr_dir_session+"/"+ "checklist.csv", 'ab')
    fp.truncate()
    ch.truncate()
    fpWriter = csv.writer(fp, delimiter='\t')
    chWriter = csv.writer(ch, delimiter='\t')
    fpWriter.writerow(["Date/Time", "Step", "Duration(s)"])
    if found_cache is False:
        chWriter.writerow(["Step", "Date/Time"])
    fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Cache checking", str(cache_time)])

    #Cloud API step
    if run_cloud_api_step:
        print("CLOUD API!")
        #content =Image.fromarray(QUERY_IMG)
        #content = QUERY_IMG.tobytes(None)
        ##For testing only
        #img_from_disk = (io.open('query_image.jpg', 'rb'))
        #content = img_from_disk.read()
        image_from_numpy = Image.fromarray(QUERY_IMG)
        imageBuffer = io.BytesIO()
        image_from_numpy.save(imageBuffer, format='PNG')
        imageBuffer.seek(0)
        #print(type(image_from_numpy))
        #exit(0)

        web_results = detect_image.report(detect_image.detect_web(imageBuffer))
        search_keyword = web_results[:4] #get only highest 5 results
        print (search_keyword)
        kw = open(curr_dir_session+"/"+ "keywords.txt", 'w')
        for item in search_keyword:
            kw.write("%s\n" % item)
        kw.close()

    #Image Download Step
    if run_image_download_step:
        print("Image download step")
        t0_step = time.time()
        download_images.download(search_keyword, keywords, SEARCH_DESINATION_DIR, download_limit, LOG_PATH)
        t1_step = time.time()
        step_time = t1_step-t0_step
        print("\nImage download: " + str(step_time))
        #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tImage download:\t\t\t" + str(step_time) + "\n")
        fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Image download", str(step_time)])
        chWriter.writerow(["Image download", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #retrieve_nsmallest_dist step
    print("Retrieve nsmallest distance step")

    try:
        os.makedirs(RETRIEVED_PATH)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        pass

    t0_step = time.time()
    image_retrieval.retrieve_nsmallest_dist(QUERY_IMG, SEARCH_DESINATION_DIR, RETRIEVED_PATH, N, DIST_TYPE, LOG_PATH, vgg, sess)
    t1_step = time.time()
    step_time = t1_step-t0_step
    print("\nRetrieve nsmallest distance: " + str(step_time))
    #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tRetrieve nsmallest distance:\t" + str(step_time) + "\n")
    fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Retrieve nsmallest distance", str(step_time)])
    chWriter.writerow(["Retrieve nsmallest distance", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #Upright classification step
    print("Upright classification step")

    try:
        os.makedirs(UPRIGHT_PATH)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..:
        pass

    t0_step = time.time()
    classify_image.classify(LABELS_PATH, GRAPH_PATH, RETRIEVED_PATH, UPRIGHT_PATH, LOG_PATH)
    t1_step = time.time()
    step_time = t1_step-t0_step
    print("\nUpright classification: " + str(step_time))
    #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tUpright classification:\t\t" + str(step_time) + "\n")
    fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Upright classification", str(step_time)])
    chWriter.writerow(["Upright classification", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #Image rotation step
    #The following indicates that the current working directory is availabe for further processing. Currently, only this step relies on it
    print("Image rotation step")
    try:
        os.makedirs(ROTATION_PATH)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        pass
    #t0_step = time.time()
    #image_rotation.find_rotation_matrix(cv_query_image, UPRIGHT_PATH, ROTATION_PATH, LOG_PATH)
    rospy.set_param('/visiobased_placement/cwd', cwd + "/" + curr_dir_session + "/")
    #TODO: write the rotation matrix to rotation.csv

    #t1_step = time.time()
    #step_time = t1_step-t0_step
    #print("\nImage rotation: " + str(step_time))
    #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tImage rotation:\t\t\t" + str(step_time) + "\n")
    #fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Image rotation", str(step_time)])
    #chWriter.writerow(["Image rotation", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #Record end time then log and print how long it took
    t1 = time.time()
    total = t1-t0

    print("\nTime taken: " + str(total))

    #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tTotal time taken:\t\t" + str(total) + "\n")

    fp.close()
    ch.close()
    #rospy.signal_shutdown("Object Placed. Check if anything is broken ;)")



if __name__ == '__main__':

    rospy.init_node('image_listener')
    rospack = rospkg.RosPack()
    #package_path = rospack.get_path('visiobased_object_placement')
    #paramlist = rosparam.load_file(package_path + "/params.yaml",default_namespace="visiobased_placement")
    paramlist = rosparam.load_file("params.yaml",default_namespace="visiobased_placement")
    for params, ns in paramlist:
        rosparam.upload_params(ns,params)

    # Define your image topic
    image_topic = "/bg_subtracted_image"
    # Set up your subscriber and define its callback
    sub_once = rospy.Subscriber(image_topic, msg.Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()