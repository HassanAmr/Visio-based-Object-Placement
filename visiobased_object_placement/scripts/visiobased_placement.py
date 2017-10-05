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


parser.add_argument("--force_download", 
                    help="A flag to re-download images from the web", action="store_true")

parser.add_argument("-k", "--keywords", dest="user_search_keyword", default=None,
                    help="File containing keywords defined by the user.", metavar="file_name")

args = parser.parse_args()

bridge = CvBridge()

def image_callback(msg):

    CACHED_QUERY_FILE_NAME = rospy.get_param("/visiobased_placement/CACHED_QUERY_FILE_NAME")
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
    force_download = rospy.get_param("/visiobased_placement/force_download")
    user_search_keyword_path = rospy.get_param("/visiobased_placement/user_search_keyword")
    #search_keyword = rospy.get_param("/visiobased_placement/search_keyword") #TODO: This should be removed once API is online
    
    #t0_cache = time.time()
    t0 = time.time()
    
    #Set api flag according to params
    if user_search_keyword_path is not "":
        run_cloud_api_step = False
    else:
        run_cloud_api_step = True

    
    try:
        os.makedirs(SEARCH_DESINATION_DIR)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        pass

    cwd = os.getcwd()
    cached_search_dir = cwd + "/" + SEARCH_DESINATION_DIR
    dirs_list = [ x for x in os.listdir(cached_search_dir) if os.path.isdir(cached_search_dir + "/" + x) ]
    #dirs_list = [ x for x in os.listdir(cached_search_dir)]

    curr_dir_session = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
    try:
        os.makedirs(curr_dir_session)
    except OSError, e:
        raise  # Here we raise even when folder exists because it should not exist.

    #cv2.imwrite(curr_dir_session + "/scene_image.jpg", scene_image)

    #rospy.set_param('bool_True', "true")

    LOG_PATH = curr_dir_session + "/" + LOG_PATH
    RETRIEVED_PATH= curr_dir_session + "/" + RETRIEVED_PATH
    UPRIGHT_PATH= curr_dir_session + "/" + UPRIGHT_PATH
    ROTATION_PATH= curr_dir_session + "/" + ROTATION_PATH

    try:
        os.makedirs(LOG_PATH)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        pass

    fp = open(LOG_PATH+"/main_log.csv", 'w')
    #ch = open(curr_dir_session+"/"+ "checklist.csv", 'ab')
    fp.truncate()
    #ch.truncate()
    fpWriter = csv.writer(fp, delimiter='\t')
    #chWriter = csv.writer(ch, delimiter='\t')
    fpWriter.writerow(["Date/Time", "Step", "Duration(s)"])
    
    #t0 = time.time()
    #cache_time = t0-t0_cache
    #fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Cache checking", str(cache_time)])

    ###Begin###
    #Begin pipeline

    try:
        # Convert your ROS Image message to OpenCV2
        bg_subtracted_image = bridge.imgmsg_to_cv2(msg, "bgr8")
        sub_once.unregister()
    except CvBridgeError, e:
        print(e)

    cv_query_image = image_crop.crop(bg_subtracted_image)
    #cv2.imwrite("query_image.jpg", cv_query_image)
    QUERY_IMG = np.asarray( cv2.cvtColor(cv_query_image[:,:], cv2.COLOR_BGR2RGB) )


    #Cloud API step

    if run_cloud_api_step is False:
        search_keyword = []
        f = open(user_search_keyword_path, 'rb')
        reader = csv.reader(f)
        for row in reader:
            search_keyword.append(row[0])
        print(search_keyword)



    if run_cloud_api_step:
        print("CLOUD API!")
        t0_step = time.time()

        image_from_numpy = Image.fromarray(QUERY_IMG)
        imageBuffer = io.BytesIO()
        image_from_numpy.save(imageBuffer, format='PNG')
        imageBuffer.seek(0)


        web_results = detect_image.report(detect_image.detect_web(imageBuffer))
        search_keyword = web_results[:4] #get only highest 4 results
        t1_step = time.time()
        step_time = t1_step-t0_step
        print("\nCloud API: " + str(step_time))
        fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Image download", str(step_time)])
        print (search_keyword)



    #Check if the next step should be done completely or not
    run_reduced_download_step = False
    if force_download:
        run_image_download_step = True
    else:
        print("Cached Web Search:")
        print(dirs_list)
        #we still want to save search_keyword to write it to disk at the end, so we will copy what is not cached to a new list
        search_keyword_checklist = [x for x in search_keyword if x not in dirs_list]
        print("Reduced keywords:")
        print(search_keyword_checklist)

        if not search_keyword_checklist:
            run_image_download_step = False
        else:
            run_image_download_step = True
            run_reduced_download_step = True

    #Image Download Step
    if run_image_download_step:
        print("Image download step")

        t0_step = time.time()
        if run_reduced_download_step:
            download_images.download(search_keyword_checklist, keywords, SEARCH_DESINATION_DIR, download_limit, LOG_PATH)
        else:
            download_images.download(search_keyword, keywords, SEARCH_DESINATION_DIR, download_limit, LOG_PATH)
        t1_step = time.time()
        step_time = t1_step-t0_step
        print("\nImage download: " + str(step_time))
        #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tImage download:\t\t\t" + str(step_time) + "\n")
        fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Image download", str(step_time)])
        #chWriter.writerow(["Image download", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #The following is commented because we have it already
    #cached_search_dir = cwd + "/" + SEARCH_DESINATION_DIR
    #print (cached_search_dir)
    dirs_list = [ cached_search_dir + "/" + x for x in os.listdir(cached_search_dir) if os.path.isdir(cached_search_dir + "/" + x) ]
    #retrieve_nsmallest_dist step
    print("Retrieve nsmallest distance step")

    try:
        os.makedirs(RETRIEVED_PATH)
    except OSError, e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
        pass
    
    t0_step = time.time()
    image_retrieval.retrieve_nsmallest_dist(QUERY_IMG, dirs_list, RETRIEVED_PATH, N, DIST_TYPE, WEIGHTS_PATH, LOG_PATH)
    t1_step = time.time()
    step_time = t1_step-t0_step
    print("\nRetrieve nsmallest distance: " + str(step_time))
    #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tRetrieve nsmallest distance:\t" + str(step_time) + "\n")
    fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Retrieve nsmallest distance", str(step_time)])
    #chWriter.writerow(["Retrieve nsmallest distance", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

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
    #chWriter.writerow(["Upright classification", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #Image rotation step
    #The following indicates that the current working directory is availabe for further processing. Currently, only this step relies on it

    cv2.imwrite(curr_dir_session + "/query_image.jpg", cv_query_image)


    print("\nImage rotation step...")
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

    #Do the writings at the end to finish above steps asap
    #Writing keywords to file
    kw = open(curr_dir_session+"/"+ "keywords.txt", 'w')
    for item in search_keyword:
        kw.write("%s\n" % item)
    kw.close()
    

    #save raw bg_subtracte_image to file
    cv2.imwrite(curr_dir_session + "/bg_subtracted_image.jpg", bg_subtracted_image)


    #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tTotal time taken:\t\t" + str(total) + "\n")

    fp.close()
    #ch.close()
    #rospy.signal_shutdown("Object Placed. Check if anything is broken ;)")

if __name__ == '__main__':

    rospy.init_node('image_listener')
    rospack = rospkg.RosPack()
    #package_path = rospack.get_path('visiobased_object_placement')
    #paramlist = rosparam.load_file(package_path + "/params.yaml",default_namespace="visiobased_placement")
    paramlist = rosparam.load_file("params.yaml",default_namespace="visiobased_placement")

    for params, ns in paramlist:
        rosparam.upload_params(ns,params)

    rospy.set_param('/visiobased_placement/force_download', args.force_download)

    if args.user_search_keyword is not None:
        rospy.set_param('/visiobased_placement/user_search_keyword', args.user_search_keyword)
    else:
        rospy.set_param('/visiobased_placement/user_search_keyword', "")


    # Define your image topic
    image_topic = "/bg_subtracted_image"
    # Set up your subscriber and define its callback
    sub_once = rospy.Subscriber(image_topic, msg.Image, image_callback)
    # Spin until ctrl + c
    rospy.spin()