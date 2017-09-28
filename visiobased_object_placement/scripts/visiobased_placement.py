#!/usr/bin/env python

import rospy
import errno
import os
import datetime
import time
import csv
import image_retrieval
import classify_image
import download_images
import image_rotation
import image_crop
import ros_image_listener
import tensorflow as tf
import cv2 # OpenCV2 for saving an image


#Some of the following should be set in a roslaunch file
## download_images parameters
search_keyword = ['cheez-it', 'junk food']          #TODO: These should come from the API module.
keywords = ['']
download_limit = 20                                #TODO: impelement in download_images and get this number from the roslaunch file



WEIGHTS_PATH = '/home/hassan/Tools/data/vgg16_weights.npz'

GRAPH_PATH='/home/hassan/Tools/data/output_graph.pb'
LABELS_PATH='/home/hassan/Tools/data/output_labels.txt'


LOG_PATH='logs'
SEARCH_DESINATION_DIR = 'seach_results'
RETRIEVED_PATH='retrieved_results'
UPRIGHT_PATH='upright_results'
ROTATION_PATH='rotation_results'
#QUERY_IMG='query_image.jpeg'                                  #TODO: This should come from an image subscriber


CACHED_QUERY_FILE_NAME = "/query_image.jpeg"        #TODO: This should come from the roslaunch file settins

#TODO: The following 2 should come from the roslaunch file
N=10
DIST_TYPE='euc'
CACHE_THRESHOLD = 10

if __name__ == '__main__':


    t0_cache = time.time()

    bg_subtracted_image = "bg_subtracted_image.jpeg"
    scene_image = "scene_image.jpeg"
    QUERY_IMG = image_crop.crop(bg_subtracted_image)
    cv2.imwrite("cropped.jpg", QUERY_IMG)
    print ("done")
    time.sleep(10)
    #TODO: run ros_image_saver.py to set QUERY_IMG from it or from a input arg
    #ros_image_listener.main()
    #while (ros_image_listener.images_received is False):
    #    pass

    #QUERY_IMG = ros_image_listener.cv2_bg_image
    #ros_image_listener.image_listener_shutdown()

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
        test_image = curr_dir + CACHED_QUERY_FILE_NAME

        curr_dist = image_retrieval.retrieve_dist(QUERY_IMG, test_image, DIST_TYPE, vgg, sess)
        if curr_dist < min_dist:
            min_dist = curr_dist
            curr_dir_session = curr_dir
            found_cache = True

    #True is the default case for all steps unl
    run_cloud_api_step = False
    run_image_download_step = True
    run_retrieve_nsmallest_dist_step = True
    run_upright_classification_step = True
    run_image_rotation_step = True

    if found_cache:
        print(curr_dir_session)
        QUERY_IMG=curr_dir_session + CACHED_QUERY_FILE_NAME # we set it from the cached query_image since we want all the stats to correspond to the same exact image. 
        f = open(curr_dir_session + "/checklist.csv", 'rb')
        reader = csv.reader(f)
        for row in reader:
            print (row)
        #TODO: Check checklist and set the following accordingly
        run_cloud_api_step = False
        run_image_download_step = False
        run_retrieve_nsmallest_dist_step = False
        run_upright_classification_step = False
        run_image_rotation_step = False
        f.close()
    else:
        curr_dir_session = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
        try:
            os.makedirs(curr_dir_session)
        except OSError, e:
            raise  # Here we raise even when folder exists because it should not exist.

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
        search_keyword = detect_image.detect_web(QUERY_IMG)
        #TODO: Save results to file to cache it

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
    if run_retrieve_nsmallest_dist_step:
        print("Retrieve nsmallest distance step")
        t0_step = time.time()
        image_retrieval.retrieve_nsmallest_dist(QUERY_IMG, SEARCH_DESINATION_DIR, RETRIEVED_PATH, N, DIST_TYPE, LOG_PATH, vgg, sess)
        t1_step = time.time()
        step_time = t1_step-t0_step
        print("\nRetrieve nsmallest distance: " + str(step_time))
        #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tRetrieve nsmallest distance:\t" + str(step_time) + "\n")
        fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Retrieve nsmallest distance", str(step_time)])
        chWriter.writerow(["Retrieve nsmallest distance", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #Upright classification step
    if run_upright_classification_step:
        print("Upright classification step")
        t0_step = time.time()
        classify_image.classify(LABELS_PATH, GRAPH_PATH, RETRIEVED_PATH, UPRIGHT_PATH, LOG_PATH)
        t1_step = time.time()
        step_time = t1_step-t0_step
        print("\nUpright classification: " + str(step_time))
        #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tUpright classification:\t\t" + str(step_time) + "\n")
        fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Upright classification", str(step_time)])
        chWriter.writerow(["Upright classification", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #Image rotation step
    if run_image_rotation_step:
        print("Image rotation step")
        t0_step = time.time()
        image_rotation.find_rotation_matrix(QUERY_IMG, UPRIGHT_PATH, ROTATION_PATH, LOG_PATH)
        t1_step = time.time()
        step_time = t1_step-t0_step
        print("\nImage rotation: " + str(step_time))
        #TODO: write the rotation matrix to rotation.csv
        #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tImage rotation:\t\t\t" + str(step_time) + "\n")
        fpWriter.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), "Image rotation", str(step_time)])
        chWriter.writerow(["Image rotation", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())])

    #Record end time then log and print how long it took
    t1 = time.time()
    total = t1-t0

    print("\nTime taken: " + str(total))

    #fp.write(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + ":\tTotal time taken:\t\t" + str(total) + "\n")

    #TODO: Save the following inside curr_session_dir
    #bg_subtracted_image = "bg_subtracted_image.jpeg"
    #scene_image = "scene_image.jpeg"
    #QUERY_IMG =
    fp.close()
    ch.close()
