#!/usr/bin/env python


import rospy
import os
import time
import image_retrieval
import classify_image
import download_images


#Some of the following should be set in a roslaunch file
## download_images parameters
search_keyword = ['cheez-it', 'junk food']          #TODO: These should come from the API module.
keywords = ['']
download_limit = 100                                #TODO: impelement in download_images and get this number from the roslaunch file



WEIGHTS_PATH = '/home/hassan/Tools/data/vgg16_weights.npz'

GRAPH_PATH='/home/hassan/Tools/data/output_graph.pb'
LABELS_PATH='/home/hassan/Tools/data/output_labels.txt'


LOG_PATH='logs'
SEARCH_DESINATION_DIR = 'seach_results'
RETRIEVED_PATH='retrieved_results'
UPRIGHT_PATH='upright_results'
QUERY_IMG='18.jpg'                                  #TODO: This should come from an image subscriber

#TODO: The following 2 should come from the roslaunch file
N=20
DIST_TYPE='euc'

if __name__ == '__main__':
    try:
        os.makedirs(LOG_PATH)
    except OSError, e:
        if e.errno != 17:
            raise
        pass
    t0 = time.time()

    download_images.download(search_keyword, keywords, SEARCH_DESINATION_DIR, download_limit, LOG_PATH)
    image_retrieval.retrieve_nsmallest_dist(QUERY_IMG, SEARCH_DESINATION_DIR, RETRIEVED_PATH, N, DIST_TYPE, WEIGHTS_PATH, LOG_PATH)

    classify_image.classify(LABELS_PATH, GRAPH_PATH, RETRIEVED_PATH, UPRIGHT_PATH, LOG_PATH)

    #Record end time then log and print how long it took
    t1 = time.time()
    total = t1-t0
    fp = open(LOG_PATH+"/"+ "main_log.txt", 'w')
    fp.truncate()
    print("\nTime taken: " + str(total))
    fp.write("\n\nTime taken: " + str(total) + "\n")
    fp.close()

