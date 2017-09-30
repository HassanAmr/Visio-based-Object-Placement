#!/usr/bin/env python


########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################
# Kent Sommer, 2016                                                                    #
# Modified for Image retrieval on the UKBenchmark image set using fc2 features         #
########################################################################################

import rospy
import tensorflow as tf
import numpy as np
import scipy.spatial.distance
import errno
from os import listdir
from os.path import join
from os.path import basename
from shutil import copyfile
from scipy.misc import imread, imresize
from math import*
import heapq
import time
import cv2
import csv



class vgg16:
    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.convlayers()
        self.fc_layers()
        self.probs = tf.nn.softmax(self.fc3l)
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)


    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                                     stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fc1w = tf.Variable(tf.truncated_normal([shape, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc1b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.name_scope('fc2') as scope:
            fc2w = tf.Variable(tf.truncated_normal([4096, 4096],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc2b = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3
        with tf.name_scope('fc3') as scope:
            fc3w = tf.Variable(tf.truncated_normal([4096, 1000],
                                                         dtype=tf.float32,
                                                         stddev=1e-1), name='weights')
            fc3b = tf.Variable(tf.constant(1.0, shape=[1000], dtype=tf.float32),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        print ('Load weights...')
        for i, k in enumerate(keys):
            sess.run(self.parameters[i].assign(weights[k]))
        print ('Load complete.')

    def square_rooted(x):
        return round(sqrt(sum([a*a for a in x])),3)

    def cosine_similarity(x,y):
        numerator = sum(a*b for a,b in zip(x,y))
        denominator = square_rooted(x)*square_rooted(y)
        return round(numerator/float(denominator),3)

def retrieve_nsmallest_dist(query_image, test_dirs, out_dir, n, dist_type, weights_path, log_dir):
    """This function will compare a query image against all images provided in the test_dir, and save/log the smallest n images to the out_dir

    Args:
        query_path (str):   Query image object
        test_dirs (str):    List of directories containing the test images
        out_dir (str):      Location to the save the retrieved images
        n (int):            Number of images to retrieve
        dist_type (str):    The distance algorithm (euc, cos, chev)
        weights_path (str): CNN weights path
        log_path (str):     Path of the logs directory to save the logs in

    """


    # Setup Session
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, weights_path, sess)
    # Get number of images to match (default 4)
    #dist_type = raw_input("Enter distance algorithm (euc, cos, chev): \n") or "euc"
    #print("distance type selected: " + dist_type)
    #dist_type="euc"

    

    ####################
    ###Perform Search###
    ####################

    #Timer and precision count total + open file for saving data
    t0 = time.time()
    feat_dict = {}
    #fp = open("Last_Run.txt", 'w')
    #fp.truncate()

    # Retrieve feature vector for query image 
    #Setup Dict and precision tracking
    img_dict = {}
    #img_query = imread(query_path)
    img_query = imresize(query_image, (224, 224))

    # Extract image descriptor in layer fc2/Relu. If you want, change fc2 to fc1
    layer_query = sess.graph.get_tensor_by_name('fc2/Relu:0')
    # layer_query = sess.graph.get_tensor_by_name('fc1/Relu:0')
    # Run the session for feature extract at 'fc2/Relu' layer
    _feature_query = sess.run(layer_query, feed_dict={vgg.imgs: [img_query]})
    # Convert tensor variable into numpy array
    # It is 4096 dimension vector
    feature_query = np.array(_feature_query)
    print (test_dirs)
    for test_dir in test_dirs:
        print (test_dir)
        # Set dataset directory path
        data_dir_test = test_dir
        datalist_test = [join(data_dir_test, f) for f in listdir(data_dir_test)]

        # Retrieve feature vector for test image
        for j in datalist_test:
            try:
                img_test = imread(j)
            except:
                continue
            img_test = cv2.cvtColor(img_test, cv2.COLOR_BGRA2BGR)
            img_test = imresize(img_test, (224, 224))

            # Extract image descriptor in layer fc2/Relu. If you want, change fc2 to fc1
            layer_test = sess.graph.get_tensor_by_name('fc2/Relu:0')
            # layer_test = sess.graph.get_tensor_by_name('fc1/Relu:0')
            # Run the session for feature extract at 'fc2/Relu' layer
            _feature_test = sess.run(layer_test, feed_dict={vgg.imgs: [img_test]})
            # Convert tensor variable into numpy array
            # It is 4096 dimension vector
            feature_test = np.array(_feature_test)
            feat_dict[j] = feature_test

            # Calculate Euclidean distance between two feature vectors
            if dist_type == "euc":
                curr_dist = scipy.spatial.distance.euclidean(feature_query, feature_test)
            # Calculate Cosine distance between two feature vectors
            if dist_type == "cos":
                curr_dist = scipy.spatial.distance.cosine(feature_query, feature_test)
            # Calculate Chevyshev distance between two feature vectors
            if dist_type == "chev":
                curr_dist = scipy.spatial.distance.chebyshev(feature_query, feature_test)

            # Add to dictionary
            img_dict[curr_dist] = str(j)

    fp = open(log_dir+"/"+ "retrieval_log.csv", 'w')
    fp.truncate()
    fpWriter = csv.writer(fp, delimiter='\t')
    fpWriter.writerow(["File", "Distance"])
    # Get Results for Query 
    keys_sorted = heapq.nsmallest(n, img_dict)
    for y in range(0,n):
        print(str(y+1) + ":\t" + "Distance: " + str(keys_sorted[y]) + ", FileName: " + basename(img_dict[keys_sorted[y]]))
        #fp.write(str(y+1) + ":\t" + "Distance: " + str(keys_sorted[y]) + ", FileName: " + basename(img_dict[keys_sorted[y]]) +"\n")
        fpWriter.writerow([basename(img_dict[keys_sorted[y]]), str(keys_sorted[y])])
        #basename(img_dict[keys_sorted[y]])
        copyfile(img_dict[keys_sorted[y]], out_dir + "/" + basename(img_dict[keys_sorted[y]]))

    t1 = time.time()
    total = t1-t0
    #print("\nTime taken: " + str(total))
    #fpWriter.writerow([])
    #fp.write("\n\nTime taken: " + str(total) + "\n")
    fp.close()


def retrieve_dist(query_image, test_path, dist_type, weights_path):
    """This function will compare a query image against all images provided in the test_dir, and save/log the smallest n images to the out_dir

    Args:
        query_image (np.darray):    Query image path
        test_dir (str):             Location of the test images
        dist_type (str):            The distance algorithm (euc, cos, chev)
        vgg (class vgg16):          Object of the vgg16 class
        sess(tf Session):           Object of the tensorflow session

    """

    # Setup Session
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, weights_path, sess)
    # Get number of images to match (default 4)
    #dist_type = raw_input("Enter distance algorithm (euc, cos, chev): \n") or "euc"
    #print("distance type selected: " + dist_type)
    #dist_type="euc"

    ####################
    ###Perform Search###
    ####################

    #Timer and precision count total + open file for saving data
    t0 = time.time()
    feat_dict = {}
    #fp = open("Last_Run.txt", 'w')
    #fp.truncate()

    # Retrieve feature vector for query image 
    #Setup Dict and precision tracking
    img_dict = {}
    #img_query = imread(query_path)
    img_query = imresize(query_image, (224, 224))
    img_test = imread(test_path)
    img_test = imresize(img_test, (224, 224))

    # Extract image descriptor in layer fc2/Relu. If you want, change fc2 to fc1
    layer_query = sess.graph.get_tensor_by_name('fc2/Relu:0')
    # layer_query = sess.graph.get_tensor_by_name('fc1/Relu:0')
    # Run the session for feature extract at 'fc2/Relu' layer
    _feature_query = sess.run(layer_query, feed_dict={vgg.imgs: [img_query]})
    # Convert tensor variable into numpy array
    # It is 4096 dimension vector
    feature_query = np.array(_feature_query)

    # Extract image descriptor in layer fc2/Relu. If you want, change fc2 to fc1
    layer_test = sess.graph.get_tensor_by_name('fc2/Relu:0')
    # layer_test = sess.graph.get_tensor_by_name('fc1/Relu:0')
    # Run the session for feature extract at 'fc2/Relu' layer
    _feature_test = sess.run(layer_test, feed_dict={vgg.imgs: [img_test]})
    # Convert tensor variable into numpy array
    # It is 4096 dimension vector
    feature_test = np.array(_feature_test)

    # Calculate Euclidean distance between two feature vectors
    if dist_type == "euc":
        curr_dist = scipy.spatial.distance.euclidean(feature_query, feature_test)
    # Calculate Cosine distance between two feature vectors
    if dist_type == "cos":
        curr_dist = scipy.spatial.distance.cosine(feature_query, feature_test)
    # Calculate Chevyshev distance between two feature vectors
    if dist_type == "chev":
        curr_dist = scipy.spatial.distance.chebyshev(feature_query, feature_test)

    return curr_dist
