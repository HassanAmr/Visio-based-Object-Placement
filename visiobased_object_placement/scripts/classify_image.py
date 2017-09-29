#!/usr/bin/env python


# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Simple image classification with Inception.

Run image classification with your model.

This script is usually used with retrain.py found in this same
directory.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. You are required
to pass in the graph file and the txt file.

It outputs human readable strings of the top 5 predictions along with
their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Example usage:
python label_image.py --graph=retrained_graph.pb
  --labels=retrained_labels.txt
  --image=flower_photos/daisy/54377391_15648e8d18.jpg

NOTE: To learn to use this file and retrain.py, please see:

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import rospy
import errno
import csv
from os import listdir
from os.path import join
from os.path import basename
from shutil import copyfile
from scipy.misc import imread

import tensorflow as tf


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    node_id = top_k[0]
    score = predictions[node_id]
    return [node_id, score]
    #print (top_k)
    #print (labels)
    #for node_id in top_k:
      #human_string = labels[node_id]
      #score = predictions[node_id]
      #print(node_id)
      #print('%s (score = %.5f)' % (human_string, score))



def classify(labels_path, graph_path, in_dir, out_dir, log_dir):
  """Runs inference on an image."""
  # load image

  # load labels
  labels = load_labels(labels_path)

  # load graph, which is stored in the default session
  load_graph(graph_path)

  # Set dataset directory path
  data_dir_test = in_dir
  datalist_test = [join(data_dir_test, f) for f in listdir(data_dir_test)]
  if datalist_test > 0:
    fp = open(log_dir+"/"+ "classifier_log.csv", 'w')
    fp.truncate()
    fpWriter = csv.writer(fp, delimiter='\t')
    fpWriter.writerow(["File", "Classification"])

  for j in datalist_test:
    #print(j)
    try:
      image_data = load_image(j)
      [node_id, score] = run_graph(image_data, labels, 'DecodeJpeg/contents:0', 'final_result:0', 1)
    except:
      continue
    human_string = labels[node_id]
    #fp.write(j + "\t"+ "|" + "\t" + "%s (score = %.5f)\n" % (human_string, score))
    fpWriter.writerow([j, "%s (score = %.5f)"% (human_string, score)])

    if (labels[node_id] == labels[0]):#labels[0] is the upright class
      copyfile(j, out_dir + "/" + basename(j))
  
  if datalist_test > 0:
    fp.close()

#if __name__ == '__main__':

  #tf.app.run(main=main, argv=sys.argv[:1]+unparsed)