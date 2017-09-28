#! /usr/bin/python
# Copyright (c) 2015, Rethink Robotics, Inc.

# Using this CvBridge Tutorial for converting
# ROS images to OpenCV2 images
# http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython

# Using this OpenCV2 tutorial for saving Images:
# http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_image_display/py_image_display.html

# rospy for the subscriber
import rospy
# ROS Image message
import message_filters
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
import cv2 # OpenCV2 for saving an image


# Instantiate CvBridge
bridge = CvBridge()

def callback(bg_image, scene_image):
    # Solve all of perception here...
    try:
        # Convert your ROS Image message to OpenCV2
        #if images_received is False:
        cv2_bg_image = bridge.imgmsg_to_cv2(bg_image, "bgr8")
        cv2_scene_image = bridge.imgmsg_to_cv2(scene_image, "bgr8")
        #images_received = True

    except CvBridgeError, e:
        print(e)
    else:
            #return [cv2_bg_image, cv2_scene_image]
            cv2.imwrite('bg_subtracted_image.jpg', cv2_bg_image)
            cv2.imwrite('scene_image.jpg', cv2_scene_image)
            rospy.signal_shutdown("Images saved.")



def main():
    rospy.init_node('image_listener')
    # Define your image topic
    bg_image_topic = "/bg_subtracted_image"
    scene_image_topic = "/kinect2/hd/image_color"

    bg_image_sub = message_filters.Subscriber(bg_image_topic, Image)
    scene_image_sub = message_filters.Subscriber(scene_image_topic, Image)
    ts = message_filters.ApproximateTimeSynchronizer([bg_image_sub, scene_image_sub], 10, 1)
    ts.registerCallback(callback)

    # Spin until ctrl + c
    rospy.spin()

if __name__ == '__main__':
    main()