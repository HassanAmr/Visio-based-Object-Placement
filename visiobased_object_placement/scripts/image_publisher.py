#!/usr/bin/env python
# license removed for brevity
import rospy
from sensor_msgs import msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("-b", "--bgs_image", dest="bgs_image", default=None,
                    help="Image file location.", metavar="file_name")

parser.add_argument("-c", "--cropped_image", dest="cropped_image", default=None,
                    help="Image file location.", metavar="file_name")

args = parser.parse_args()

bridge = CvBridge()

def image_publisher():
    pub1 = rospy.Publisher('/bg_subtracted_image', msg.Image, queue_size=1)
    pub2 = rospy.Publisher('/cropped_image', msg.Image, queue_size=1)
    rospy.init_node('image_publisher', anonymous=True)
    try:
        image1 = cv2.imread(args.bgs_image, cv2.IMREAD_COLOR)
        image_message1 = bridge.cv2_to_imgmsg(image1, encoding="bgr8")
        image2 = cv2.imread(args.cropped_image, cv2.IMREAD_COLOR)
        image_message2 = bridge.cv2_to_imgmsg(image2, encoding="bgr8")
    except CvBridgeError as e:
        print(e)

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub1.publish(image_message1)
        pub2.publish(image_message2)
        rate.sleep()

if __name__ == '__main__':

    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass