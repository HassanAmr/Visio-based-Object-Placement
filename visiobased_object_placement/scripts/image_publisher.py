#!/usr/bin/env python
# license removed for brevity
import rospy
from sensor_msgs import msg
import cv2
from cv_bridge import CvBridge, CvBridgeError
import argparse

parser = argparse.ArgumentParser()


parser.add_argument("-i", "--image", dest="user_image", default=None,
                    help="Image file location.", metavar="file_name")

args = parser.parse_args()

bridge = CvBridge()

def image_publisher():
    pub = rospy.Publisher('/bg_subtracted_image', msg.Image, queue_size=1)
    rospy.init_node('image_publisher', anonymous=True)
    try:
        image = cv2.imread(args.user_image, cv2.IMREAD_COLOR)
        image_message = bridge.cv2_to_imgmsg(image, encoding="bgr8")
    except CvBridgeError as e:
        print(e)

    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        pub.publish(image_message)
        rate.sleep()

if __name__ == '__main__':

    try:
        image_publisher()
    except rospy.ROSInterruptException:
        pass