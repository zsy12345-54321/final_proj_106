#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from datetime import datetime
import message_filters

def synchronized_callback(rgb_msg, depth_msg):
    try:
        bridge = CvBridge()
        
        # Convert RGB Image
        cv_image_rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        filename_rgb = f'realsense_rgb_image.jpg'
        cv2.imwrite(filename_rgb, cv_image_rgb)
        rospy.loginfo(f"RGB Image saved as {filename_rgb}")
        cv2.imshow("RGB Image", cv_image_rgb)
        
        # Convert Depth Image
        if depth_msg.encoding == "16UC1":
            cv_image_depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            cv_image_depth = cv_image_depth/1000.0
        elif depth_msg.encoding == "32FC1":
            cv_image_depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")

        else:
            rospy.logwarn(f"Unsupported depth encoding: {depth_msg.encoding}")
            return
        
        filename_depth = f'realsense_depth_image.png'
        cv2.imwrite(filename_depth, cv_image_depth)
        rospy.loginfo(f"Depth Image saved as {filename_depth}")
        cv2.imshow("Depth Image", cv_image_depth)
        cv2.waitKey(1)
        
        # Shutdown after capturing both images
        rospy.signal_shutdown("Captured both images, shutting down.")
        
    except Exception as e:
        rospy.logerr(f"Error processing synchronized images: {e}")

def main():
    rospy.init_node('realsense_image_capture', anonymous=True)
    bridge = CvBridge()
    
    # Define topic names
    rgb_topic = "/camera/color/image_raw"
    depth_topic = "/camera/depth/image_rect_raw"
    
    # Initialize Subscribers with message_filters
    rgb_sub = message_filters.Subscriber(rgb_topic, Image)
    depth_sub = message_filters.Subscriber(depth_topic, Image)
    
    # Synchronize the topics
    ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub], queue_size=10, slop=0.1)
    ts.registerCallback(synchronized_callback)
    
    rospy.loginfo("Waiting for synchronized RGB and Depth images...")
    rospy.spin()
    
    # Clean up OpenCV windows
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

