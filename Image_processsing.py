#SatyamRaina
#222196882
#!/usr/bin/env python3

# Python Libs
import sys
import time
import random

# numpy
import numpy as np

# OpenCV
import cv2
from cv_bridge import CvBridge

# ROS Libraries
import rospy
import roslib

# ROS Message Types
from sensor_msgs.msg import CompressedImage

class LaneDetectorWithColorFilter:
    def __init__(self):
        self.cv_bridge = CvBridge()

        # Subscribing to the image topic
        self.image_sub = rospy.Subscriber('/satyuvi/camera_node/image/compressed', CompressedImage, self.image_callback, queue_size=1)
        
        rospy.init_node("my_lane_detector_with_color_filter")

        # Initialize color filter parameters
        self.hue_min = 0
        self.hue_max = 179
        self.sat_min = 0
        self.sat_max = 255
        self.val_min = 0
        self.val_max = 255

        # Initialize color filter update rate
        self.update_rate = 0.1  # Update color filter every 0.1 seconds

        # Initialize last update time
        self.last_update_time = rospy.get_time()

    def image_callback(self, msg):
        rospy.loginfo("image_callback")

        # Convert compressed image message to OpenCV image
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, "bgr8")

        ## Crop the input image to show only the road
        top = 200
        bottom = 400
        left = 100
        right = 500
        cropped_img = img[top:bottom, left:right]

        ## Convert the cropped image to HSV Color Space
        hsv_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

        ## Apply dynamic color filtering
        current_time = rospy.get_time()
        if current_time - self.last_update_time > self.update_rate:
            self.update_color_filter()
            self.last_update_time = current_time

        lower_color = np.array([self.hue_min, self.sat_min, self.val_min])
        upper_color = np.array([self.hue_max, self.sat_max, self.val_max])
        color_mask = cv2.inRange(hsv_img, lower_color, upper_color)

        ## Apply Canny Edge Detector to the cropped image
        edges = cv2.Canny(cropped_img, 50, 150)

        ## Apply Hough Transform to the Color-filtered image
        color_lines = self.apply_hough_transform(color_mask)

        ## Draw lines found on Hough Transform on the cropped image
        hough_img = cropped_img.copy()
        self.draw_lines(hough_img, color_lines)

        ## Convert the processed image back to RGB Color Space
        processed_img = cv2.cvtColor(hough_img, cv2.COLOR_BGR2RGB)

        # Display the images in separate windows
        cv2.imshow('trim', cropped_img)
        cv2.imshow('ColorMask', color_mask)
        cv2.imshow('Original', img)
        cv2.imshow('Hough Transforms', processed_img)
        cv2.waitKey(1)

    def apply_hough_transform(self, img):
        # Apply Hough Transform
        lines = cv2.HoughLinesP(img, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=50)
        return lines

    def draw_lines(self, img, lines):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def update_color_filter(self):
        # Update color filter parameters randomly
        self.hue_min = random.randint(0, 179)
        self.hue_max = random.randint(0, 179)
        self.sat_min = random.randint(0, 255)
        self.sat_max = random.randint(0, 255)
        self.val_min = random.randint(0, 255)
        self.val_max = random.randint(0, 255)

        rospy.loginfo("Updated color filter: H(%d, %d), S(%d, %d), V(%d, %d)", self.hue_min, self.hue_max, self.sat_min, self.sat_max, self.val_min, self.val_max)

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    try:
        lane_detector_instance = LaneDetectorWithColorFilter()
        lane_detector_instance.run()
        
    except rospy.ROSInterruptException:
        pass
