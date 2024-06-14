#SatyamRaina
#222196882
#!/usr/bin/env python3

import rospy
import math
from duckietown_msgs.msg import Twist2DStamped
from duckietown_msgs.msg import AprilTagDetectionArray

class TargetFollowerWithObstacleAvoidance:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node('target_follower_with_obstacle_avoidance_node', anonymous=True)

        # When shutdown signal is received, we run clean_shutdown function
        rospy.on_shutdown(self.clean_shutdown)

        # Initialize publisher and subscriber with your robot's name
        self.cmd_vel_pub = rospy.Publisher('/satyuvi/car_cmd_switch_node/cmd', Twist2DStamped, queue_size=1)
        rospy.Subscriber('/satyuvi/tag_detections', AprilTagDetectionArray, self.tag_callback, queue_size=1)

        # Initialize variables
        self.target_distance_threshold = 0.3  # Distance threshold for target following
        self.obstacle_distance_threshold = 0.2  # Distance threshold for obstacle avoidance
        self.front_dis = 0.0  # Distance to the front obstacle

        # Subscribe to the front ToF sensor
        rospy.Subscriber('/satyuvi/front_center_tof_driver_node/range', Range, self.range_callback, queue_size=1)

        rospy.spin()  # Spin forever but listen to message callbacks

    # April Tag Detection Callback
    def tag_callback(self, msg):
        self.move_robot(msg.detections)

    # Range Sensor Callback
    def range_callback(self, msg):
        self.front_dis = msg.range

    # Stop Robot before node has shut down. This ensures the robot keeps moving with the last velocity command
    def clean_shutdown(self):
        rospy.loginfo("System shutting down. Stopping robot...")
        self.stop_robot()

    # Sends zero velocity to stop the robot
    def stop_robot(self):
        cmd_msg = Twist2DStamped()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.v = 0.0
        cmd_msg.omega = 0.0
        self.cmd_vel_pub.publish(cmd_msg)

    def move_robot(self, detections):
        if len(detections) == 0:
            cmd_msg = Twist2DStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.v = 0
            cmd_msg.omega = 0.2  # Rotate in place if no target is detected
            self.cmd_vel_pub.publish(cmd_msg)
            return

        # Check for obstacles
        if self.front_dis < self.obstacle_distance_threshold:
            self.stop_robot()
            rospy.sleep(1)
            self.goal_angle(math.pi, 1.0)  # Turn 180 degrees to avoid obstacle
            return

        x = detections[0].transform.translation.x
        y = detections[0].transform.translation.y
        z = detections[0].transform.translation.z
        rospy.loginfo("x, y, z: %f, %f, %f", x, y, z)

        if abs(x) > self.target_distance_threshold:
            self.stop_robot()
            rospy.sleep(1)
            if x > 0:
                self.goal_angle(-math.pi / 2, 1.0)  # Turn left
            else:
                self.goal_angle(math.pi / 2, 1.0)  # Turn right
        else:
            cmd_msg = Twist2DStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.v = 0.2  # Move forward towards the target
            cmd_msg.omega = 0.0
            self.cmd_vel_pub.publish(cmd_msg)

    def goal_angle(self, angle, angular_speed):
        # Calculate time needed to rotate the desired angle at the given speed
        init_tick = self.tick_count
        target_ticks = (angle * self.wheel_base) / (2 * self.wheel_radius) * 100
        while abs(self.tick_count - init_tick) < target_ticks:
            cmd_msg = Twist2DStamped()
            cmd_msg.header.stamp = rospy.Time.now()
            cmd_msg.v = 0.0  # Straight line velocity
            cmd_msg.omega = angular_speed
            self.cmd_vel_pub.publish(cmd_msg)
            rospy.loginfo("Rotating!")
        self.stop_robot()

if __name__ == '__main__':
    try:
        target_follower = TargetFollowerWithObstacleAvoidance()
    except rospy.ROSInterruptException:
        pass
