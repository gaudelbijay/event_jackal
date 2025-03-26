import numpy as np
import rospy
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Bool
import math


class JackalRealRobot():
    def __init__(self, init_position=[0, 0, 0]):
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.collision_pub = rospy.Publisher('/collision', Bool, queue_size=10)

        self.prev_velocity = 0.0
        self.COLLISION_THRESHOLD_ACC = 200.0 

        self.collision_count = 0
        # Current robot pose
        self.current_pose = Odometry()

        # Subscribers
        rospy.Subscriber("/cmd_vel", Twist, self.vel_monitor)
        rospy.Subscriber('/event_camera', Float32MultiArray, self.event_callback, queue_size=10)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback, queue_size=10)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odometry_callback)
        rospy.Subscriber("/collision", Bool, self.collision_monitor)

        # Data
        self.default_event_data = np.zeros(512, dtype=np.float32)
        self.current_event_data = self.default_event_data.copy()

        self.bad_vel_count = 0
        self.vel_count = 0

    def vel_monitor(self, msg):
        velocity_magnitude = np.sqrt(msg.linear.x**2 + msg.linear.y**2 + msg.linear.z**2)
        self.bad_vel_count += (velocity_magnitude <= 0)
        self.vel_count += 1

    def get_bad_vel_num(self):
        bad_vel, vel = self.bad_vel_count, self.vel_count
        self.bad_vel_count = 0
        self.vel_count = 0
        return bad_vel, vel
    
    def collision_monitor(self, msg):
        self.collision_count += int(msg.data)

    def get_hard_collision(self):
        collided = self.collision_count > 0
        self.collision_count = 0
        return collided
    
    def imu_callback(self, msg):
        acc_x = msg.linear_acceleration.x
        acc_y = msg.linear_acceleration.y
        acc_z = msg.linear_acceleration.z
        acc_magnitude = math.sqrt(acc_x**2 + acc_y**2)

        if acc_magnitude > self.COLLISION_THRESHOLD_ACC:  # Use corrected name
            collision_msg = Bool()
            collision_msg.data = True
            rospy.logwarn("Collision detected due to high acceleration: %.2f", acc_magnitude)
            self.collision_pub.publish(collision_msg)

    def odometry_callback(self, msg):
        """Updates the current pose of the robot from odometry data."""
        # print(msg.pose)
        self.current_pose = msg#.pose

    def get_model_state(self):
        """Mimics Gazebo's get_model_state to return the current pose of the real robot."""
        return self.current_pose

    def event_callback(self, msg):
        self.current_event_data = np.array(msg.data)

    def get_events(self):
        return self.current_event_data
    
    @property
    def event_shape(self):
        return self.current_event_data.shape

    def publish_velocity(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self._cmd_vel_pub.publish(twist)


if __name__ == "__main__":
    rospy.init_node('jackal_real_robot_node', anonymous=True)
    robot = JackalRealRobot()
    # print(robot.get_events().shape)
    print(robot.get_model_state())
    rospy.spin()
