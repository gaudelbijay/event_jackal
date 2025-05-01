#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import cv2
import math

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from e2d.model import DepthNetwork

# Expected image dimensions
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Torch device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path="/home/sas/event_ws/src/event_jackal/e2d/outputs/best_model_checkpoint.pth"):
    """
    Load the DepthNetwork model and its weights.
    """
    model = DepthNetwork(
        num_in_channels=1,
        num_out_channels=1,
        form_BEV=1,
        evs_min_cutoff=1e-3,
        embedding_dim=1024
    )
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("network_state_dict", checkpoint)
    model_dict = model.state_dict()
    # Filter only matching keys
    filtered = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(filtered)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    return model

# Load model once globally
model = load_model()


class JackalRealRobot:
    def __init__(self, init_position=[0, 0, 0]):
        rospy.loginfo("Starting JackalRealRobot node…")
        self.bridge = CvBridge()

        # Publishers
        self._cmd_vel_pub  = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.collision_pub = rospy.Publisher('/collision', Bool,  queue_size=10)

        # Subscribers
        rospy.Subscriber('/accumulator/image', Image,      self.event_frame_callback, queue_size=1)
        rospy.Subscriber('/imu/data',        Imu,        self.imu_callback,         queue_size=10)
        rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback,        queue_size=10)
        rospy.Subscriber('/cmd_vel',         Twist,      self.vel_monitor)
        rospy.Subscriber('/collision',       Bool,       self.collision_monitor)

        # Internal state
        self.current_pose      = Odometry()
        self.embedding         = np.zeros(1024, np.float32)
        self.collision_count   = 0
        self.bad_vel_count     = 0
        self.vel_count         = 0
        self.COLLISION_THRESHOLD_ACC = 200.0

    def vel_monitor(self, msg: Twist):
        """
        Monitor commanded velocities for zero-speed occurrences.
        """
        speed = math.hypot(msg.linear.x, msg.linear.y)
        self.bad_vel_count += int(speed <= 0)
        self.vel_count     += 1

    def get_bad_vel_num(self):
        """
        Return and reset counts of zero-speed commands over total.
        """
        bad, total = self.bad_vel_count, self.vel_count
        self.bad_vel_count = 0
        self.vel_count     = 0
        return bad, total

    def collision_monitor(self, msg: Bool):
        """
        Increment collision counter on collision topic.
        """
        self.collision_count += int(msg.data)

    def get_hard_collision(self):
        """
        Return and reset if any hard collision detected.
        """
        hit = (self.collision_count > 0)
        self.collision_count = 0
        return hit

    def imu_callback(self, msg: Imu):
        """
        Detect collisions via IMU acceleration spikes.
        """
        ax, ay, az = (msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z)
        mag = math.sqrt(ax*ax + ay*ay + az*az)
        if mag > self.COLLISION_THRESHOLD_ACC:
            rospy.logwarn("High accel collision: %.2f", mag)
            self.collision_pub.publish(Bool(data=True))

    def odom_callback(self, msg: Odometry):
        """
        Update current_pose from odometry.
        """
        self.current_pose = msg

    def get_model_state(self):
        """
        Return the latest odometry as model pose.
        """
        return self.current_pose

    def event_frame_callback(self, msg: Image):
        """
        Process mono8 event frames: denoise, binarize, infer embedding.
        """
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as e:
            rospy.logwarn("cv_bridge failed: %s", e)
            return

        # Sanity check
        h, w = cv_img.shape
        if (w, h) != (IMAGE_WIDTH, IMAGE_HEIGHT):
            rospy.logwarn(
                "Image shape %dx%d != expected %dx%d", w, h, IMAGE_WIDTH, IMAGE_HEIGHT
            )
            return

        # Binary threshold any pixel >1 -> 1
        frame = (cv_img > 1).astype(np.uint8)

        # Light denoise: 2×2 opening removes specks
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

        # Prune tiny blobs <3px
        MIN_PIXELS = 3
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for lab in range(1, n_labels):
            if stats[lab, cv2.CC_STAT_AREA] < MIN_PIXELS:
                mask[labels == lab] = 0

        # Prepare tensor [1×1×H×W]
        bin_frame = mask.astype(np.float32)
        tframe = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model(tframe, return_depth=False)
            self.embedding = emb.cpu().numpy()

        # Optional visualization
        vis = (frame * 255).astype(np.uint8)
        cv2.imshow("Event Frame (light denoise)", vis)
        cv2.waitKey(1)

    def get_embedding(self):
        """
        Return the latest embedding array.
        """
        return self.embedding

    @property
    def embedding_shape(self):
        """
        Shape of the embedding vector.
        """
        return self.embedding.shape

    def publish_velocity(self, linear_x: float, angular_z: float):
        """
        Publish velocity commands to the robot.
        """
        twist = Twist()
        twist.linear.x  = linear_x
        twist.angular.z = angular_z
        self._cmd_vel_pub.publish(twist)


if __name__ == '__main__':
    rospy.init_node('jackal_real_robot_node', anonymous=True)
    robot = JackalRealRobot(init_position=[0, 0, 0])
    rospy.loginfo("Jackal initial pose: %s", robot.get_model_state())
    rospy.spin()
