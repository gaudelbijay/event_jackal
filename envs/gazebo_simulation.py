#!/usr/bin/env python3
import numpy as np
import rospy
import torch
import cv2

from std_srvs.srv import Empty
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState
from geometry_msgs.msg import Quaternion, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, UInt8MultiArray
from e2d.model import DepthNetwork

# Image dimensions
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model(checkpoint_path="/home/sas/event_ws/src/event_jackal/e2d/outputs/best_model_checkpoint.pth"):
    # Use num_in_channels=1 and form_BEV=1 to match the training configuration.
    model = DepthNetwork(num_in_channels=1, num_out_channels=1, form_BEV=1,
                           evs_min_cutoff=1e-3, embedding_dim=1024)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # If checkpoint is a training checkpoint, it likely has a key 'network_state_dict'
    if isinstance(checkpoint, dict) and "network_state_dict" in checkpoint:
        state_dict = checkpoint["network_state_dict"]
    else:
        state_dict = checkpoint

    # Filter state_dict: only keep keys that are in the model.
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for key in model_state_dict.keys():
        if key in state_dict:
            new_state_dict[key] = state_dict[key]
        else:
            print(f"Warning: Missing key in checkpoint: {key}")
    
    # Load the filtered state_dict into the model.
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

model = load_model()

def create_model_state(x, y, z, angle):
    """
    Create a ModelState object for the 'jackal' model with the given position and orientation.
    """
    state = ModelState()
    state.model_name = 'jackal'
    state.pose.position.x = x
    state.pose.position.y = y
    state.pose.position.z = z
    # Calculate quaternion for a rotation about the Z axis
    state.pose.orientation = Quaternion(0, 0, np.sin(angle / 2.0), np.cos(angle / 2.0))
    state.reference_frame = "world"
    return state


class GazeboSimulation:
    def __init__(self, init_position=[0, 0, 0]):
        # Setup ROS service proxies
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._reset = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self._model_state_getter = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

        # Initialize the model state
        self._init_model_state = create_model_state(init_position[0], init_position[1], 0, init_position[2])

        # Initialize counters and embedding storage
        self.collision_count = 0
        self.bad_vel_count = 0
        self.vel_count = 0
        self.embedding = np.zeros(1024, np.float32)

        # Setup ROS subscribers
        rospy.Subscriber("/collision", Bool, self.collision_monitor)
        rospy.Subscriber("/jackal_velocity_controller/cmd_vel", Twist, self.vel_monitor)
        rospy.Subscriber('/output/event', UInt8MultiArray, self.event_frame_callback, queue_size=2)

    def vel_monitor(self, msg):
        """
        Monitor velocity commands and update counters.
        """
        velocity = np.sqrt(msg.linear.x**2 + msg.linear.y**2 + msg.linear.z**2)
        self.bad_vel_count += (velocity <= 0)
        self.vel_count += 1

    def get_bad_vel_num(self):
        """
        Return the number of bad velocity commands along with total commands, then reset counts.
        """
        bad_vel, total_vel = self.bad_vel_count, self.vel_count
        self.bad_vel_count = 0
        self.vel_count = 0
        return bad_vel, total_vel

    def collision_monitor(self, msg):
        """
        Update collision count based on collision messages.
        """
        self.collision_count += int(msg.data)

    def get_hard_collision(self):
        """
        Check if a hard collision occurred and reset the collision counter.
        """
        collided = self.collision_count > 0
        self.collision_count = 0
        return collided

    def pause(self):
        """
        Pause the Gazebo simulation.
        """
        try:
            self._pause()
        except rospy.ServiceException:
            rospy.logwarn("/gazebo/pause_physics service call failed")

    def unpause(self):
        """
        Unpause the Gazebo simulation.
        """
        try:
            self._unpause()
        except rospy.ServiceException:
            rospy.logwarn("/gazebo/unpause_physics service call failed")

    def reset(self):
        """
        Reset the model state in the Gazebo simulation.
        """
        try:
            self._reset(self._init_model_state)
        except rospy.ServiceException:
            rospy.logwarn("/gazebo/set_model_state service call failed")

    def get_laser_scan(self):
        """
        Wait for a laser scan message from the 'front/scan' topic.
        """
        try:
            return rospy.wait_for_message('front/scan', LaserScan, timeout=1)
        except rospy.ROSException:
            return None

    def get_model_state(self):
        """
        Retrieve the current model state from Gazebo.
        """
        try:
            return self._model_state_getter('jackal', 'world')
        except rospy.ServiceException:
            rospy.logwarn("/gazebo/get_model_state service call failed")

    def get_embedding(self):
        """
        Get the most recently computed embedding.
        """
        return self.embedding

    @property
    def embedding_shape(self):
        """
        Return the shape of the current embedding.
        """
        return self.embedding.shape

    def reset_init_model_state(self, init_position=[0, 0, 0]):
        """
        Reset the initial model state with a new position.
        """
        self._init_model_state = create_model_state(init_position[0], init_position[1], 0, init_position[2])

    def event_frame_callback(self, msg):
        """
        Callback for processing an event frame message.
        Converts the incoming message into an image, processes it with the model, and displays the depth output.
        """
        data = np.frombuffer(msg.data, dtype=np.uint8)
        frame = data.reshape((IMAGE_HEIGHT, IMAGE_WIDTH)).astype(np.float32)

        # frame_plot = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        # cv2.imshow("depth", frame_plot)
        # cv2.waitKey(1)


        frame = (frame - 128) * 0.2
        frame_tensor = torch.from_numpy(frame)
        frame_tensor = (frame_tensor.abs() > 0).float()
        empty_mask = (frame_tensor == 0)
        noise_mask = (torch.rand_like(frame_tensor) < 0.005) & empty_mask 
        frame_tensor[noise_mask] = 1.0

        input_tensor = frame_tensor.unsqueeze(0).unsqueeze(0).float().to(device)
        with torch.no_grad():
            embeddings = model(input_tensor, return_depth=False)
            embeddings = embeddings.cpu().detach().numpy()
            self.embedding = embeddings


        #     ## print(embeddings)

        #     depth, embeddings = model(input_tensor, return_depth=True)
        #     embeddings = embeddings.cpu().detach().numpy()
        #     self.embedding = embeddings
        #     depth = depth.cpu().detach().numpy()
        #     depth = depth.squeeze(0)
        #     depth = np.transpose(depth, (1, 2, 0))

        # depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        # depth_uint8 = depth_normalized.astype(np.uint8)

        # cv2.imshow("depth", depth_uint8)
        # cv2.waitKey(1)


if __name__ == "__main__":
    try:
        rospy.init_node('gazebo_simulation_node', anonymous=True)
        simulation = GazeboSimulation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
