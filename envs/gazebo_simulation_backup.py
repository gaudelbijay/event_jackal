from typing import Optional
import gym
import time
import numpy as np
import os
from os.path import join
import subprocess
from gym.spaces import Box, Tuple

try:
    import rospy
    import rospkg
except ModuleNotFoundError:
    pass

from envs.gazebo_simulation import GazeboSimulation

class JackalGazebo(gym.Env):
    def __init__(
        self,
        world_name="jackal_world.world",
        gui=False,
        init_position=[0, 0, 0],
        goal_position=[4, 0, 0],
        max_step=100,
        time_step=1,
        slack_reward=-0.0,
        failure_reward=-0,
        success_reward=20,
        collision_reward=-2,
        goal_reward=2,
        max_collision=100,
        verbose=True,
        init_sim=True
    ):
        super().__init__()
        # Config
        self.gui = gui
        self.verbose = verbose
        self.init_sim = init_sim

        # Sim config
        self.world_name = world_name
        self.init_position = init_position
        self.goal_position = goal_position

        # Env config
        self.time_step = time_step
        self.max_step = max_step


        self.slack_reward = slack_reward
        self.failure_reward = failure_reward
        self.success_reward = success_reward
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.max_collision = max_collision

       
        # Calculate world frame goal
        self.world_frame_goal = (
            self.init_position[0] + self.goal_position[0],
            self.init_position[1] + self.goal_position[1],
        )

        # Initialize distance metrics
        self.initial_distance = np.sqrt(
            (self.goal_position[0])**2 + 
            (self.goal_position[1])**2
        )
        
        if init_sim:
            rospy.logwarn(">>>>>>>>>>>>>>>>>> Load world: %s <<<<<<<<<<<<<<<<<<" %(world_name))
            rospack = rospkg.RosPack()
            self.BASE_PATH = rospack.get_path('jackal_helper')
            world_name = join(self.BASE_PATH, "worlds", world_name)
            launch_file = join(self.BASE_PATH, 'launch', 'gazebo_launch.launch')

            self.gazebo_process = subprocess.Popen(['roslaunch',
                                                    launch_file,
                                                    'world_name:=' + world_name,
                                                    'gui:=' + ("true" if gui else "false"),
                                                    'verbose:=' + ("true" if verbose else "false"),
                                                    ])
            time.sleep(10)

            rospy.init_node('gym', anonymous=True, log_level=rospy.FATAL)
            rospy.set_param('/use_sim_time', True)

            self.gazebo_sim = GazeboSimulation(init_position=self.init_position)
            self.event_shape = self.gazebo_sim.embedding_shape

        # State tracking
        self.action_space = None
        self.observation_space = None
        self.step_count = 0
        self.collision_count = 0
        self.collided = 0
        self.start_time = self.current_time = None
        self.last_distance = None
        self.min_distance_to_goal = float('inf')
        
    def seed(self, seed):
        np.random.seed(seed)

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """Take an action and step the environment"""
        # Execute action
        self._take_action(action)
        self.step_count += 1
        pos, psi = self._get_pos_psi()

        # Update state
        self.gazebo_sim.unpause()
        obs = self._get_observation(pos, psi, action)

        # Calculate goal metrics
        goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])
        current_distance = np.linalg.norm(goal_pos)
        
        # Track best progress
        self.min_distance_to_goal = min(self.min_distance_to_goal, current_distance)
        progress = 1 - (current_distance / self.initial_distance)

        # Check termination conditions
        flip = pos.z > 0.1
        success = current_distance < 0.4
        timeout = self.step_count >= self.max_step
        collided = self.gazebo_sim.get_hard_collision()
        self.collision_count += int(collided)
        
        done = flip or success or timeout or self.collision_count >= self.max_collision

        rew = self.slack_reward
        if done and not success:
            rew += self.failure_reward
        if success:
            rew += self.success_reward
        if collided:
            rew += self.collision_reward

        rew += (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(goal_pos)) * self.goal_reward
        self.last_goal_pos = goal_pos

        info = {
            'collision': self.collision_count,
            'collided': collided,
            'goal_position': goal_pos,
            'time': self.current_time - self.start_time,
            'success': success,
            'world': self.world_name,
            'distance': current_distance,
            'min_distance': self.min_distance_to_goal,
            'progress': progress
        }

        self.gazebo_sim.pause()
        return obs, rew, done, timeout, info

    def _take_action(self, action):
        current_time = rospy.get_time()
        while current_time - self.current_time < self.time_step:
            time.sleep(0.01)
            current_time = rospy.get_time()
        self.current_time = current_time

    def _get_pos_psi(self):
        pose = self.gazebo_sim.get_model_state().pose
        pos = pose.position

        q1 = pose.orientation.x
        q2 = pose.orientation.y
        q3 = pose.orientation.z
        q0 = pose.orientation.w
        psi = np.arctan2(2 * (q0*q3 + q1*q2), (1 - 2*(q2**2+q3**2)))
        assert -np.pi <= psi <= np.pi, psi

        return pos, psi

    def close(self):
        os.system("killall -9 rosmaster")
        os.system("killall -9 gzclient")
        os.system("killall -9 gzserver")
        os.system("killall -9 roscore")


class JackalGazeboEvents(JackalGazebo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        obs_dim = 720 + 2 + self.action_dim  # 720 dim laser scan + goal position + action taken in this time step 
        self.observation_space = Box(
            low=0,
            high=laser_clip,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_embedding_data(self):
        return self.gazebo_sim.get_embedding()

    def _observation_events(self):
        return self._get_embedding_data()

    def _get_observation(self, pos, psi, action):
        # Get event data
        event_data = self._observation_events().astype(np.float32)

        # Transform and normalize goal position
        goal_pos = self.transform_goal(self.world_frame_goal, pos, psi)
        goal_pos = goal_pos / (self.initial_distance * 1.5)

        # Normalize action
        bias = (self.action_space.high + self.action_space.low) / 2.0
        scale = (self.action_space.high - self.action_space.low) / 2.0
        action = (action - bias) / scale

        # Combine goal position and action
        goal_action = np.concatenate([goal_pos, action]).astype(np.float32)

        obs = [event_data, goal_pos, action]
        
        obs = np.concatenate(obs)

        return obs

    def transform_goal(self, goal_pos, pos, psi):
        R_r2i = np.array([
            [np.cos(psi), -np.sin(psi), pos.x],
            [np.sin(psi),  np.cos(psi), pos.y],
            [0,            0,           1]
        ])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.array([goal_pos[0], goal_pos[1], 1])
        pr = R_i2r @ pi
        lg = pr[:2]
        return lg