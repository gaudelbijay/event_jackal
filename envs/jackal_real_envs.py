from typing import Optional
import gym
import time
import numpy as np
import os
from os.path import join
import subprocess
from gym.spaces import Box, Tuple

try:  # Import ROS packages if available
    import rospy
    import rospkg
except ModuleNotFoundError:
    pass

from envs.jackal_real import JackalRealRobot


class JackalRobotEnv(gym.Env):
    def __init__(
        self,
        world_name="jackal_world.world",
        gui=False,
        init_position=[0, 0, 0],
        goal_position=[4, 0, 0],
        max_step=100,
        time_step=1,
        slack_reward=-1,
        failure_reward=-50,
        success_reward=0,
        collision_reward=0,
        goal_reward=1,
        max_collision=10000,
        verbose=True,
        init_sim=False
    ):
        super().__init__()

        # Configuration parameters
        self.gui = gui
        self.verbose = verbose
        self.init_sim = init_sim
        self.world_name = world_name
        self.init_position = init_position
        self.goal_position = goal_position
        self.time_step = time_step
        self.max_step = max_step
        self.slack_reward = slack_reward
        self.failure_reward = failure_reward
        self.success_reward = success_reward
        self.collision_reward = collision_reward
        self.goal_reward = goal_reward
        self.max_collision = max_collision

        self.world_frame_goal = (
            self.init_position[0] + self.goal_position[0],
            self.init_position[1] + self.goal_position[1],
        )

        self.jackal_robot = JackalRealRobot(init_position=self.init_position)
        self.event_shape = self.jackal_robot.embedding_shape

        # Initialize environment attributes
        self.action_space = None
        self.observation_space = None
        self.step_count = 0
        self.collision_count = 0
        self.collided = False
        self.start_time = self.current_time = None
        
    def seed(self, seed: Optional[int] = None):
        """Set the seed for the environment's random number generator."""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]
    

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        """Take an action and step the environment"""
        self._take_action(action)
        self.step_count += 1

        pos, psi = self._get_pos_psi()

        obs = self._get_observation(pos, psi, action)
        goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])

        done, success, collided = self._check_termination(pos, goal_pos)
        reward = self._compute_reward(success, collided, goal_pos)
        print("reward: ", reward)
        self.last_goal_pos = goal_pos
        info = self._get_info(success, goal_pos, collided)

        # Determine if the episode was truncated (e.g., max steps exceeded)
        truncated = self.step_count >= self.max_step

        # Ensure observations are float32 and within the observation space
        obs = self._ensure_observation_dtype(obs)

        return obs, reward, done, truncated, info

    def _ensure_observation_dtype(self, obs):
        """Ensure the observation is of type float32 and within the observation space."""
        if isinstance(obs, np.ndarray):
            obs = obs.astype(np.float32)
        elif isinstance(obs, tuple):
            obs = tuple(o.astype(np.float32) for o in obs)
        # Add more type checks if necessary
        return obs

    def _take_action(self, action):
        """Executes the given action with timing constraints"""
        target_time = self.current_time + self.time_step
        while rospy.get_time() < target_time:
            pass  # Busy wait for time_step duration
        self.current_time = rospy.get_time()

    def _get_observation(self, pos, psi, action):
        raise NotImplementedError()

    def _get_pos_psi(self):
        """Get the current position and orientation (psi)"""
        pose = self.jackal_robot.get_model_state().pose.pose
        pos = pose.position

        q0, q1, q2, q3 = pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
        psi = np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 ** 2 + q3 ** 2))
        return pos, psi

    def _check_termination(self, pos, goal_pos):
        """Check if the episode should terminate"""
        flip = pos.z > 0.1  # Check if the robot has flipped
        success = np.linalg.norm(goal_pos) < 1
        timeout = self.step_count >= self.max_step
        collided = self.jackal_robot.get_hard_collision() and self.step_count > 1
        self.collision_count += int(collided)

        done = flip or success or timeout or self.collision_count >= self.max_collision
        return done, success, collided

    def _compute_reward(self, success, collided, goal_pos):
        """Compute the reward based on the current state"""
        reward = self.slack_reward
        if success:
            reward += self.success_reward
        if collided:
            reward += self.collision_reward
        if not success and (collided or self.step_count >= self.max_step):
            reward += self.failure_reward

        # Reward based on distance to goal
        reward += (np.linalg.norm(self.last_goal_pos) - np.linalg.norm(goal_pos)) * self.goal_reward
        self.last_goal_pos = goal_pos
        return reward

    def _get_info(self, success, goal_pos, collided):
        """Gather and return episode information"""
        info = dict(
            collision=self.collision_count,
            collided=collided,
            goal_position=goal_pos,
            time=rospy.get_time() - self.start_time,
            success=success,
            world=self.world_name
        )
        return info

    def close(self):
        """Shutdown the ROS processes"""
        os.system("killall -9 rosmaster gzclient gzserver roscore")


class JackalRobotEvents(JackalRobotEnv):
    def __init__(self, event_clip=2.0, **kwargs):
        super().__init__(**kwargs)
        self.embedding_clip = event_clip
        
        obs_dim = 1024 + 2 + 2 #self.action_dim
        self.observation_space = Box(
            low=np.float32(-2.0),
            high=np.float32(event_clip),
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _get_embedding_data(self):
        return self.jackal_robot.get_embedding()

    def _observation_events(self):
        return self._get_embedding_data()

    def _get_observation(self, pos, psi, action):
        # Get event data
        event_data = self._observation_events().astype(np.float32)

        # Transform and normalize goal position
        goal_pos = self.transform_goal(self.world_frame_goal, pos, psi) / 5.0 - 1  # roughly (-1, 1) range

        # Normalize action
        bias = (self.action_space.high + self.action_space.low) / 2.0
        scale = (self.action_space.high - self.action_space.low) / 2.0
        action = (action - bias) / scale

        obs = [event_data.flatten(), goal_pos.astype(np.float32), action.astype(np.float32)]
        # print(obs)
        obs = np.concatenate(obs)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs
    
    def transform_goal(self, goal_pos, pos, psi):
        """ transform goal in the robot frame
        params:
            pos_1
        """
        R_r2i = np.matrix([[np.cos(psi), -np.sin(psi), pos.x], [np.sin(psi), np.cos(psi), pos.y], [0, 0, 1]])
        R_i2r = np.linalg.inv(R_r2i)
        pi = np.matrix([[goal_pos[0]], [goal_pos[1]], [1]])
        pr = np.matmul(R_i2r, pi)
        lg = np.array([pr[0,0], pr[1, 0]])
        return lg