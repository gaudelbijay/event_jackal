#!/usr/bin/env python3
"""
train.py

A comprehensive training script for reinforcement learning using Soft Actor-Critic (SAC) and other algorithms.
Ensures correct policy loading, isolated logging per run, unique policy filenames, and robust logging mechanisms.
"""

import argparse
import yaml
import numpy as np
import gym
from datetime import datetime
from os.path import join, exists
import sys
import os
import shutil
import logging
import collections
import time
import uuid
from pprint import pformat

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experience import get_replay_buffer

try:
    import GPUtil
    from tensorboardX import SummaryWriter
except ImportError:
    GPUtil = None
    SummaryWriter = None

from envs import registration
from envs.wrappers import StackFrame
from rl_algos import algo_class
from rl_algos.net import *
from rl_algos.base_rl_algo import ReplayBuffer
from rl_algos.sac import GaussianActor
from rl_algos.td3 import Actor, Critic
from rl_algos.model_based import Model


def initialize_config(config_path, save_path):
    logging.info(f"Initializing configuration from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path
    return config


def initialize_logging(config):
    env_config = config["env_config"]
    training_config = config["training_config"]

    # Config logging
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    unique_id = uuid.uuid4().hex[:6]

    save_path = join(
        env_config["save_path"], 
        env_config["env_id"], 
        training_config['algorithm'], 
        timestamp,
        unique_id
    )
    logging.info(f"Saving logs to {save_path}")
    os.makedirs(save_path, exist_ok=True)

    if SummaryWriter is not None:
        writer = SummaryWriter(save_path)
    else:
        writer = None
        logging.warning("TensorBoardX is not installed. Skipping SummaryWriter.")

    shutil.copyfile(
        env_config["config_path"], 
        join(save_path, "config.yaml")    
    )

    return save_path, writer


def initialize_envs(config):
    env_config = config["env_config"]
    env_config["kwargs"]["init_sim"] = False
    logging.info("Using actors on apptainer or condor.")
    env = gym.make("motion_control_continuous_placeholder-v0")
    env = StackFrame(env, stack_frame=env_config.get("stack_frame", 1))
    logging.info("Placeholder environment is initialized.")

    return env


def seed(config):
    logging.info("Seeding the random number generators...")
    env_config = config["env_config"]
    seed_value = env_config.get('seed', None)
    if seed_value is not None:
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        logging.info(f"Random seeds set to {seed_value}.")
    else:
        logging.warning("No seed value provided in configuration.")


def get_encoder(encoder_type, args):
    logging.info(f"Initializing encoder of type: {encoder_type}")
    if encoder_type == "mlp":
        encoder = MLPEncoder(**args)
    elif encoder_type in ['lstm', 'gru']:
        encoder = RNNEncoder(encoder_type=encoder_type, **args)
    elif encoder_type == "cnn":
        encoder = CNNEncoder(**args)
    elif encoder_type == "transformer":
        encoder = TransformerEncoder(**args)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    return encoder


def initialize_policy(config, env, init_buffer=True, device=None):
    logging.info("Initializing policy...")
    training_config = config["training_config"]

    if isinstance(env.observation_space, gym.spaces.Tuple):
        event_data_space, goal_action_space = env.observation_space.spaces
        event_data_dim = event_data_space.shape
        goal_action_dim = goal_action_space.shape
    else:
        state_dim = env.observation_space.shape
        event_data_dim, goal_action_dim = state_dim, state_dim

    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high

    # Device selection simplified within initialize_policy
    if device is None:
        if GPUtil is not None:
            available_gpus = GPUtil.getAvailable(
                order='first',
                limit=1,
                maxLoad=0.8,
                maxMemory=0.8,
                includeNan=False,
                excludeID=[],
                excludeUUID=[]
            )
            if len(available_gpus) > 0:
                device = f"cuda:{available_gpus[0]}"
            else:
                device = "cpu"
        else:
            logging.warning("GPUtil is not installed. Falling back to CPU.")
            device = "cpu"
    logging.info(f"Selected device: {device}")

    encoder_type = training_config["encoder"]
    encoder_args = {
        'input_dim': (np.prod(event_data_dim), goal_action_dim),
        'num_layers': training_config['encoder_num_layers'],
        'hidden_size': training_config['encoder_hidden_layer_size'],
        'history_length': config["env_config"].get("stack_frame", 1),
    }
    encoder = get_encoder(encoder_type, encoder_args)

    algo = training_config["algorithm"]

    # Define allowed policy_args per algorithm
    allowed_policy_args = {
        "SAC": {"tau", "gamma", "alpha", "automatic_entropy_tuning", "n_step"},
        "TD3": {"tau", "gamma", "policy_noise", "noise_clip", "policy_freq"},
        # Add other algorithms and their allowed policy_args here if needed
    }

    # Extract and filter policy_args based on the algorithm
    policy_args = training_config.get("policy_args", {}).copy()
    allowed_args = allowed_policy_args.get(algo, set())
    filtered_policy_args = {k: v for k, v in policy_args.items() if k in allowed_args}

    if len(policy_args) != len(filtered_policy_args):
        ignored_args = set(policy_args.keys()) - set(filtered_policy_args.keys())
        logging.warning(f"Ignored policy_args for algorithm '{algo}': {ignored_args}")

    logging.debug(f"Filtered policy_args for algorithm '{algo}': {filtered_policy_args}")

    # Initialize model only for specific algorithms
    model = None
    model_optim = None
    if any(sub_algo in algo for sub_algo in ["Dyna", "SMCP", "MBPO"]):
        model = Model(
            encoder=encoder,
            head=MLP(training_config['hidden_layer_size'] + np.prod(action_dim),
                     training_config['encoder_num_layers'],
                     training_config['encoder_hidden_layer_size']),
            state_dim=(event_data_dim, goal_action_dim),
            deterministic=training_config.get('deterministic', False)
        ).to(device)
        model_optim = torch.optim.Adam(model.parameters(), lr=training_config['model_lr'])
        logging.info("Model initialized.")

    # Select the appropriate actor and critic classes based on the algorithm
    if "SAC" in algo:
        from rl_algos.sac import SAC, GaussianActor, Critic
        actor_class = GaussianActor
        critic_class = Critic
    elif "TD3" in algo:
        from rl_algos.td3 import TD3, Actor, Critic
        actor_class = Actor
        critic_class = Critic
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    input_dim = training_config['hidden_layer_size']
    actor = actor_class(
        encoder=encoder,
        head=MLP(input_dim,
                 training_config['encoder_num_layers'],
                 training_config['encoder_hidden_layer_size']),
        action_dim=action_dim
    ).to(device)

    # Handle multiple GPUs if available
    if device.startswith("cuda"):
        if GPUtil is not None:
            available_gpus = GPUtil.getAvailable(
                order='first',
                limit=100,
                maxLoad=0.8,
                maxMemory=0.8,
                includeNan=False,
                excludeID=[],
                excludeUUID=[]
            )
            if len(available_gpus) > 1:
                actor = nn.DataParallel(actor, device_ids=available_gpus)
                logging.info(f"Actor wrapped with DataParallel on GPUs {available_gpus}")

    actor_optim = torch.optim.Adam(actor.parameters(), lr=training_config['actor_lr'])
    critic = critic_class(
        encoder=encoder,
        head=MLP(input_dim + action_dim,
                 training_config['encoder_num_layers'],
                 training_config['encoder_hidden_layer_size']),
    ).to(device)

    # Handle multiple GPUs for critic
    if device.startswith("cuda"):
        if GPUtil is not None:
            available_gpus = GPUtil.getAvailable(
                order='first',
                limit=100,
                maxLoad=0.8,
                maxMemory=0.8,
                includeNan=False,
                excludeID=[],
                excludeUUID=[]
            )
            if len(available_gpus) > 1:
                critic = nn.DataParallel(critic, device_ids=available_gpus)
                logging.info(f"Critic wrapped with DataParallel on GPUs {available_gpus}")

    critic_optim = torch.optim.Adam(critic.parameters(), lr=training_config['critic_lr'])

    # Initialize the policy based on the algorithm
    if "SAC" in algo:
        # For SAC, do NOT pass model and model_optim
        policy = algo_class[algo](
            actor, actor_optim,
            critic, critic_optim,
            action_range=[action_space_low, action_space_high],
            device=device,
            **filtered_policy_args
        )
        logging.info(f"{algo} policy initialized.")
    elif "TD3" in algo:
        # For TD3, pass model and model_optim if applicable
        if model is not None and model_optim is not None:
            policy = algo_class[algo](
                model, model_optim,
                actor, actor_optim,
                critic, critic_optim,
                action_range=[action_space_low, action_space_high],
                device=device,
                **filtered_policy_args
            )
        else:
            policy = algo_class[algo](
                actor, actor_optim,
                critic, critic_optim,
                action_range=[action_space_low, action_space_high],
                device=device,
                **filtered_policy_args
            )
        logging.info(f"{algo} policy initialized.")
    elif "Safe" in algo:
        # Implement Safe algorithm initialization as needed
        policy = algo_class[algo](
            # Add necessary parameters for Safe algorithms
            actor, actor_optim,
            critic, critic_optim,
            action_range=[action_space_low, action_space_high],
            device=device,
            **filtered_policy_args
        )
        logging.info(f"{algo} safe policy initialized.")
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    # Initialize Replay Buffer if required
    if init_buffer:
        replay_buffer = get_replay_buffer(
            state_dim=(event_data_dim, goal_action_dim),
            action_dim=action_dim,
            max_size=training_config['buffer_size'],
            device=device,
        )
        logging.info("Replay buffer initialized.")
    else:
        replay_buffer = None

    return policy, replay_buffer


def train(env, policy, replay_buffer, config):
    logging.info("Starting training...")
    env_config = config["env_config"]
    training_config = config["training_config"]

    save_path, writer = initialize_logging(config)
    logging.info("Logging initialized.")


    training_args = training_config.get("training_args", {})
    pre_collect_steps = training_config.get('pre_collect', 1000)


    n_steps = pre_collect_steps
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=300)
    world_ep_buf = collections.defaultdict(lambda: collections.deque(maxlen=20))
    t0 = time.time()

    for i in range(1, 201):
        logging.info(f"Training iteration {n_iter}, total steps {n_steps}, total episodes {n_ep}")
        n_iter += 1

        loss_infos = []
        update_per_step = training_args.get("update_per_step", 1)
        for _ in range(update_per_step):
            loss_info = policy.train(replay_buffer, training_args.get("batch_size", 64))
            loss_infos.append(loss_info)
        logging.info("Training step completed.")

        aggregated_loss_info = {}
        for key in loss_infos[0].keys():
            aggregated_loss_info[key] = np.mean([li.get(key, 0) for li in loss_infos])

        t1 = time.time()
        fps = n_steps / (t1 - t0) if (t1 - t0) > 0 else 0
        t0 = t1

        log = {
            "Episode_return": np.mean([ep["ep_rew"] for ep in epinfo_buf]) if len(epinfo_buf) > 0 else 0,
            "Episode_length": np.mean([ep["ep_len"] for ep in epinfo_buf]) if len(epinfo_buf) > 0 else 0,
            "Success": np.mean([ep["success"] for ep in epinfo_buf]) if len(epinfo_buf) > 0 else 0,
            "Time": np.mean([ep["ep_time"] for ep in epinfo_buf]) if len(epinfo_buf) > 0 else 0,
            "Collision": np.mean([ep["collision"] for ep in epinfo_buf]) if len(epinfo_buf) > 0 else 0,
            "fps": fps,
            "n_episode": n_ep,
            "Steps": n_steps
        }

        if "TD3" in training_config["algorithm"]:
            log["Exploration_noise"] = policy.exploration_noise
        # if "SAC" in training_config["algorithm"]:
        #     log["Alpha"] = policy.alpha if hasattr(policy, 'alpha') else None

        log.update(aggregated_loss_info)

        logging.info(pformat(log))

        if n_iter % training_config.get("log_intervals", 10) == 0:
            if writer is not None:
                for k, v in log.items():
                    if v is not None:
                        writer.add_scalar('train/' + k, v, global_step=n_steps)
            
            logging.info(f"Logging to {save_path}")

            if writer is not None:
                for world, ep_deque in world_ep_buf.items():
                    if len(ep_deque) == 0:
                        continue
                    writer.add_scalar(f"{world}/Episode_return", np.mean([ep["ep_rew"] for ep in ep_deque]), global_step=n_steps)
                    writer.add_scalar(f"{world}/Episode_length", np.mean([ep["ep_len"] for ep in ep_deque]), global_step=n_steps)
                    writer.add_scalar(f"{world}/Success", np.mean([ep["success"] for ep in ep_deque]), global_step=n_steps)
                    writer.add_scalar(f"{world}/Time", np.mean([ep["ep_time"] for ep in ep_deque]), global_step=n_steps)
                    writer.add_scalar(f"{world}/Collision", np.mean([ep["collision"] for ep in ep_deque]), global_step=n_steps)


def main():
    torch.set_num_threads(8)

    parser = argparse.ArgumentParser(description='Start training')
    parser.add_argument('--config_path', dest='config_path', default="configs/e2e_default_SAC.yaml",
                        help='Path to the YAML configuration file.')
    parser.add_argument('--device', dest='device', default=None,
                        help='Specific device to use (e.g., "cuda:0" or "cuda"). If not set, the script auto-selects GPUs.')
    parser.add_argument('--log_level', dest='log_level', default="INFO",
                        help='Logging level (e.g., DEBUG, INFO, WARNING).')
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"Invalid log level: {args.log_level}. Defaulting to INFO.")
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format='%(asctime)s - %(levelname)s - %(message)s')

    CONFIG_PATH = args.config_path
    SAVE_PATH_ROOT = "logging/"
    logging.info(f"Loading the configuration from {CONFIG_PATH}")
    config = initialize_config(CONFIG_PATH, SAVE_PATH_ROOT)

    seed(config)
    logging.info("Creating the environments")
    env = initialize_envs(config)

    logging.info("Initializing the policy")
    policy, replay_buffer = initialize_policy(config, env, device=args.device)

    logging.info("Starting the training loop")
    train(env, policy, replay_buffer, config)


if __name__ == "__main__":
    main()
