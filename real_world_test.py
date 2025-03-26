import numpy as np
from train import initialize_policy
import os
from os.path import dirname, abspath
import gym
import argparse
import yaml
import rospy
import pickle
from os.path import join, exists
import time 
import sys
sys.path.append(dirname(dirname(abspath(__file__))))
from envs.wrappers import StackFrame

BUFFER_PATH = os.getenv('BUFFER_PATH', 'local_buffer')
print("buffer path:", BUFFER_PATH)

def load_policy(policy, policy_path):
    print("policy:", policy)
    policy_name = "policy"
    while True:
        print("     >>>>>> policy loading started ")
        try:
            if not exists(join(BUFFER_PATH, f"{policy_name}_copy_actor")):
                policy.load(policy_path, policy_name)
            break
        except FileNotFoundError:
            time.sleep(1)
        except Exception as e:
            time.sleep(1)
    print("     >>>>>> policy loaded ")
    return policy

def write_buffer(traj, actor_id):
    actor_path = join(BUFFER_PATH, f'actor_{actor_id}')
    file_names = os.listdir(actor_path)
    ep = max([int(f.split("_")[-1].split(".pickle")[0]) for f in file_names if f.endswith('.pickle')], default=-1) + 1

    if len(file_names) < 10:
        with open(join(actor_path, f'traj_{ep}.pickle'), 'wb') as f:
            try:
                pickle.dump(traj, f)
            except OSError as e:
                pass
    return ep

def main(args):
    rospy.init_node('real_world_test_node', anonymous=True)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    num_trials = 40
    env_config = config['env_config']
    world_name = args.world_name
    test_object = "local"

    env_config["kwargs"]["world_name"] = world_name
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])

    policy, _ = initialize_policy(config, env, init_buffer=False)

    # if config["training_config"]["algorithm"] == "MPC":
    #     policy.exploration_noise = config["training_config"]["exploration_noise_end"]
    #     print("exploration_noise: %.2f" % policy.exploration_noise)

    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" % world_name)
    ep = 0
    bad_traj_count = 0

    try:
        while ep < num_trials:
            obs, _ = env.reset()
            traj = []
            done = False
            while not done:
                if test_object == "local":
                    actions = policy.select_action(obs)
                elif test_object == "dwa":
                    actions = env_config["kwargs"]["param_init"]

                print("action generated:", actions)
                obs_new, rew, done, truncated, info = env.step(actions)
                info["world"] = world_name
                traj.append([None, None, rew, done, info])  
                obs = obs_new

            time_per_step = info['time'] / len(traj)
            if len(traj) >= 1 and time_per_step < (0.05 + config["env_config"]["kwargs"]["time_step"]):
                ep = write_buffer(traj, args.id)
            else:
                bad_traj_count += 1
                if bad_traj_count >= 5:
                    break

    except KeyboardInterrupt:
        print("\nProcess interrupted. Shutting down gracefully.")

    finally:
        rospy.signal_shutdown("Script terminated by user")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='start an tester')
    parser.add_argument('--id', dest='id', type=int, default=0)
    parser.add_argument('--world_name', dest='world_name', type=str, default="0", help="Path to the config.yaml file")
    parser.add_argument('--config', dest='config', type=str, required=True, help="Path to the config.yaml file")
    args = parser.parse_args()
    main(args)
