# Event-Driven Reinforcement Learning for Autonomous Navigation

![Model diagram](media/model.png)
![Perception diagram](media/perception_model.png)

This repository implements an event driven reinforcement learning framework for autonomous robot navigation using neuromorphic event cameras. Instead of relying on dense frame based sensors or laser scanners, the system processes asynchronous event streams to produce compact Binary Event Maps and learns a self supervised contrastive perception embedding optimized for navigation. The learned representation is used by reinforcement learning policies including MLP, CNN, GRU, and Transformer agents trained with Soft Actor Critic. The approach achieves navigation performance comparable to laser based systems in simulation and successfully transfers to a real Jackal UGV with minimal real world calibration, demonstrating the practicality of event based perception for robotic navigation.

---
## System Platform

- Operating System: Ubuntu 20.04 LTS  
- ROS Distribution: ROS Noetic  
- GPU: 2× NVIDIA GeForce RTX 4500 Ada  
- CPU: AMD Ryzen Threadripper  
- System Memory: 128 GB RAM  

This platform was used for training reinforcement learning agents and self supervised event based perception models.

---

## Dependencies

### ROS Noetic

https://wiki.ros.org/noetic/Installation/Ubuntu

### Clearpath Jackal Simulator (ROS Noetic)

```bash
sudo apt update
sudo apt install -y ros-noetic-jackal-simulator \
                   ros-noetic-jackal-desktop \
                   ros-noetic-jackal-navigation
```
## Install Apptainer

This project uses Apptainer for containerized workflows.

- Official Apptainer documentation: https://apptainer.org/docs/
- Direct installation instructions: https://apptainer.org/docs/admin/main/installation.html

After installing, verify the installation by running:

```bash
apptainer --version
```

## Build the apptainer image
```bash
apptainer build event_nav.sif Singularity.def
```
**Move the built apptainer image `event_nav.sif` to local_buffer directory**

## Install catkin tools if needed:

```bash
sudo apt update
sudo apt install -y python3-catkin-tools
```

## Clone, build and source the workspace
```bash
source /opt/ros/noetic/setup.bash

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone git@github.com:gaudelbijay/event_jackal.git

cd ~/catkin_ws
catkin build

source ~/catkin_ws/devel/setup.bash
```

## Training

Quick start
```bash
./start.sh
```

Overview
- Training is controlled by YAML config files under `configs/`.  
- Pass the path to the config you want with the `--config` argument to `train.py`. For example you can change the last line of `start.sh` file like:
```bash
python train.py --config configs/generalization/num_world_50.yaml
```


## Testing
```bash
python tester.py
```
**Note:** Configuration files do not need to be specified manually. During training, the active configuration is automatically copied into the `local_buffer` directory, and `tester.py` uses the same stored configuration for evaluation.

## Results

![training-result](media/training_metrics.png)

![test-table](media/table.png)

![vis](media/vis.png)