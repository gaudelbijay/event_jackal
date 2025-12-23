![Model diagram](media/model.png)

This repository implements an event driven reinforcement learning framework for autonomous robot navigation using neuromorphic event cameras. Instead of relying on dense frame based sensors or laser scanners, the system processes asynchronous event streams to produce compact Binary Event Maps and learns a self supervised contrastive perception embedding optimized for navigation. The learned representation is used by reinforcement learning policies including MLP, CNN, GRU, and Transformer agents trained with Soft Actor Critic. The approach achieves navigation performance comparable to laser based systems in simulation and successfully transfers to a real Jackal UGV with minimal real world calibration, demonstrating the practicality of event based perception for robotic navigation.

## System Platform

This project was developed and evaluated using the following system configuration:

- **Operating System:** Ubuntu 20.04 LTS  
- **ROS Distribution:** ROS Noetic  
- **GPU:** 2× NVIDIA GeForce RTX 4500 Ada  
- **CPU:** AMD Ryzen Threadripper
- **System Memory:** 128 GB RAM  

This platform was used for training reinforcement learning agents, and self supervised event based perception models.

## Install Apptainer

This project uses Apptainer for containerized workflows.

- Official Apptainer documentation: https://apptainer.org/docs/
- Direct installation instructions: https://apptainer.org/docs/user/latest/installation.html

After installing, verify the installation by running:

```bash
apptainer --version
```