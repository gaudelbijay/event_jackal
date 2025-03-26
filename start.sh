#!/bin/bash

killall -9 roslaunch
killall -9 rosmaster
killall -9 roscore
killall -9 gzserver
killall -9 gzclient

rosclean purge -y

cd ~/event_ws/src/event_jackal/
# Activate the Python virtual environment
source ~/.jackal/bin/activate

# Source the ROS workspace setup file
source ~/event_ws/devel/setup.sh

# Run the Python training script with the specified config file
python train.py --config configs/generalization/num_world_5.yaml

