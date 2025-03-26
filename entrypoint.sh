#!/bin/bash
set -e

# Source Python virtual environment
if [ -f /venv/bin/activate ]; then
    source /venv/bin/activate
else
    echo "Error: Virtual environment at /venv/bin/activate not found"
    exit 1
fi

# Source ROS environment
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
else
    echo "Error: ROS Noetic setup script not found"
    exit 1
fi

# Source Catkin workspace setup script
if [ -f /event_ws/devel/setup.bash ]; then
    source /event_ws/devel/setup.bash
else
    echo "Warning: /event_ws/devel/setup.bash not found"
fi

# Change directory to the event jackal workspace
if [ -d /event_ws/src/event_jackal/ ]; then
    cd /event_ws/src/event_jackal/
else
    echo "Error: Directory /event_ws/src/event_jackal/ not found"
    exit 1
fi

# Check if a command is provided
if [ -z "$1" ]; then
    echo "No command provided to entrypoint"
    exit 1
fi

# Execute the passed command
exec "$@"
