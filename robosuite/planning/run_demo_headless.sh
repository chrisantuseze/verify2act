#!/bin/bash
# Script to run demo_phase2.py in headless mode using xvfb

# Set MuJoCo rendering backend to GLFW (works with xvfb)
export MUJOCO_GL=glfw

# Run with virtual display
xvfb-run -a python demo_phase2.py "$@"
