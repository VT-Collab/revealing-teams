import numpy as np
import kinpy as kp
import rospy
import pygame
import sys
import pickle
import time

from teleop import main as play

from sensor_msgs.msg import (
      JointState
)


robot = sys.argv[1]
block = sys.argv[2]

traj_file = open('data/'+robot+'_'+block+'.pkl', 'rb')
trajectory = pickle.load(traj_file)

for goal in trajectory:
    print(goal)
    play(goal)

# play(trajectory[2])
