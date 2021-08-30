import numpy as np
import kinpy as kp
import rospy
import pygame
import sys
import pickle
import time

# from teleop import main as play
from teleop import main as tele

from sensor_msgs.msg import (
      JointState
)


# robot = sys.argv[1]
# block = sys.argv[2]
#
# traj_file = open('data/'+robot+'_'+block+'.pkl', 'rb')
# trajectory = pickle.load(traj_file)

traj_file1 = open('data/fetch_2.pkl', 'rb')
trajectory1 = pickle.load(traj_file1)


traj_file2 = open('data/panda_1.pkl', 'rb')
trajectory2 = pickle.load(traj_file2)

# for goal_n, goal in enumerate(trajectory2):
#     print(goal_n)
#     tele(goal, 'panda')

# for goal_n, goal in enumerate(trajectory1):
#     print(goal)
#     tele(goal_n, goal, 'fetch')

tele(trajectory1, trajectory2)
