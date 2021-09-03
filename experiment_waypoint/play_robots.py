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

task = sys.argv[1]
n = sys.argv[2]
m = sys.argv[3]
#


# task = 'task1'
# m = '1'

traj_file1 = open('data/'+task+'/fetch_'+n+'.pkl', 'rb')
trajectory1 = pickle.load(traj_file1)

traj_file2 = open('data/'+task+'/panda_'+m+'.pkl', 'rb')
trajectory2 = pickle.load(traj_file2)

# for goal_n, goal in enumerate(trajectory2):
#     print(goal_n)
#     tele(goal, 'panda')
#
# for goal_n, goal in enumerate(trajectory1):
#     print(goal)
#     tele(goal_n, goal, 'fetch')

tele(trajectory1, trajectory2)




# filename3 = "data/task1/robots_states.pkl"
# states = pickle.load(open(filename3, "rb"))
#
# trajectory2 = []
# for state in states[0]:
#     trajectory2.append(np.array(state[:3]))
#
# tele(trajectory2, trajectory2)
