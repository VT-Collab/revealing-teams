import numpy as np
import kinpy as kp
import rospy
import pygame
import sys
import pickle
import time

# from teleop import main as play
from teleop import main as tele
from utils.grid_world import transformFromPygame

from sensor_msgs.msg import (
      JointState
)

task = sys.argv[1]


def savedData(task, file):
    filename = "data/"+task+"/"+file+".pkl"
    data = pickle.load(open(filename, "rb"))
    return data

# import score and allocation saved data
allocations = savedData(task,'allocations')
scores = savedData(task,'scores')

# import states saved data
states = savedData(task,'robots_states')


# sort scores in descending order, ranked by legibility
ranked_scores = scores[scores[:, 1].argsort()]
ranked_scores = ranked_scores[::-1]
# print('[*] Ranked based on legibility: ','\n',ranked_scores)

# # sort based on fairness
# ranked_scores = ranked_scores[ranked_scores[:,2].argsort(kind='mergesort')]
# print('[*] Ranked based on fairness: ','\n',ranked_scores)

# # main loop
# for gstar_idx, item in enumerate(ranked_scores):
#     print('[*] Allocation: ', gstar_idx)
#     print(states[gstar_idx])
#     x


# initial end-effector height
positions_panda = savedData(task, 'panda_1')
h0_panda = positions_panda[0][2]
positions_fetch = savedData(task, 'fetch_1')
h0_fetch = positions_fetch[0][2]

trajectory_panda = []
trajectory_fetch = []

for idx in range(len(states[0])):

    tuple_panda = transformFromPygame(states[0][idx][0],states[0][idx][1])
    trajectory_panda.append(list(tuple_panda)+[h0_panda])
    
    tuple_fetch = transformFromPygame(states[0][idx][2],states[0][idx][3])
    trajectory_fetch.append(list(tuple_fetch)+[h0_fetch])


tele(trajectory_panda, trajectory_fetch)
