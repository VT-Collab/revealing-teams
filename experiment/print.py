import numpy as np
import random
import time
import pickle
import sys
import socket
import matplotlib.pyplot as plt


from utils.world import *
from utils.robot_actions import actionSpace
from utils.panda_home import main
from utils.grid_world import *


envGoals('task1')











# # plot robot actions
# A = actionSpace()
# for a in A:
#     x1 = a[0]
#     y1 = a[1]
#     x2 = a[2]
#     y2 = a[3]
#     plt.scatter(x1,y1)
#     plt.scatter(x2,y2)
# plt.show()



# def savedGoals(task, data):
#     filename = 'data/'+task+'/'+data+'.pkl'
#     # filename = 'data/'+task+'/scores.pkl'
#     waypoints = pickle.load(open(filename, "rb"))
#     return waypoints


# for task in ['task1', 'task2']:
#     scores = savedGoals('task2', 'panda', '1')
#
#     # sort scores in descending order, ranked by legibility
#     ranked_scores = scores[scores[:, 1].argsort()]
#
#     # sort based on fairness
#     ranked_scores = ranked_scores[ranked_scores[:,2].argsort(kind='mergesort')]
#
#     ranked_scores = ranked_scores[::-1]
#
#     for i in range(6):
#         # print('[*] Ranked based on legibility: ','\n',ranked_scores[i,0])
#         print(ranked_scores[i,0])
#
#         # print('[*] Ranked based on fairness: ','\n',ranked_scores[i,0])
#     print(ranked_scores[:,2])
#     print('\n')
