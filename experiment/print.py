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
# from utils.grid_world import *


panda_to_obj1 = savedGoals('task1', 'panda', '1')
fetch_to_obj1 = savedGoals('task1', 'fetch', '1')

print('Obj1 from Panda:',panda_to_obj1[1])
print()
print('Obj1 from Fetch:',fetch_to_obj1[1])
print()
x = transform(fetch_to_obj1[1])
print('Transformed Obj1 from Fetch:',x)
print()

y = transform(x, back_to_fetch=True)
print('Back transformed Obj1 from Fetch:',y)
print()



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
