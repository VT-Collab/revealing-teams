import numpy as np
import random
import time
import pickle
import sys
import socket
import matplotlib.pyplot as plt


from utils.grid_world import *
from utils.robot_actions import actionSpace
from utils.panda_home import main


def savedGoals(task, file):
    filename = "data/"+task+"/"+file+".pkl"
    data = pickle.load(open(filename, "rb"))
    return data

print()


fetch_to_obj1 = savedGoals('task1', 'fetch_1')
print('All recorded positions: ', fetch_to_obj1)
print()

print('Obj1 from Fetch:',fetch_to_obj1[1])
print()

x = transform(fetch_to_obj1[1])
print('Transformed:',x)
print()

y = transform(x, back_to_fetch=True)
print('Back transformed:',y)
print()
#
#
#
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
