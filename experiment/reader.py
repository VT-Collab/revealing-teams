import numpy as np
import random
import time
import pickle
import sys

from utils.world import *


def savedGoals(task, robot, goal_n):
    filename = "data/"+task+"/"+robot+"_"+goal_n+".pkl"
    filename = 'data/'+task+'/scores.pkl'
    waypoints = pickle.load(open(filename, "rb"))
    return waypoints

def rotZ(p):
    location = np.concatenate((p,np.array([1])), axis=0)
    print(location)
    # translations
    dx = 0
    dy = 0
    dz = 0
    T = np.array([[np.cos(np.pi),-np.sin(np.pi),0,dx],
                [np.sin(np.pi),np.cos(np.pi),0,dy],
                [0,0,1,dz],
                [0,0,0,1]])
    return np.matmul(T,location)


# initial end-effector positions
fetch_p0 = np.array([0.51211858, 0.21932284, 1.02747358])
panda_p0 = np.array([ 3.10175333e-01, -4.84159689e-06,  4.87596777e-01])

# p1 = rotZ(panda_p0)

location('task1', 'panda')









# for task in ['task1', 'task2']:
#     scores = savedGoals('task2', 'panda', '1')
#
#     # sort scores in descending order, ranked by legibility
#     ranked_scores = scores[scores[:, 1].argsort()]
#
#     # sort based on fairness
#     # ranked_scores = ranked_scores[ranked_scores[:,2].argsort(kind='mergesort')]
#
#     ranked_scores = ranked_scores[::-1]
#
#     for i in range(6):
#         # print('[*] Ranked based on legibility: ','\n',ranked_scores[i,0])
#         print(ranked_scores[i,0])
#
#         # print('[*] Ranked based on fairness: ','\n',ranked_scores[i,0])
#     print(ranked_scores[:,1])
#     print('\n')
