import numpy as np
import random
import time
import pickle
import sys

from utils.world import *


# def savedGoals(task, robot, goal_n):
#     filename = "data/"+task+"/"+robot+"_"+goal_n+".pkl"
#     filename = 'data/'+task+'/scores.pkl'
#     waypoints = pickle.load(open(filename, "rb"))
#     return waypoints

fetch_p0 = np.array([0.51211858, 0.21932284, 1.02747358])
new = transform(fetch_p0)
print(new)
print(transform(new, True))


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
