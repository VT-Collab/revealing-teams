import numpy as np
import random
import time
import pickle
import sys

from utils.world import *


def savedGoals(task, data):
    filename = 'data/'+task+'/'+data+'.pkl'
    # filename = 'data/'+task+'/scores.pkl'
    waypoints = pickle.load(open(filename, "rb"))
    return waypoints



states = savedGoals('task2', 'robots_states')
print(states)


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
