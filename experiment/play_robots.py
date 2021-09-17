import numpy as np
import pygame
import sys
import pickle
import time

from teleop_class import main as tele
from utils.world import *



task = sys.argv[1]


def savedData(task, file):
    filename = "data/"+task+"/"+file+".pkl"
    data = pickle.load(open(filename, "rb"))
    return data

# import score and allocation saved data
allocations = savedData(task,'allocations')
scores = savedData(task,'scores')

# import states saved data
States_panda = savedData(task,'States_panda')
States_fetch = savedData(task,'States_fetch')

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


# recorded end-effector positions
positions_panda = savedData(task, 'panda_1')
positions_fetch = savedData(task, 'fetch_2')

# initial end-effector height
h0_panda = positions_panda[0][2]
h0_fetch = positions_fetch[0][2]

# beans
panda_bean = savedData(task, 'panda_bean')
fetch_bean = savedData(task, 'fetch_bean')

n = 1
# for gstar in states:
gstar_panda = States_panda[n]
gstar_fetch = States_fetch[n]


trajectory_panda = []
trajectory_fetch = []

for idx in range(len(gstar_panda)):

    pos_panda = transformFromPygame(gstar_panda[idx][0],gstar_panda[idx][1])
    trajectory_panda.append(list(pos_panda)+[h0_panda])


    pos_fetch = transformFromPygame(gstar_fetch[idx][0],gstar_fetch[idx][1])
    h0_fetch_tfmd = transform(np.array([0,0,h0_fetch]))[-1]
    pos_fetch_tfmd = transform(list(pos_fetch)+[h0_fetch_tfmd], back_to_fetch=True)
    trajectory_fetch.append(pos_fetch_tfmd)


trajectory_panda.append(positions_panda[2])
trajectory_panda.append(trajectory_panda[-2])
trajectory_panda.append(panda_bean[0])


trajectory_fetch.append(positions_fetch[2])
trajectory_fetch.append(trajectory_fetch[-2])
trajectory_fetch.append(fetch_bean[0])


tele(trajectory_panda, trajectory_fetch)
