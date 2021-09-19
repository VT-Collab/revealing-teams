import numpy as np
import pygame
import sys
import pickle
import time

from teleop import main as tele
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

# main loop
# for gstar_idx, item in enumerate(ranked_scores):
#     print('[*] Allocation: ', gstar_idx)
#     allocation_name = item[0]
#     pandaallocation_name[5]
#     allocation_name[-1]
#     print(allocation_name)
#     x

print(ranked_scores)
# x
n = 0
allocation_name = ranked_scores[n][0]
print(allocation_name)
result = np.where(scores == allocation_name)
allocation_n = result[0][0]

# recorded end-effector positions (robots might have no target)
flag_panda = False
flag_fetch = False
panda_target = 'panda_' + allocation_name[5]
fetch_target = 'fetch_' + allocation_name[-1]

try:
    positions_panda = savedData(task, panda_target)
except:
    panda_target = panda_target.replace('0','1')
    positions_panda = savedData(task, panda_target)
    flag_panda = True

try:
    positions_fetch = savedData(task, fetch_target)
except:
    fetch_target = fetch_target.replace('0','1')
    positions_fetch = savedData(task, fetch_target)
    flag_fetch = True


# initial end-effector height
h0_panda = positions_panda[0][2]
h0_fetch = positions_fetch[0][2]

# beans
panda_bean = savedData(task, 'panda_bean')
fetch_bean = savedData(task, 'fetch_bean')

# n = 1
# for gstar in states:
gstar_panda = States_panda[allocation_n]
gstar_fetch = States_fetch[allocation_n]

trajectory_panda = []
trajectory_fetch = []

for idx_panda in range(len(gstar_panda)):
    pos_panda = transformFromPygame(gstar_panda[idx_panda][0],gstar_panda[idx_panda][1])
    trajectory_panda.append(list(pos_panda)+[h0_panda])

for idx_fetch in range(len(gstar_fetch)):
    pos_fetch = transformFromPygame(gstar_fetch[idx_fetch][0],gstar_fetch[idx_fetch][1])
    h0_fetch_tfmd = transform(np.array([0,0,h0_fetch]),True ,task, allocation_name[-1])[-1]
    pos_fetch_tfmd = transform(list(pos_fetch)+[h0_fetch_tfmd],True ,task, allocation_name[-1], back_to_fetch=True)
    trajectory_fetch.append(pos_fetch_tfmd)

trajectory_panda.append(list(positions_panda[2]))
trajectory_panda.append(trajectory_panda[-2])
trajectory_panda.append(list(panda_bean[0]))


trajectory_fetch.append(positions_fetch[2])
trajectory_fetch.append(trajectory_fetch[-2])
trajectory_fetch.append(fetch_bean[0])

allocation_panda = [flag_panda, trajectory_panda]
allocation_fetch = [flag_fetch, trajectory_fetch]


tele(allocation_panda, allocation_fetch)
