import numpy as np
import pygame
import sys
import pickle
import time

from teleop import main as tele
from utils.world import *


task = sys.argv[1]
test = sys.argv[2]

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

# create a batch of allocations
batch_allocation = np.empty([4, 4])
if test == 'legible':
    for item in scores:
        print(item[:2])
    # order in the batch: legible, illegible, illegible, legible
    batch_allocation = np.concatenate((scores[7:],scores[6:7],scores[5:6],scores[1:2]))
    print()
    for item in batch_allocation:
        print(item[:2])
    x
elif test == 'fairness':
    pass
else:
    print('WRONG INPUT!')






ALLOCATION_PANDA = []
ALLOCATION_FETCH = []

# this loop iterates ranked allocations and creates ACTUAL robot trajectory
for gstar_idx in range(len(batch_allocation)):

    # finds ranked score table allocations in the original score table
    allocation_name = batch_allocation[gstar_idx][0]
    result = np.where(scores == allocation_name)
    allocation_n = result[0][0]

    # recorded end-effector positions (robots might have no target)
    flag_panda = False
    flag_fetch = False
    panda_target = 'panda_' + allocation_name[5]
    fetch_target = 'fetch_' + allocation_name[-1]

    # check if robots have 'no-play' allocation
    if allocation_name[5] == '0':
        flag_panda = True
    if allocation_name[-1] == '0':
        flag_fetch = True
    positions_panda = savedData(task, panda_target)
    positions_fetch = savedData(task, fetch_target)

    # initial end-effector height
    h0_panda = positions_panda[0][2]
    h0_fetch = positions_fetch[0][2]

    # bean location of robots
    panda_bean = savedData(task, 'panda_bean')
    fetch_bean = savedData(task, 'fetch_bean')

    # robots states for the allocation
    gstar_panda = States_panda[allocation_n]
    gstar_fetch = States_fetch[allocation_n]

    trajectory_panda = []
    trajectory_fetch = []

    # transform Pygame saved states to Actual robot states
    for idx_panda in range(len(gstar_panda)):
        pos_panda = transformFromPygame(gstar_panda[idx_panda][0],gstar_panda[idx_panda][1])
        trajectory_panda.append(list(pos_panda)+[h0_panda])

    for idx_fetch in range(len(gstar_fetch)):
        pos_fetch = transformFromPygame(gstar_fetch[idx_fetch][0],gstar_fetch[idx_fetch][1])
        h0_fetch_tfmd = transform(np.array([0,0,h0_fetch]),True ,task, allocation_name[-1])[-1]
        pos_fetch_tfmd = transform(list(pos_fetch)+[h0_fetch_tfmd],True ,task, allocation_name[-1], back_to_fetch=True)
        trajectory_fetch.append(pos_fetch_tfmd)

    # add pick and drop locations to end of robot legible trajectories
    trajectory_panda.append(list(positions_panda[2]))
    trajectory_panda.append(trajectory_panda[-2])
    trajectory_panda.append(list(panda_bean[0]))

    trajectory_fetch.append(positions_fetch[2])
    trajectory_fetch.append(trajectory_fetch[-2])
    trajectory_fetch.append(fetch_bean[0])

    # append the working status ('play' or 'no-play') of robots to the trajectory
    allocation_panda = [flag_panda, trajectory_panda]
    allocation_fetch = [flag_fetch, trajectory_fetch]

    # append to the trajectory batch
    ALLOCATION_PANDA.append(allocation_panda)
    ALLOCATION_FETCH.append(allocation_fetch)

# play trajectories
tele(ALLOCATION_PANDA, ALLOCATION_FETCH, test)
