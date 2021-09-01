import numpy as np
import random
import time
import pickle
import sys

from utils.world import allocations


def savedGoals(task, robot, goal_n):
    filename = "data/"+task+"/"+robot+"_"+goal_n+".pkl"
    filename = 'data/task1/scores.pkl'
    waypoints = pickle.load(open(filename, "rb"))
    return waypoints



data = savedGoals('task1', 'panda', '1')
print(data)
