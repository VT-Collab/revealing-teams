import numpy as np
import pygame
import sys
import pickle
import time

task = sys.argv[1]


def savedData(task, file):
    filename = "data/"+task+"/"+file+".pkl"
    data = pickle.load(open(filename, "rb"))
    return data



fetch_home = savedData('task1','fetch_home')

for obj in [1,2,3]:
    fetch_pos = savedData(task,'fetch_' + str(obj))
    fetch_pos[0] = fetch_home[0]
    savename1 = "data/"+task+"/fetch_" + str(obj) + ".pkl"
    pickle.dump(fetch_pos, open(savename1, "wb"))
print(fetch_pos)
print('Done!')
