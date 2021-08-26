from misc.game.gameplay_record import GamePlay
from utils.parse import *

import numpy as np
import random
import argparse
from collections import namedtuple
import pickle
import gym

folder = 'fairness'
# folder = 'legibility'

arglist = parse_arguments()
if arglist.play:
    if folder == 'fairness':
        n_allocations = 9
    else:
        n_allocations = 10
    for idx in range(n_allocations):
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(env.filename, env.world, env.sim_agents)
        data = game.on_execute()
        savename = "data/"+folder+"/actions_"+str(idx+1)+".pkl"
        pickle.dump(data, open(savename, "wb"))
        print('[*] Saved: Allocation ' +str(idx+1))
