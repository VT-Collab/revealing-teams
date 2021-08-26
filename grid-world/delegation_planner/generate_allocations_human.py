import sys
sys.path.append('../')

from utils.world import (
        Object, getState, updateState,
        resetPos, envAgents, envGoals, initGroup
)
from utils.actions import actionSpace
from navigation_planner.legible import legibleRobots, humanAgent

import pygame
import os
import numpy as np
import time
import pickle



def main():
    # create game
    pygame.init()
    A = actionSpace()
    team = envAgents()
    goals, agent_goals = envGoals()

    # the game will draw everything in the sprite list
    sprite_list = pygame.sprite.Group()
    initGroup(sprite_list, goals, team)

    # import pool of allocations used for the robot group
    filename = "../data/human-robots/allocations.pkl"
    G = pickle.load(open(filename, "rb"))

    # main loop
    # betas = [0, 50, 100, 500, 800]
    # for beta in betas:
    states = []
    beta = 100
    for gstar_idx in range(len(G)):
        resetPos(team)

        # pick the desired allocation
        gstar = np.copy(G[gstar_idx])
        print('[*] Allocation: ', gstar_idx+1)

        # human boltzmann model
        states_h = humanAgent(team, gstar_idx, gstar, A, beta)
        states.append(states_h)

    savename3 = "../data/human-robots/human_states_"+str(beta)+".pkl"
    pickle.dump(states, open(savename3, "wb"))



if __name__ == "__main__":
    main()
