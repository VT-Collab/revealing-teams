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
    # define the structure of the team
    mode = sys.argv[1]
    # create game
    pygame.init()
    A = actionSpace()
    team = envAgents()
    goals, agent_goals = envGoals()

    # the game will draw everything in the sprite list
    sprite_list = pygame.sprite.Group()
    initGroup(sprite_list, goals, team)

    # pool of allocations
    G = []
    for goal_a1 in agent_goals[0]:
        for goal_a2 in agent_goals[1]:
            for goal_a3 in agent_goals[2]:
                tau = np.asarray(goal_a1 + goal_a2 + goal_a3)
                G.append(tau)

    G = G[3:5]

    # main loop
    states = []
    scores = np.empty([len(G),3])
    P_aloc = np.empty([len(G),len(G)])

    for gstar_idx in range(len(G)):
        resetPos(team)

        # pick the desired allocation
        gstar = np.copy(G[gstar_idx])
        print('[*] Allocation: ', gstar_idx+1)

        # fairness
        fairness = np.empty([len(G),len(team)])
        dist = abs(getState(team)- gstar)
        dist_normed = []
        for idx in range(len(dist)):
          if idx % 2 == 0:
            dist_normed.append(np.linalg.norm(dist[idx:idx+2]))
        fairness[gstar_idx] = dist_normed

        # legible robot motion
        P_aloc[gstar_idx], states_r = legibleRobots(mode, team, gstar_idx, gstar, A, G)

        if mode == 'human-robots':
            # human boltzmann model
            states_h = humanAgent(team, gstar_idx, gstar, A)
            states.append(states_h + states_r)
        else:
            states.append(states_r)

        # index, legibility score, and fairness score of each allocation
        scores[gstar_idx,0] = gstar_idx + 1
        scores[gstar_idx,1] = np.max(P_aloc[gstar_idx])
        scores[gstar_idx,2] = np.var(fairness[gstar_idx])


    # create save paths and store the data
    savename1 = "../data/"+mode+"/allocations.pkl"
    pickle.dump(G, open(savename1, "wb"))
    savename2 = "../data/"+mode+"/scores.pkl"
    pickle.dump(scores, open(savename2, "wb"))
    savename3 = "../data/"+mode+"/states.pkl"
    pickle.dump(states, open(savename3, "wb"))



if __name__ == "__main__":
    main()
