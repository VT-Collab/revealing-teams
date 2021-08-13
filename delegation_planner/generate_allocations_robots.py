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

    # main loop
    states = []
    scores = np.empty([len(G),4])

    P_aloc = np.empty([len(G),len(G)])

    for gstar_idx in range(len(G)):
        resetPos(team)

        # pick the desired allocation
        gstar = np.copy(G[gstar_idx])
        print('[*] Allocation: ', gstar_idx+1)

        # compute distance to goals for each agent
        Dist = np.empty([len(G),len(team)])
        dist = abs(getState(team)- gstar)
        dist_normed = []
        for idx in range(len(dist)):
          if idx % 2 == 0:
            dist_normed.append(np.linalg.norm(dist[idx:idx+2]))
        Dist[gstar_idx] = dist_normed

        # legible robot motion
        P_aloc[gstar_idx], states_r = legibleRobots(mode, team, gstar_idx, gstar, A, G)
        states.append(states_r)

        # index allocations
        scores[gstar_idx,0] = gstar_idx + 1
        # legibility of allocations
        scores[gstar_idx,1] = np.max(P_aloc[gstar_idx])
        # fairness of allocations (i.e., variance of distances to goals)
        scores[gstar_idx,2] = np.var(Dist[gstar_idx])
        # ability score of allocations (i.e., human distance to goal)
        scores[gstar_idx,3] = Dist[gstar_idx][0]


    if mode == 'human-robots':
        ext = "_2agt"
    else:
        ext = "_3agt"

    # create save paths and store the data
    savename1 = "../data/"+mode+"/allocations.pkl"
    pickle.dump(G, open(savename1, "wb"))
    savename2 = "../data/"+mode+"/scores.pkl"
    pickle.dump(scores, open(savename2, "wb"))
    savename3 = "../data/"+mode+"/robots_states"+ext+".pkl"
    pickle.dump(states, open(savename3, "wb"))



if __name__ == "__main__":
    main()
