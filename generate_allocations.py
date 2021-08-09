from utils.world import Object, getState, updateState, resetPos
from utils.actions import actionSpace
from navigation_planner.legible import bayes, Legible

import pygame
import sys
import os
import numpy as np
import time
import pickle



def main():
    # create game
    pygame.init()
    A = actionSpace()

    # add as many agents as you want
    agent1 = Object((0.1, 0.4), [0, 0, 255], 25)
    agent2 = Object((0.1, 0.6), [0, 255, 0], 25)
    agent3 = Object((0.1, 0.5), [255, 0, 0], 25)
    team = [agent1, agent2, agent3]

    # define the subtasks and the possible subtask allocations
    goal1 = Object((1.0, 0.4), [100, 100, 100], 50)
    goal2 = Object((1.0, 0.6), [100, 100, 100], 50)
    goal3 = Object((0.5, 1), [100, 100, 100], 50)

    # each agent's goal options
    agent1_goals = [list(goal1.state), list(goal2.state), list(goal3.state)]#, list(agent1.state)]
    agent2_goals = [list(goal1.state), list(goal2.state), list(goal3.state)]#, list(agent2.state)]
    agent3_goals = [list(goal1.state), list(goal2.state), list(goal3.state)]#, list(agent3.state)]

    # pool of allocations
    G = []
    for goal_a1 in agent1_goals:
        for goal_a2 in agent2_goals:
            for goal_a3 in agent3_goals:
                tau = np.asarray(goal_a1 + goal_a2 + goal_a3)
                G.append(tau)

    # the game will draw everything in the sprite list
    sprite_list = pygame.sprite.Group()
    sprite_list.add(goal1)
    sprite_list.add(goal2)
    sprite_list.add(goal3)
    sprite_list.add(agent1)
    sprite_list.add(agent2)
    sprite_list.add(agent3)


    # main loop
    states_aloc = []
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

        # legible motion
        P_aloc[gstar_idx], states = Legible(team, gstar_idx, gstar, A, G)
        states_aloc.append(states)

        # index, legibility score, and fairness score of each allocation
        scores[gstar_idx,0] = gstar_idx + 1
        scores[gstar_idx,1] = np.max(P_aloc[gstar_idx])
        scores[gstar_idx,2] = np.var(fairness[gstar_idx])


    # create save paths and store the data
    savename1 = 'data/allocations.pkl'
    pickle.dump(G, open(savename1, "wb"))
    savename2 = 'data/scores.pkl'
    pickle.dump(scores, open(savename2, "wb"))
    savename3 = 'data/states.pkl'
    pickle.dump(states_aloc, open(savename3, "wb"))



if __name__ == "__main__":
    main()
