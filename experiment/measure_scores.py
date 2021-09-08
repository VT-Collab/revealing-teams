import numpy as np
import random
import time
import pickle
import sys

from utils.robot_actions import actionSpace
from utils.grid_world import *


# bayes rule with boltzmann rational
def bayes(s, a, A, G_ls, beta=20.0):
    P = []
    for g in G_ls:
        num = np.exp(-beta * np.linalg.norm(g - (s + a)))
        den = 0
        for ap in A:
            den += np.exp(-beta * np.linalg.norm(g - (s + ap)))
        P.append(num / den)
    P = np.asarray(P)
    return P / sum(P)


def legibleRobots(team, gstar_idx, gstar, A, G_ls):
    # constrained optimization to find revealing but efficient action
    states_panda = []
    states_fetch = []
    step = 1
    step_max = 30
    p_aloc = np.ones(len(G_ls))
    while True:
        print('---Step: ',step)
        s = getState(team)
        epsilon = 0.02
        if step > 20:
            epsilon = 0.00001
        Q = {}
        Qmax = -np.Inf
        for a in A:
            Q[str(a)] = np.linalg.norm(gstar - s) - np.linalg.norm(gstar - (s + a))
            if Q[str(a)] > Qmax:
                Qmax = Q[str(a)]
        value = -np.Inf
        astar = None
        for a in A:
            likelihood = bayes(s, a, A, G_ls)
            if likelihood[gstar_idx] > value and Qmax - Q[str(a)] < epsilon:
                astar = np.copy(a)
                value = likelihood[gstar_idx]

        threshold = 0.001
        if np.linalg.norm(gstar[:2] - s[:2]) < threshold:
            astar[:2] = [0,0]
        elif np.linalg.norm(gstar[2:] - s[2:]) < threshold:
            astar[2:] = [0,0]

        # update for next time step
        updateState(team, s + astar)
        new_state = s + astar
        states_panda.append(new_state[:2])
        states_fetch.append(new_state[2:])

        if step <= step_max:
            p_aloc = np.multiply(p_aloc, bayes(s, astar, A, G_ls))
            # if np.linalg.norm(astar) < 0.0001:
            if step == step_max:
                print("[*] Done!", '\n')
                break
        step +=1

    return p_aloc/np.sum(p_aloc), states_panda, states_fetch


def main():
    task = sys.argv[1]
    team = envAgents()
    goals, agent_goals = envGoals(task, team)
    G, G_ls = allocations(task)
    A = actionSpace()

    # the game will draw everything in the sprite list
    sprite_list = pygame.sprite.Group()
    initGroup(sprite_list, goals, team)

    # main loop
    States_panda = []
    States_fetch = []
    scores = np.empty([len(G),4], dtype=object)
    P_aloc = np.empty([len(G),len(G)])
    gstar_idx = 0

    for key, positions in G.items():
        resetPos(team)
        # pick the desired allocation
        gstar = np.copy(G[key])
        print('[*] Allocation ', gstar_idx+1,':', key)

        # compute distance to goals for each agent
        Dist = np.empty([len(G),len(team)])
        dist = abs(getState(team)- gstar)
        dist_normed = []
        for idx in range(len(dist)):
          if idx % 2 == 0:
            dist_normed.append(np.linalg.norm(dist[idx:idx+2]))
        Dist[gstar_idx] = dist_normed

        # legible robot motion f
        P_aloc[gstar_idx], states_panda, states_fetch = legibleRobots(team, gstar_idx, gstar, A, G_ls)
        States_panda.append(states_panda)
        States_fetch.append(states_fetch)

        # index allocations
        scores[gstar_idx,0] = key
        # legibility of allocations
        scores[gstar_idx,1] = np.max(P_aloc[gstar_idx])
        # fairness of allocations (i.e., variance of distances to goals)
        scores[gstar_idx,2] = np.var(Dist[gstar_idx])
        # ability score of allocations (i.e., human distance to goal)
        scores[gstar_idx,3] = Dist[gstar_idx][0]

        gstar_idx += 1

    # create save paths and store the data
    savename1 = "data/"+task+"/allocations.pkl"
    pickle.dump(G, open(savename1, "wb"))
    savename2 = "data/"+task+"/scores.pkl"
    pickle.dump(scores, open(savename2, "wb"))
    savename3 = "data/"+task+"/States_panda.pkl"
    pickle.dump(States_panda, open(savename3, "wb"))
    savename4 = "data/"+task+"/States_fetch.pkl"
    pickle.dump(States_fetch, open(savename4, "wb"))

if __name__=="__main__":
    main()
