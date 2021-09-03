import numpy as np
import random
import time
import pickle
import sys

from utils.robot_actions import actionSpace
from utils.grid_world import *


# bayes rule with boltzmann rational
def bayes(s, a, A, G, beta=20.0):
    P = []
    for g in G:
        num = np.exp(-beta * np.linalg.norm(g - (s + a)))
        den = 0
        for ap in A:
            den += np.exp(-beta * np.linalg.norm(g - (s + ap)))
        P.append(num / den)
    P = np.asarray(P)
    return P / sum(P)


def legibleRobots(team, gstar_idx, gstar, A, G):
    # constrained optimization to find revealing but efficient action
    states = []
    step_max = 5
    step = 1
    p_aloc = np.ones(len(G))
    while True:
        print(step)
        s = getState(team)
        epsilon = 0.02
        Q = {}
        Qmax = -np.Inf
        for a in A:
            Q[str(a)] = np.linalg.norm(gstar - s) - np.linalg.norm(gstar - (s + a))
            if Q[str(a)] > Qmax:
                Qmax = Q[str(a)]
        value = -np.Inf
        astar = None
        for a in A:
            likelihood = bayes(s, a, A, G)
            if likelihood[gstar_idx] > value and Qmax - Q[str(a)] < epsilon:
                astar = np.copy(a)
                value = likelihood[gstar_idx]

        # update for next time step
        updateState(team, s + astar)
        states.append(s+astar)
        step +=1

        # if step <= step_max:
        p_aloc = np.multiply(p_aloc, bayes(s, astar, A, G))
        if step == step_max:
            print("[*] Done!", '\n')
            break

    return p_aloc/np.sum(p_aloc), states


def main():
    task = sys.argv[1]
    team_loc = envAgents()
    goals, agent_goals = envGoals(task)
    G, G_ls = allocations(task)
    A = actionSpace()

    # main loop
    states = []
    scores = np.empty([len(G),4], dtype=object)
    P_aloc = np.empty([len(G),len(G)])
    gstar_idx = 0

    for key, positions in G.items():
        # pick the desired allocation
        gstar = np.copy(G[key])
        print('[*] Allocation ', gstar_idx+1,':', key)

        # compute distance to goals for each agent
        Dist = np.empty([len(G),len(team_loc)])
        s0 = list(team_loc[0].state) + list(team_loc[1].state)
        dist = abs(s0-gstar)
        dist_normed = []
        for idx in range(len(dist)):
          if idx % 2 == 0:
            dist_normed.append(np.linalg.norm(dist[idx:idx+2]))
        Dist[gstar_idx] = dist_normed

        # legible robot motion f
        P_aloc[gstar_idx], states_r = legibleRobots(team_loc, gstar_idx, gstar, A, G_ls)
        states.append(states_r)

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
    savename3 = "data/"+task+"/robots_states.pkl"
    pickle.dump(states, open(savename3, "wb"))

if __name__=="__main__":
    main()
