import numpy as np
import random
import time
import pickle


from utils.robot_actions import actionSpace
from utils.world import envAgents, envGoals, allocations, updateState

# s: just update in the main loop with simple np summation
# a: are basically elements of the action space
# A: it's generated for two robots
# G: pool of allocations




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


def legibleRobots(team_loc, gstar_idx, gstar, A, G):
    # constrained optimization to find revealing but efficient action
    s = list(team_loc[0]) + list(team_loc[1])
    states = []
    step = 1
    p_aloc = np.ones(len(G))
    step_max = 3

    while True:
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
        s = updateState(team_loc, astar)
        states.append(s)
        step +=1

        if step <= step_max:
            p_aloc = np.multiply(p_aloc, bayes(s, astar, A, G))
            if step == step_max:
                print("[*] Done!", '\n')
                break

    return p_aloc/np.sum(p_aloc), states


def main():
    team_loc = envAgents()
    goals, agent_goals = envGoals()
    G = allocations()
    A = actionSpace()

    # main loop
    states = []
    scores = np.empty([len(G),4])
    P_aloc = np.empty([len(G),len(G)])

    for gstar_idx in range(len(G)):
        # pick the desired allocation
        gstar = np.copy(G[gstar_idx])
        print('[*] Allocation: ', gstar_idx+1)

        # # compute distance to goals for each agent
        # Dist = np.empty([len(G),len(team)])
        # dist = abs(getState(team)- gstar)
        # dist_normed = []
        # for idx in range(len(dist)):
        #   if idx % 2 == 0:
        #     dist_normed.append(np.linalg.norm(dist[idx:idx+2]))
        # Dist[gstar_idx] = dist_normed

        # legible robot motion
        P_aloc[gstar_idx], states_r = legibleRobots(team_loc, gstar_idx, gstar, A, G)
        # states.append(states_r)

        # # index allocations
        # scores[gstar_idx,0] = gstar_idx + 1
        # # legibility of allocations
        # scores[gstar_idx,1] = np.max(P_aloc[gstar_idx])
        # # fairness of allocations (i.e., variance of distances to goals)
        # scores[gstar_idx,2] = np.var(Dist[gstar_idx])
        # # ability score of allocations (i.e., human distance to goal)
        # scores[gstar_idx,3] = Dist[gstar_idx][0]

if __name__=="__main__":
    main()
