from utils.world import Object, getState, updateState, resetPos
from utils.actions import actionSpace

import pygame
import sys
import os
import numpy as np
import time
import pickle



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


def Legible(team, gstar_idx, gstar, A, G):
    # constrained optimization to find revealing but efficient action
    states = []
    p_aloc = np.ones(len(G))
    step = 1
    while True:
        # hyperparameter for optimization trade-off
        epsilon = 0.03
        s = getState(team)
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

        # compute only the first 15 steps
        if step <= 20:
            p_aloc = np.multiply(p_aloc, bayes(s, astar, A, G))
            if step == 20:
                print("[*] Done!", '\n')
                break

    return p_aloc/np.sum(p_aloc), states
