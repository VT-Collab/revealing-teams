from utils.world import Object, getState, updateState, resetPos
from utils.actions import actionSpace

import pygame
import sys
import os
import numpy as np
import time
import pickle
import random


step_max = 30

# likelihood of human action given state s and allocation g
# input is state, goal, beta; output is sampled action
def bayes_actor(s, A, g, beta):
    P = []
    for a in A:
        num = np.exp(-beta * np.linalg.norm(g - (s + a)))
        P.append(num)
    P = np.asarray(P)
    P /= sum(P)
    #monte carlo
    indices = np.arange(A.shape[0])
    action = A[np.random.choice(indices, p=P)]
    return action


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


def legibleRobots(mode, team, gstar_idx, gstar, A, G):
    # constrained optimization to find revealing but efficient action
    states = []
    step = 1

    if mode == 'human-robots':
        # slice robots' action space, task allocation
        team = team[1:]
        gstar = gstar[2:]
        A = A[:,2:]
        G = list(np.asarray(G)[:,2:])

    p_aloc = np.ones(len(G))
    while True:
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

        # compute only the first 15 steps
        if step <= step_max:
            p_aloc = np.multiply(p_aloc, bayes(s, astar, A, G))
            if step == step_max:
                print("[*] Done!", '\n')
                break

    return p_aloc/np.sum(p_aloc), states


def humanAgent(team, gstar_idx, gstar, A, beta = 0):
    # constrained optimization to find revealing but efficient action
    states = []
    step = 1

    # slice human's action space, task allocation
    A = A[:,:2]
    gstar = gstar[:2]
    team = team[:1]

    while True:
        s = getState(team)

        astar = bayes_actor(s, A, gstar, beta)

        # update for next time step
        updateState(team, s + astar)
        states.append(s + astar)
        step +=1

        # compute only the first 15 steps
        if step == step_max:
            print("[*] Done!", '\n')
            break

    return states
