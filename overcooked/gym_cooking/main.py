from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag
from utils.parse import *
from utils.allocation import combination

import numpy as np
import random
import argparse
from collections import namedtuple
import time
import gym
import pygame



folder = 'fairness'
# folder = 'legibility'

def actionSpace():
    # discretized actions
    n_actions = 5
    single_action = [[0, 1], [0, -1], [-1, 0], [1, 0], [0, 0]]
    A = []
    for idx in range(n_actions):
        for jdx in range(n_actions):
            for kdx in range(n_actions):
                A.append(list(single_action[idx]) + list(single_action[jdx]) + list(single_action[kdx]))
    A = np.asarray(A)
    return A


# returns n x 1 vector containing the agent positions
def getState(team):
    s = []
    for agent in team:
        s += list(agent.location)
    return np.asarray(s)


def envGoals(env):
    # agent locations
    agent1_loc = list(env.sim_agents[0].location)
    agent2_loc = list(env.sim_agents[1].location)
    agent3_loc = list(env.sim_agents[2].location)

    # locations of objects
    goal1 = list(env.world.objects['Plate'][0].location)
    goal2 = list(env.world.objects['Tomato'][0].location)
    goal3 = list(env.world.objects['Lettuce'][0].location)
    # goal3 = list(env.world.objects['Cutboard'][0].location)
    # pos_d = list(env.world.objects['Delivery'][0].location)

    G = combination(goal1, goal2, goal3, agent1_loc, agent2_loc, agent3_loc, folder)
    return G

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


def legibleRobots(team, gstar_idx, gstar, A, allocations, agents_actions):
    # constrained optimization to find revealing but efficient action
    states = []

    agent1_actions = agents_actions['agent-1']
    agent2_actions = agents_actions['agent-2']
    agent3_actions = agents_actions['agent-3']

    lengths = []
    for key in agents_actions:
        lengths.append(len(agents_actions[key]))
    steps = max(lengths)

    names = ['agent-1', 'agent-2', 'agent-3']
    step_max = 5
    p_aloc = np.ones(len(allocations))
    for step in range(steps):
        s = []
        astar = []
        # image_name = '{}_{}_{}_{}.png'.format('alloc', str(idx+1), 'frame', str(step+1))
        # pygame.image.save(game.screen, '{}/{}'.format('screenshots', image_name))
        for j, agent in enumerate(team):
            agent_actions = agents_actions[names[j]]
            if step+1 <= len(agent_actions):
                agent.action = agent_actions[step]
            else:
                agent.action = (0,0)
            # interact(agent, env.world)

            s += list(agent.location)
            astar += list(agent.action)
        s = np.array(s)
        astar = np.array(astar)

        if step <= step_max:
            p_aloc = np.multiply(p_aloc, bayes(s, astar, A, allocations))
            if step == step_max:
                print("[*] Done!", '\n')
                break

    return p_aloc/np.sum(p_aloc)


def savedActions(gstar_idx):
    filename = "data/"+folder+"/actions_"+str(gstar_idx+1)+".pkl"
    actions = pickle.load(open(filename, "rb"))
    return actions

def main():
    pygame.init()

    # command line inputs
    arglist = parse_arguments()
    # initialize the environment
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    env.reset()
    team = env.sim_agents
    mode = 'robots'
    allocations = envGoals(env)
    A = actionSpace()

    scores = np.empty([len(allocations),4])
    P_aloc = np.empty([len(allocations),len(allocations)])

    for gstar_idx in range(len(allocations)):
        env.reset()

        # pick the desired allocation
        gstar = np.copy(allocations[gstar_idx])
        print('[*] Allocation: ', gstar_idx+1)

        # compute distance to goals for each agent
        Dist = np.empty([len(allocations),len(team)])
        dist = abs(getState(team)- gstar)
        dist_normed = []
        for idx in range(len(dist)):
          if idx % 2 == 0:
            dist_normed.append(np.linalg.norm(dist[idx:idx+2]))
        Dist[gstar_idx] = dist_normed

        agents_actions = savedActions(gstar_idx)
        P_aloc[gstar_idx] = legibleRobots(team, gstar_idx, gstar, A, allocations, agents_actions)

        # index allocations
        scores[gstar_idx,0] = gstar_idx + 1
        # legibility of allocations
        scores[gstar_idx,1] = np.max(P_aloc[gstar_idx])
        # fairness of allocations (i.e., variance of distances to goals)
        scores[gstar_idx,2] = np.var(Dist[gstar_idx])
        # ability score of allocations (i.e., human distance to goal)
        scores[gstar_idx,3] = Dist[gstar_idx][0]

        # sort scores in descending order, ranked by legibility
        ranked_scores = scores[scores[:, 1].argsort()]
        ranked_scores = ranked_scores[::-1]

    print('[*] Ranked based on legibility: ','\n',ranked_scores)
    # print(scores)
if __name__ == '__main__':
    main()
