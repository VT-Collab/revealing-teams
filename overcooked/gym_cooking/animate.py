from misc.game.gameplay import GamePlay
from utils.interact import interact
from utils.parse import *

import numpy as np
import random
import argparse
from collections import namedtuple
import time
import gym
import pygame
import pickle
import sys


folder = 'fairness'
# folder = 'legibility'

def main():
    pygame.init()

    # command line inputs
    arglist = parse_arguments()

    if folder == 'fairness':
        n_allocations = 9
    else:
        n_allocations = 10
    for idx in range(n_allocations):
        print('[*] Allocation ' +str(idx+1))
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        # visualize
        game = GamePlay(env.filename, env.world, env.sim_agents)
        game.on_init()
        game.on_render()

        # import saved data
        filename = "data/"+folder+"/actions_"+str(idx+1)+".pkl"
        agents_actions = pickle.load(open(filename, "rb"))
        agent1_actions = agents_actions['agent-1']
        agent2_actions = agents_actions['agent-2']
        agent3_actions = agents_actions['agent-3']

        lengths = []
        for key in agents_actions:
            lengths.append(len(agents_actions[key]))
        steps = max(lengths)
        team = game.sim_agents
        names = ['agent-1', 'agent-2', 'agent-3']
        for step in range(steps):
            image_name = '{}_{}_{}_{}.png'.format('alloc', str(idx+1), 'frame', str(step+1))
            pygame.image.save(game.screen, '{}/{}'.format('screenshots/'+folder, image_name))
            for j, agent in enumerate(team):
                agent_actions = agents_actions[names[j]]
                if step+1 <= len(agent_actions):
                    agent.action = agent_actions[step]
                else:
                    agent.action = (0,0)
                interact(agent, env.world)

            time.sleep(0.1)
            game.on_render()
        image_name = '{}_{}_{}_{}.png'.format('alloc', str(idx+1), 'frame', str(step+1))
        pygame.image.save(game.screen, '{}/{}'.format('screenshots/'+folder, image_name))



if __name__ == '__main__':
    main()
