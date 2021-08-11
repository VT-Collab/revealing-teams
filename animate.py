from utils.world import (
        Object, getState, updateState,
        resetPos, envAgents, envGoals, initGroup
)
from utils.actions import actionSpace

import pygame
import sys
import os
import numpy as np
import time
import pickle


def animate(sprite_list, team, states):
    world = pygame.display.set_mode([700,700])
    # create game
    clock = pygame.time.Clock()
    fps = 10
    # animate
    world.fill((255,255,255))
    sprite_list.draw(world)
    pygame.display.flip()
    clock.tick(fps)

    for state in states:
        # update for next time step
        updateState(team, state)
        # animate
        world.fill((255,255,255))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)
        time.sleep(0.2)

def savedStates(mode, beta):
    if mode == 'human-robots':
        print("[*]beta: ", beta,'\n')
        filename3 = "data/"+mode+"/human_states_"+str(beta)+".pkl"
        states_h = pickle.load(open(filename3, "rb"))
        filename4 = "data/"+mode+"/robots_states_2agt.pkl"
        states_r = pickle.load(open(filename4, "rb"))
        states = [list(np.concatenate(
            (np.vstack(states_h), np.vstack(states_r)), axis=1))]
    else:
        filename3 = "data/"+mode+"/robots_states_3agt.pkl"
        states = pickle.load(open(filename3, "rb"))
    return states


def main():
    # define the structure of the team
    mode = sys.argv[1]

    # human rationality
    betas = [0, 50, 100, 500, 800]

    # create game
    pygame.init()
    A = actionSpace()
    team = envAgents()
    goals,_ = envGoals()

    # the game will draw everything in the sprite list
    sprite_list = pygame.sprite.Group()
    initGroup(sprite_list, goals, team)

    # import score and allocation saved data
    filename1 = "data/"+mode+"/allocations.pkl"
    allocations = pickle.load(open(filename1, "rb"))
    filename2 = "data/"+mode+"/scores.pkl"
    scores = pickle.load(open(filename2, "rb"))

    for beta in betas:
        # import states saved data
        states = savedStates(mode, beta)

        # sort scores in descending order, ranked by legibility
        ranked_scores = scores[scores[:, 1].argsort()]
        ranked_scores = ranked_scores[::-1]
        # print('[*] Ranked based on legibility: ','\n',ranked_scores)

        # # sort based on fairness
        # ranked_scores = ranked_scores[ranked_scores[:,2].argsort(kind='mergesort')]
        # print('[*] Ranked based on fairness: ',ranked_scores)

        # main loop
        for item in ranked_scores:
            resetPos(team)
            # convert allocation # to python indices
            gstar_idx = int(item[0]-1)
            # print('[*] Allocation: ', gstar_idx)

            # aniamte the environment
            animate(sprite_list, team, states[gstar_idx])


if __name__ == "__main__":
    main()
