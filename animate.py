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
        time.sleep(0.3)


def main():
    # create game
    pygame.init()
    A = actionSpace()
    team = envAgents()
    goals,_ = envGoals()

    # the game will draw everything in the sprite list
    sprite_list = pygame.sprite.Group()
    initGroup(sprite_list, goals, team)

    # import the possible questions we have saved
    filename1 = "data/allocations.pkl"
    allocations = pickle.load(open(filename1, "rb"))
    filename2 = "data/scores.pkl"
    scores = pickle.load(open(filename2, "rb"))
    filename3 = "data/states.pkl"
    states = pickle.load(open(filename3, "rb"))

    # sort scores in descending order, ranked by legibility
    ranked_scores = scores[scores[:, 1].argsort()]
    ranked_scores = ranked_scores[::-1]
    print('[*] Ranked based on legibility: ','\n',ranked_scores)

    # remove the case of no moving agents
    # ranked_scores = ranked_scores[1:,:]

    # # sort based on fairness
    # ranked_scores = ranked_scores[ranked_scores[:,2].argsort(kind='mergesort')]
    # print('[*] Ranked based on fairness: ',ranked_scores)

    # slice the first 5
    # ranked_scores = ranked_scores[25,:]


    # main loop
    for item in ranked_scores:
        resetPos(team)

        gstar_idx = int(item[0]-1)

        # pick the desired allocation
        gstar = np.copy(allocations[gstar_idx])
        print('[*] Allocation: ', gstar_idx)

        # aniamte the environment
        animate(sprite_list, team, states[gstar_idx])


if __name__ == "__main__":
    main()
