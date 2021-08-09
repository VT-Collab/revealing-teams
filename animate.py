from utils.world import Object, getState, updateState, resetPos
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

    # the game will draw everything in the sprite list
    sprite_list = pygame.sprite.Group()
    sprite_list.add(goal1)
    sprite_list.add(goal2)
    sprite_list.add(goal3)
    sprite_list.add(agent1)
    sprite_list.add(agent2)
    sprite_list.add(agent3)

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
    print('[*] Ranked based on legibility: ',ranked_scores)

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


        if True:#item[1] > 0.7 and item[2] < 0.05:
            gstar_idx = int(item[0]-1)

            # pick the desired allocation
            gstar = np.copy(allocations[gstar_idx])
            print('[*] Allocation: ', gstar_idx)


            # aniamte the environment
            animate(sprite_list, team, states[gstar_idx])


if __name__ == "__main__":
    main()
