from utils.grid_world import (
        Object, getState, updateState,
        resetPos, envAgents, envGoals, initGroup
)
from utils.robot_actions import actionSpace

import pygame
import sys
import os
import numpy as np
import time
import pickle


def animate(sprite_list, team, states_panda, states_fetch, gstar_idx):
    world = pygame.display.set_mode([700,700])
    # create game
    clock = pygame.time.Clock()
    fps = 10
    # animate
    world.fill((255,255,255))
    sprite_list.draw(world)
    pygame.display.flip()
    clock.tick(fps)
    frame = 1
    image_name = '{}_{}_{}_{}.png'.format('alloc', str(gstar_idx), 'frame', '0')
    # pygame.image.save(world, '{}/{}'.format('screenshots', image_name))
    for i in range(len(states_panda)):
        state = np.concatenate((states_panda[i],states_fetch[i]),axis=0)
        # update for next time step
        updateState(team, state)
        # animate
        world.fill((255,255,255))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)
        time.sleep(0.05)
        image_name = '{}_{}_{}_{}.png'.format('alloc', str(gstar_idx), 'frame', str(frame))
        # pygame.image.save(world, '{}/{}'.format('screenshots', image_name))
        frame += 1

def savedData(task, file):
    filename = "data/"+task+"/"+file+".pkl"
    data = pickle.load(open(filename, "rb"))
    return data

def main():
    # pick which task to simulate
    task = sys.argv[1]

    # create game
    pygame.init()
    A = actionSpace()
    team = envAgents()
    goals,_ = envGoals(task, team)

    # the game will draw everything in the sprite list
    sprite_list = pygame.sprite.Group()
    initGroup(sprite_list, goals, team)

    # import score and allocation saved data
    allocations = savedData(task, 'allocations')
    scores = savedData(task, 'scores')

    # import states saved data
    states_panda = savedData(task, 'States_panda')
    states_fetch = savedData(task, 'States_fetch')

    # sort scores in descending order, ranked by legibility
    ranked_scores = scores[scores[:, 1].argsort()]
    ranked_scores = ranked_scores[::-1]
    # print('[*] Ranked based on legibility: ','\n',ranked_scores)

    # # sort based on fairness
    # ranked_scores = ranked_scores[ranked_scores[:,2].argsort(kind='mergesort')]
    # print('[*] Ranked based on fairness: ','\n',ranked_scores)

    # main loop
    for gstar_idx, item in enumerate(ranked_scores):
        resetPos(team)
        print('[*] Allocation: ', gstar_idx)
        # aniamte the environment
        animate(sprite_list, team, states_panda[gstar_idx], states_fetch[gstar_idx], gstar_idx)

if __name__ == "__main__":
    main()
# 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
