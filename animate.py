import pygame
import sys
import os
import numpy as np
import time
import pickle


# class for the objects on the screen (agents, goals, etc.)
class Object(pygame.sprite.Sprite):

    def __init__(self, position, color, size):

        # create square sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((size, size))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        # self.reset()

        # initial conditions
        self.start_x = position[0]
        self.start_y = position[1]
        self.state = np.asarray([self.start_x, self.start_y])
        self.rect.x = int((self.start_x * 500) + 100 - self.rect.size[0] / 2)
        self.rect.y = int((self.start_y * 500) + 100 - self.rect.size[1] / 2)

    def reset(self):
        s = [self.start_x, self.start_y]
        return s

    # move the sprite to position s
    def update(self, s):
        self.rect = self.image.get_rect(center=self.rect.center)
        self.x = s[0]
        self.y = s[1]
        self.state = np.asarray([self.x, self.y])
        self.rect.x = int((self.x * 500) + 100 - self.rect.size[0] / 2)
        self.rect.y = int((self.y * 500) + 100 - self.rect.size[1] / 2)


# returns n x 1 vector containing the agent positions
def getState(group):
    s = []
    for agent in group:
        s += list(agent.state)
    return np.asarray(s)

# sets the location of the agents
def updateState(group, newstate):
    n_agents = len(group)
    for idx in range(n_agents):
        s_idx = [newstate[idx * 2], newstate[idx * 2 + 1]]
        group[idx].update(s_idx)

def resetPos(team):
    # initialize team
    for idx, agent in enumerate(team):
        agent_reset = agent.update(agent.reset())


# bayes rule with boltzmann rational
def bayes(s, a, A, allocations, beta=20.0):
    P = []
    for g in allocations:
        num = np.exp(-beta * np.linalg.norm(g - (s + a)))
        den = 0
        for ap in A:
            den += np.exp(-beta * np.linalg.norm(g - (s + ap)))
        P.append(num / den)
    P = np.asarray(P)
    return P / sum(P)


# discretize the space of actions for the team
n_actions = 5
r = 0.05
single_action = []
angles = np.linspace(-np.pi/2, np.pi/2, n_actions)
for angle in angles:
    action = [r * np.cos(angle), r * np.sin(angle)]
    single_action.append(action)
single_action.append([0,0])
A = []
for idx in range(n_actions):
    for jdx in range(n_actions):
        for kdx in range(n_actions):
            A.append(list(single_action[idx]) +
            list(single_action[jdx])+ list(single_action[kdx]))
A = np.asarray(A)


def Legible(team, gstar_idx, gstar, A, allocations):
    # constrained optimization to find revealing but efficient action
    states = []
    p_aloc = np.ones(len(allocations))
    step = 1
    while True:
        # hyperparameter for optimization trade-off
        epsilon = 0.01
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
            likelihood = bayes(s, a, A, allocations)
            if likelihood[gstar_idx] > value and Qmax - Q[str(a)] < epsilon:
                astar = np.copy(a)
                value = likelihood[gstar_idx]

        # update for next time step
        updateState(team, s + astar)
        states.append(s+astar)
        step +=1

        # compute only the first 15 steps
        if step == 10:
            print("[*] Done!", '\n')
            break

        # # determine when to switch to the next allocation
        # if np.all(abs(s - gstar) < 0.1):
        #     print("[*] Done!", '\n')
        #     # pygame.quit(); sys.exit()
        #     break

    return states


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


def main():
    # create game
    pygame.init()

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
    filename1 = "Data/allocations.pkl"
    allocations = pickle.load(open(filename1, "rb"))
    filename2 = "Data/scores.pkl"
    scores = pickle.load(open(filename2, "rb"))

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

            # legible motion
            states = Legible(team, gstar_idx, gstar, A, allocations)

            # aniamte the environment
            animate(sprite_list, team, states)


if __name__ == "__main__":
    main()
