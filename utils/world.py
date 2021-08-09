import pygame
import sys
import os
import numpy as np
import time


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

def envAgents():
    # add as many agents as you want
    agent1 = Object((0.1, 0.4), [0, 0, 255], 25)
    agent2 = Object((0.1, 0.6), [0, 255, 0], 25)
    agent3 = Object((0.1, 0.5), [255, 0, 0], 25)
    team = [agent1, agent2, agent3]
    return team

def envGoals():
    # define the subtasks and the possible subtask allocations
    goal1 = Object((1.0, 0.2), [100, 100, 100], 50)
    goal2 = Object((1.0, 0.4), [100, 100, 100], 50)
    goal3 = Object((0.9, 0.9), [100, 100, 100], 50)
    goals = [goal1, goal2, goal3]
    # each agent's goal options
    agent1_goal = [list(goal1.state), list(goal2.state), list(goal3.state)]#, list(agent1.state)]
    agent2_goal = [list(goal1.state), list(goal2.state), list(goal3.state)]#, list(agent2.state)]
    agent3_goal = [list(goal1.state), list(goal2.state), list(goal3.state)]#, list(agent3.state)]
    agent_goals = [agent1_goal, agent2_goal, agent3_goal]
    return goals, agent_goals

def initGroup(sprite_list, goals, team):
    for goal in goals:
        sprite_list.add(goal)
    for agent in team:
        sprite_list.add(agent)
