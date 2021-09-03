import pygame
import sys
import os
import numpy as np
import time

from utils.world import savedGoals, transform

# class for the objects on the screen (agents, goals, etc.)
class Object(pygame.sprite.Sprite):

    def __init__(self, position, color, size, type):

        # create square sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        self.rect = self.image.get_rect()

        if type == 'agent':
            self.image.fill(color)
        elif type == 'goal1':
            radius = size // 2
            pygame.draw.circle(self.image, color, (radius, radius), radius)
        elif type == 'goal2':
            radius = size // 2
            h = int(np.sqrt(size**2-(size/2)**2))
            pygame.draw.polygon(self.image, color, [(h, radius), (radius, radius),
                    (radius, h)])
        else:
            pygame.draw.rect(self.image, color,
            (position[0], 5*position[1], 5*size, size))


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

def coord(x,y):
    "Convert world coordinates to pixel coordinates."
    return (x-0.2, -y+0.5)

def envAgents():
    # initial end-effector positions
    panda_p0 = np.array([ 3.10175333e-01, -4.84159689e-06,  4.87596777e-01])
    fetch_p0 = transform(np.array([0.51211858, 0.21932284, 1.02747358]))
    # add as many agents as you want
    agent_r1 = Object(coord(panda_p0[0], panda_p0[1]), [0, 0, 255], 25, 'agent')
    agent_r2 = Object(coord(float(fetch_p0[0]), float(fetch_p0[1])), [255, 0, 0], 25, 'agent')
    team = [agent_r1, agent_r2]
    return team

def envGoals(task):
    # import Panda's recorded positions
    panda_to_obj1 = savedGoals(task, 'panda', '1')
    panda_to_obj2 = savedGoals(task, 'panda', '2')
    panda_to_obj3 = savedGoals(task, 'panda', '3')
    # define the subtasks and the possible subtask allocations
    goal1 = Object(coord(panda_to_obj1[2][0], panda_to_obj1[2][1]), [255, 153, 0], 50, 'goal1')
    goal2 = Object(coord(panda_to_obj2[2][0], panda_to_obj2[2][1]), [255, 153, 0], 150, 'goal2')
    goal3 = Object(coord(panda_to_obj3[2][0], panda_to_obj3[2][1]), [255, 153, 0], 50, 'goal3')
    goals = [goal1, goal2, goal3]
    # each agent's goal options
    agent1_goal = [list(goal1.state), list(goal2.state), list(goal3.state)]
    agent2_goal = [list(goal1.state), list(goal2.state), list(goal3.state)]
    agent_goals = [agent1_goal, agent2_goal]
    return goals, agent_goals

def initGroup(sprite_list, goals, team):
    for goal in goals:
        sprite_list.add(goal)
    for agent in team:
        sprite_list.add(agent)
