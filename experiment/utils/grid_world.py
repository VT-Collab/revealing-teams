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
        self.image = pygame.Surface((size, size), pygame.SRCALPHA)
        self.rect = self.image.get_rect()
        self.image.fill(color)
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

def savedGoals(task, robot, goal_n):
    filename = "data/"+task+"/"+robot+"_"+goal_n+".pkl"
    waypoints = pickle.load(open(filename, "rb"))
    return waypoints

def transform(p, back_to_fetch=False):
    location = np.concatenate((p,np.array([1])), axis=0)
    # compute the distance between fetch and panda
    panda_to_obj1 = savedGoals('task1', 'panda', '1')
    fetch_to_obj1 = savedGoals('task1', 'fetch', '1')
    dx = panda_to_obj1[1][0]+fetch_to_obj1[1][0]
    dy = fetch_to_obj1[1][1]-abs(panda_to_obj1[1][1])
    if back_to_fetch:
        dz = fetch_to_obj1[1][2]-panda_to_obj1[1][2]
    else:
        dz = panda_to_obj1[1][2]-fetch_to_obj1[1][2]
    # transformation matrix (rotation Z axis + translation)
    T = np.array([[np.cos(np.pi),-np.sin(np.pi),0,dx],
                [np.sin(np.pi),np.cos(np.pi),0,dy],
                [0,0,1,dz],
                [0,0,0,1]])
    p_prime = np.matmul(T,location)
    return p_prime[:3]

def transformToPygame(x,y):
    "Convert world coordinates to pixel coordinates."
    return (x-0.2, -y+0.5)

def transformFromPygame(x,y):
    "Convert world coordinates to pixel coordinates."
    return (x+0.2, 0.5-y)

def envAgents():
    # initial end-effector positions
    panda_p0 = np.array([0.38204478, 0.01169821, 0.24424794])
    fetch_p0 = transform(np.array([0.71579027, 0.19279565, 0.74217811]))
    # add as many agents as you want
    agent_r1 = Object(transformToPygame(panda_p0[0], panda_p0[1]), [0, 0, 255], 25)
    agent_r2 = Object(transformToPygame(float(fetch_p0[0]), float(fetch_p0[1])), [255, 0, 0], 25)
    team = [agent_r1, agent_r2]
    return team

def envGoals(task, team):
    # import Panda's recorded positions
    panda_to_obj1 = savedGoals(task, 'panda', '1')
    panda_to_obj2 = savedGoals(task, 'panda', '2')
    panda_to_obj3 = savedGoals(task, 'panda', '3')
    # define the subtasks and the possible subtask allocations
    goal1 = Object(transformToPygame(panda_to_obj1[1][0], panda_to_obj1[1][1]),
                    [255, 153, 0], 50)
    goal2 = Object(transformToPygame(panda_to_obj2[1][0], panda_to_obj2[1][1]),
                    [255, 153, 0], 50)
    goal3 = Object(transformToPygame(panda_to_obj3[1][0], panda_to_obj3[1][1]),
                    [255, 153, 0], 50)
    goals = [goal1, goal2, goal3]
    # each agent's goal options
    agent1_goal = [list(team[0].state), list(goal1.state), list(goal2.state), list(goal3.state)]
    agent2_goal = [list(team[1].state), list(goal1.state), list(goal2.state), list(goal3.state)]
    agent_goals = [agent1_goal, agent2_goal]
    return goals, agent_goals


def allocations(task):
    # define the subtasks and the possible subtask allocations
    G = {}
    G_ls = []
    goals, agent_goals = envGoals(task, envAgents())
    for idx, goal_a1 in enumerate(agent_goals[0]):
        for idy, goal_a2 in enumerate(agent_goals[1]):
            tau = np.asarray(goal_a1 + goal_a2)
            alloc_name = 'panda' + str(idx) + ' - fetch' + str(idy)
            G[alloc_name] = tau
            if idx != idy:
                G_ls.append(tau)
    # remove same-goal allocations: don't want robots bump to each other
    del G['panda0 - fetch0']
    del G['panda1 - fetch1']
    del G['panda2 - fetch2']
    del G['panda3 - fetch3']
    return G, G_ls

def initGroup(sprite_list, goals, team):
    for goal in goals:
        sprite_list.add(goal)
    for agent in team:
        sprite_list.add(agent)
