import numpy as np
import random
import time
import pickle



def savedGoals(task, robot, goal_n):
    filename = "data/"+task+"/"+robot+"_"+goal_n+".pkl"
    waypoints = pickle.load(open(filename, "rb"))
    return waypoints

def transform(p, back_to_fetch=False):
    location = np.concatenate((p,np.array([1])), axis=0)
    # compute the distance between fetch and panda
    panda_to_obj1 = savedGoals('task1', 'panda', '1')
    fetch_to_obj1 = savedGoals('task1', 'fetch', '1')
    dx = panda_to_obj1[2][0]+fetch_to_obj1[2][0]
    dy = fetch_to_obj1[2][1]-panda_to_obj1[2][1]
    if back_to_fetch:
        dz = fetch_to_obj1[2][2]
    else:
        dz = -fetch_to_obj1[2][2]
    # transformation matrix (rotation Z axis + translation)
    T = np.array([[np.cos(np.pi),-np.sin(np.pi),0,dx],
                [np.sin(np.pi),np.cos(np.pi),0,dy],
                [0,0,1,dz],
                [0,0,0,1]])
    p_prime = np.matmul(T,location)
    return p_prime[:3]


def envAgents():
    # initial end-effector positions
    panda_p0 = np.array([0.38204478, 0.01169821, 0.24424794])
    fetch_p0 = np.array([0.71579027, 0.19279565, 0.74217811])
    team_loc = [panda_p0, transform(fetch_p0)]
    return team_loc


def envGoals(task):
    # import Panda's recorded positions
    panda_to_obj1 = savedGoals(task, 'panda', '1')
    panda_to_obj2 = savedGoals(task, 'panda', '2')
    panda_to_obj3 = savedGoals(task, 'panda', '3')
    # the location of each goal from panda
    goal1 = panda_to_obj1[1][:2]
    goal2 = panda_to_obj2[1][:2]
    goal3 = panda_to_obj3[1][:2]
    goals = [goal1, goal2, goal3]
    # each agent's goal options
    panda_goals = [list(goal1), list(goal2), list(goal3)]
    fetch_goals = [list(goal1), list(goal2), list(goal3)]
    agent_goals = [panda_goals, fetch_goals]
    return goals, agent_goals


def allocations(task):
    # define the subtasks and the possible subtask allocations
    G = {}
    G_ls = []
    goals, agent_goals = envGoals(task)
    for idx, goal_a1 in enumerate(agent_goals[0]):
        for idy, goal_a2 in enumerate(agent_goals[1]):
            tau = np.asarray(goal_a1 + goal_a2)
            alloc_name = 'panda' + str(idx+1) + ' , fetch' + str(idy+1)
            G[alloc_name] = tau
            if idx != idy:
                G_ls.append(tau)
    # remove same-goal allocations: don't want robots bump to each other
    del G['panda1 , fetch1']
    del G['panda2 , fetch2']
    del G['panda3 , fetch3']
    return G, G_ls


# sets the location of the agents
def updateState(team_loc, astar):
    s = []
    for idx in range(len(team_loc)):
        action_idx = [astar[idx*2], astar[idx*2 + 1]]
        s_idx = team_loc[idx] + action_idx
        s += list(s_idx)
    return s
