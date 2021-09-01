import numpy as np
import random
import time
import pickle



def savedGoals(task, robot, goal_n):
    filename = "data/"+task+"/"+robot+"_"+goal_n+".pkl"
    waypoints = pickle.load(open(filename, "rb"))
    return waypoints


def location(point, robot):
    if robot == 'panda':
        # let's set the global reference at object 1 based on Panda recorded positions
        # we use this to find the distance from
        # object 1 to Panda's home, object 2, and object 3
        panda_to_obj1 = savedGoals('task1', 'panda', '1')
        obj1_pos = panda_to_obj1[2]
    if robot == 'fetch':
        # compute the fetch's home position from object 1
        fetch_to_obj1 = savedGoals('task1', 'fetch', '1')
        obj1_pos = fetch_to_obj1[2]
    position = point - obj1_pos
    return position


def envAgents():
    # initial end-effector positions
    fetch_p0 = [0.51211858, 0.21932284, 1.02747358]
    panda_p0 = [ 3.10175333e-01, -4.84159689e-06,  4.87596777e-01]
    # add as many agents as you want
    panda_loc = location(panda_p0, 'panda')
    fetch_loc = location(fetch_p0, 'fetch')
    team_loc = [panda_loc, fetch_loc]
    return team_loc


def envGoals():
    # import Panda's recorded positions
    panda_to_obj1 = savedGoals('task1', 'panda', '1')
    panda_to_obj2 = savedGoals('task1', 'panda', '2')
    panda_to_obj3 = savedGoals('task1', 'panda', '3')
    # compute the location of each goal wrt object 1
    goal1 = location(panda_to_obj1[2], 'panda')
    goal2 = location(panda_to_obj2[2], 'panda')
    goal3 = location(panda_to_obj3[2], 'panda')
    goals = [goal1, goal2, goal3]
    # each agent's goal options
    panda_goals = [list(goal1), list(goal2), list(goal3)]
    fetch_goals = [list(goal1), list(goal2), list(goal3)]
    agent_goals = [panda_goals, fetch_goals]
    return goals, agent_goals


def allocations():
    # define the subtasks and the possible subtask allocations
    G = []
    goals, agent_goals = envGoals()
    for goal_a1 in agent_goals[0]:
        for goal_a2 in agent_goals[1]:
            tau = np.asarray(goal_a1 + goal_a2)
            G.append(tau)
    # remove same-goal allocations: don't want robots bump to each other
    match = []
    for g in G:
        if np.all(g[:3] == g[3:]):
            match.append(g)
    for item in match:
        G = [x for x in G if not (x==item).all()]
    return G


# sets the location of the agents
def updateState(team_loc, astar):
    s = []
    for idx in range(len(team_loc)):
        action_idx = [astar[idx*3], astar[idx*3 + 1], astar[idx*3 + 2]]
        s_idx = team_loc[idx] + action_idx
        s += list(s_idx)
    return s
