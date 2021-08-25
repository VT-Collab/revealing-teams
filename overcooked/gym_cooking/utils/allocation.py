import numpy as np


def combination(goal1, goal2, goal3, agent1_loc, agent2_loc, agent3_loc, folder):
    # recorded allocations
    # each agent has a goal
    if folder == 'legibility':
        tau1 = np.asarray(goal1+goal2+goal3)
        tau2 = np.asarray(goal1+goal3+goal2)
        tau3 = np.asarray(goal1+goal3+goal2)
        tau4 = np.asarray(goal2+goal1+goal3)
        tau5 = np.asarray(goal2+goal3+goal1)
        tau6 = np.asarray(goal2+goal3+goal1)
        tau7 = np.asarray(goal3+goal2+goal1)
        tau8 = np.asarray(goal3+goal1+goal2)
        # only two agents have goals, one agent does two tasks
        tau9 = np.asarray(goal1+goal3+agent3_loc)
        tau10 = np.asarray(goal3+agent2_loc+goal1)
        G = [tau1, tau2, tau3, tau4, tau5,
        tau6, tau7, tau8, tau9, tau10]
    elif folder == 'fairness':
        tau1 = np.asarray(goal1 + goal2 + goal3)
        tau2 = np.asarray(goal1 + goal2 + goal3)
        tau3 = np.asarray(goal3 + goal2 + goal1)
        # only two agents have goals, one agent does two tasks
        tau4 = np.asarray(goal1 + goal3 + agent3_loc)
        tau5 = np.asarray(goal3 + agent2_loc + goal1)
        tau6 = np.asarray(agent1_loc + goal1 + goal2)
        # one agent has three goals
        tau7 = np.asarray(agent1_loc + agent2_loc + goal1)
        tau8 = np.asarray(goal1 + agent2_loc + agent3_loc)
        tau9 = np.asarray(agent1_loc + goal1 + agent3_loc)
        G = [tau1, tau2, tau3, tau4, tau5,
        tau6, tau7, tau8, tau9]


    return G
