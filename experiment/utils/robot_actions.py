import numpy as np

def actionSpace():
    # discretize the space of actions for the team
    n_actions = 25
    r = 0.02
    single_action_1 = []
    single_action_2 = []
    angles = np.linspace(-np.pi/2, np.pi/2, n_actions)
    for angle_h in angles:
        action_1 = [r * np.cos(angle_h), r * np.sin(angle_h)]
        action_2 = [-r * np.cos(angle_h), r * np.sin(angle_h)]
        single_action_1.append(action_1)
        single_action_2.append(action_2)
    single_action_1.append([0,0])
    single_action_2.append([0,0])
    A = []
    for idx in range(n_actions+1):
        for jdx in range(n_actions+1):
            A.append(list(single_action_1[idx]) +
            list(single_action_2[jdx]))
    A = np.asarray(A)
    return A
