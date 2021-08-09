import numpy as np

def actionSpace():
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
    return A
