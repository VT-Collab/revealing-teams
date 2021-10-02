import time
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt


users_n = 11
measure = ['choice', 'time']
tasks = ['task1', 'task2']

user_ball_legible_task1 = np.array([1,2,2,1])
user_ball_legible_task2 = np.array([1,2,3,2])

user_ans_task1 = np.empty([users_n, 4])
user_ans_task2 = np.empty([users_n, 4])

for user_idx in range(1,users_n+1):
    for task in tasks:
        answers = []
        filename = '{}/{}_{}_{}_{}.pkl'.format('data/user_study', 'user'+str(user_idx), measure[0], 'legible', task)
        data = pickle.load(open(filename, "rb"))
        for key in data.keys():
            answers.append(data[key])
        if task == 'task1':
            user_ans_task1[user_idx-1] = answers
        elif task == 'task2':
            user_ans_task2[user_idx-1] = answers
boolarr_task1 = np.equal(user_ans_task1, user_ball_legible_task1)
boolarr_task2 = np.equal(user_ans_task2, user_ball_legible_task2)


########################################################
#averaged time across users for each pair of allocations
########################################################
correct_pred_task1 = np.count_nonzero(boolarr_task1, axis=0)*100/11
correct_pred_task2 = np.count_nonzero(boolarr_task2, axis=0)*100/11
########################################################
#averaged time across users for each pair of allocations
########################################################


########################################################
# averaged time across all legible/illegible allocations
########################################################
correct_pred_legible_task1 = (correct_pred_task1[0] + correct_pred_task1[3])/2
correct_pred_illegible_task1 = (correct_pred_task1[1] + correct_pred_task1[2])/2

correct_pred_legible_task2 = (correct_pred_task2[0] + correct_pred_task2[2])/2
correct_pred_illegible_task2 = (correct_pred_task2[1] + correct_pred_task2[3])/2
########################################################
# averaged time across all legible/illegible allocations
########################################################


# # plot users' correct predictions across all legible/illegible allocations
# plt.figure()
# X = ['Task 1', 'Task 2']
# X_axis = np.arange(len(X))
# legible = np.array([correct_pred_legible_task1, correct_pred_legible_task2])
# illegible = np.array([correct_pred_illegible_task1, correct_pred_illegible_task2])
#
# plt.bar(X_axis - 0.2, legible, 0.4, label = 'legible')
# plt.bar(X_axis + 0.2, illegible, 0.4, label = 'illegible')
# plt.xticks(X_axis, X)
# plt.ylabel("Number of Correct Predictions")
# plt.ylim([0,100])
# plt.legend()
# plt.savefig('user_prediction.svg')
# plt.show()


# plot users' correct predictions across users for each pair of allocations
plt.figure()
X = ['Pair 1', 'Pair 2', 'Pair 3', 'Pair 4']
X_axis = np.arange(len(X))
legible = np.array([correct_pred_task1[0], correct_pred_task1[3],
                    correct_pred_task2[0], correct_pred_task2[2]])
illegible = np.array([correct_pred_task1[1], correct_pred_task1[2],
                    correct_pred_task2[1], correct_pred_task2[3]])

plt.bar(X_axis - 0.2, legible, 0.4, label = 'legible')
plt.bar(X_axis + 0.2, illegible, 0.4, label = 'illegible')
plt.xticks(X_axis, X)
plt.ylabel("Number of Correct Predictions")
plt.ylim([0,100])
plt.legend()
plt.savefig('user_prediction_pair.svg')
plt.show()
