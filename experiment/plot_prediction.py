import time
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt


users_n = 3
measure = ['choice', 'time']
tasks = ['task1', 'task2']

user_ball_legible_task1 = np.array([1,2,2,1])
user_ball_legible_task2 = np.array([3,2,1,2])

user_ans_task1 = np.empty([users_n, 4])
user_ans_task2 = np.empty([users_n, 4])

for user_idx in range(users_n):
    for task in tasks:
        answers = []
        print('User:', user_idx, '/ Task:', task)
        filename = '{}/{}_{}_{}_{}.pkl'.format('data/user_study', 'user'+str(user_idx), measure[0], 'legible', task)
        data = pickle.load(open(filename, "rb"))
        print(data)
        print()
        for key in data.keys():
            answers.append(data[key])
        if task == 'task1':
            user_ans_task1[user_idx] = answers
        elif task == 'task2':
            user_ans_task2[user_idx] = answers


boolarr_task1 = np.equal(user_ans_task1, user_ball_legible_task1)
boolarr_task2 = np.equal(user_ans_task2, user_ball_legible_task2)

correct_pred_legible_task1 = boolarr_task1[:,0].sum() + boolarr_task1[:,3].sum()
correct_pred_illegible_task1 = boolarr_task1[:,1].sum() + boolarr_task1[:,2].sum()

correct_pred_legible_task2 = boolarr_task2[:,0].sum() + boolarr_task2[:,2].sum()
correct_pred_illegible_task2 = boolarr_task2[:,1].sum() + boolarr_task2[:,3].sum()


plt.figure()
X = ['Task 1', 'Task 2']
X_axis = np.arange(len(X))
legible = np.array([correct_pred_legible_task1, correct_pred_legible_task2])
illegible = np.array([correct_pred_illegible_task1, correct_pred_illegible_task2])

plt.bar(X_axis - 0.2, legible, 0.4, label = 'legible')
plt.bar(X_axis + 0.2, illegible, 0.4, label = 'illegible')
plt.xticks(X_axis, X)
plt.ylabel("Number of Correct Predictions")
plt.legend()
plt.show()
