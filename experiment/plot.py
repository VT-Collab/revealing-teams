import time
import numpy as np
import pickle
import sys
import math



users_n = 3
measure = ['choice', 'time']
tasks = ['task1', 'task2']

user_ball_legible_task1 = [1,2,1,1]
user_ball_legible_task2 = [1,2,1,1]

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

print(user_ans_task1)
print()
print(user_ans_task2)
