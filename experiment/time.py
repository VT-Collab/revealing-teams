import time
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt


users_n = 9
measure = ['choice', 'time']
tasks = ['task1', 'task2']

user_time_task1 = np.empty([users_n, 4])
user_time_task2 = np.empty([users_n, 4])


for user_idx in range(1,users_n):
    for task in tasks:
        measured_times = []
        filename = '{}/{}_{}_{}_{}.pkl'.format('data/user_study', 'user'+str(user_idx), measure[1], 'legible', task)
        data = pickle.load(open(filename, "rb"))
        for key in data.keys():
            measured_times.append(data[key])
        if task == 'task1':
            user_time_task1[user_idx] = measured_times
        elif task == 'task2':
            user_time_task2[user_idx] = measured_times


########################################################
#averaged time across users for each pair of allocations
########################################################
avg_user_time_task1 = np.mean(user_time_task1, axis=0)
avg_user_time_task2 = np.mean(user_time_task2, axis=0)
########################################################
#averaged time across users for each pair of allocations
########################################################


########################################################
# averaged time across all legible/illegible allocations
########################################################
time_legible_task1 = np.array([np.mean(user_time_task1[:,0], axis=0), np.mean(user_time_task1[:,3], axis=0)])
time_illegible_task1 = np.array([np.mean(user_time_task1[:,1], axis=0), np.mean(user_time_task1[:,2], axis=0)])

avg_time_legible_task1 = np.mean(time_legible_task1)
avg_time_illegible_task1 = np.mean(time_illegible_task1)

time_legible_task2 = np.array([np.mean(user_time_task2[:,0], axis=0), np.mean(user_time_task2[:,2], axis=0)])
time_illegible_task2 = np.array([np.mean(user_time_task2[:,1], axis=0), np.mean(user_time_task2[:,3], axis=0)])

avg_time_illegible_task2 = np.mean(time_illegible_task2)
avg_time_illegible_task2 = np.mean(time_illegible_task2)
########################################################
# averaged time across all legible/illegible allocations
########################################################


# plot averaged time across all legible/illegible allocations
plt.figure()
X = ['Task 1', 'Task 2']
X_axis = np.arange(len(X))
legible = np.array([avg_time_legible_task1, avg_time_illegible_task2])
illegible = np.array([avg_time_illegible_task1, avg_time_illegible_task2])

plt.bar(X_axis - 0.2, legible, 0.4, label = 'legible')
plt.bar(X_axis + 0.2, illegible, 0.4, label = 'illegible')
plt.xticks(X_axis, X)
plt.ylabel("Prediction Time [s]")
plt.legend()
plt.savefig('user_time_avg.svg')
# plt.show()


# plot averaged time across users for each pair of allocations
plt.figure()
X = ['Pair 1', 'Pair 2', 'Pair 3', 'Pair 4']
X_axis = np.arange(len(X))
legible = np.array([avg_user_time_task1[0], avg_user_time_task1[3],
avg_user_time_task2[0], avg_user_time_task2[2]])

illegible = np.array([avg_user_time_task1[1], avg_user_time_task1[2],
avg_user_time_task2[1], avg_user_time_task2[3]])

plt.bar(X_axis - 0.2, legible, 0.4, label = 'legible')
plt.bar(X_axis + 0.2, illegible, 0.4, label = 'illegible')
plt.xticks(X_axis, X)
plt.ylabel("Prediction Time [s]")
plt.legend()
plt.savefig('user_time_avg_pair.svg')
# plt.show()
