import time
import numpy as np
import pickle
import sys
import matplotlib.pyplot as plt


users_n = 11
measure = ['choice', 'time']
tasks = ['task1', 'task2']

user_time_task1 = np.empty([users_n, 4])
user_time_task2 = np.empty([users_n, 4])


for user_idx in range(1,users_n+1):
    for task in tasks:
        measured_times = []
        filename = '{}/{}_{}_{}_{}.pkl'.format('data/user_study', 'user'+str(user_idx), measure[1], 'legible', task)
        data = pickle.load(open(filename, "rb"))
        for key in data.keys():
            measured_times.append(data[key])
        if task == 'task1':
            user_time_task1[user_idx-1] = measured_times
        elif task == 'task2':
            user_time_task2[user_idx-1] = measured_times


########################################################
#averaged time across users for each pair of allocations
########################################################
pair_mean_user_time_task1 = np.mean(user_time_task1, axis=0)
pair_mean_user_time_task2 = np.mean(user_time_task2, axis=0)
########################################################
#averaged time across users for each pair of allocations
########################################################

########################################################
# averaged time across all legible/illegible allocations
########################################################
mean_user_time_task1 = np.mean(user_time_task1, axis=0)
mean_user_time_task2 = np.mean(user_time_task2, axis=0)

mean_time_legible_task1 = np.mean(np.array([mean_user_time_task1[0], mean_user_time_task1[3]]))
mean_time_illegible_task1 = np.mean(np.array([mean_user_time_task1[1], mean_user_time_task1[2]]))

mean_time_legible_task2 = np.mean(np.array([mean_user_time_task2[0], mean_user_time_task2[2]]))
mean_time_illegible_task2 = np.mean(np.array([mean_user_time_task2[1], mean_user_time_task2[3]]))
########################################################
# averaged time across all legible/illegible allocations
########################################################


# # plot averaged time across all legible/illegible allocations
# plt.figure()
# X = ['Task 1', 'Task 2']
# X_axis = np.arange(len(X))
# avg_legible = np.array([mean_time_legible_task1, mean_time_legible_task2])
# avg_illegible = np.array([mean_time_illegible_task1, mean_time_illegible_task2])
# plt.bar(X_axis - 0.2, avg_legible, 0.4, label = 'legible')
# plt.bar(X_axis + 0.2, avg_illegible, 0.4, label = 'illegible')
# plt.xticks(X_axis, X)
# plt.ylabel("Prediction Time [s]")
# plt.ylim([0,5])
# plt.legend()
# plt.savefig('user_time_avg.svg')
# # plt.show()


# plot averaged time across users for each pair of allocations
plt.figure()
X = ['Pair 1', 'Pair 2', 'Pair 3', 'Pair 4']
X_axis = np.arange(len(X))
legible = np.array([pair_mean_user_time_task1[0], pair_mean_user_time_task1[3],
                    pair_mean_user_time_task2[0], pair_mean_user_time_task2[2]])
illegible = np.array([pair_mean_user_time_task1[1], pair_mean_user_time_task1[2],
                    pair_mean_user_time_task2[1], pair_mean_user_time_task2[3]])

plt.bar(X_axis - 0.2, legible, 0.4, label = 'legible')
plt.bar(X_axis + 0.2, illegible, 0.4, label = 'illegible')
plt.xticks(X_axis, X)
plt.ylabel("Prediction Time [s]")
plt.ylim([0,5])
plt.legend()
plt.savefig('user_time_avg_pair.svg')
plt.show()
