import socket
import time
import numpy as np
import pickle
import pygame
import sys
import rospy
import actionlib
import kinpy as kp
import math



USER_TIME = [1]
pickle.dump(USER_TIME, open('{}_{}_{}_{}.pkl'.format('time', 'user', 'test', 'task'), "wb"))


filename = "time_user_test_task.pkl"
data = pickle.load(open(filename, "rb"))
print(data)
