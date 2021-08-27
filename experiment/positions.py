import numpy as np
import kinpy as kp
import rospy
import pygame
from utils.fetch_kn import *
from play_robots import readState, joint2pose, connect2robot
import sys
import pickle

from sensor_msgs.msg import (
      JointState
)

class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.deadband = 0.1

    def input(self):
        pygame.event.get()
        stop = self.gamepad.get_button(7)
        save = self.gamepad.get_button(0)
        return stop, save

def robotPos(robot, name):
    if name == 'fetch':
        # Fetch's current end-effector position
        listener = JointStateListener()
        pose = robot.dirkin(listener.state)
        fetch_xyz = np.asarray(pose["gripper_link"].pos)
        return fetch_xyz
    elif name == 'panda':
        # Panda's current end-effector position
        state_panda = readState(conn)
        panda_xyz = joint2pose(state_panda["q"])
        return panda_xyz


def main():
    robot = sys.argv[1]
    block = sys.argv[2]
    joystick = Joystick()

    if robot == 'fetch':
        # Fetch
        print('[*] Connecting to Fetch...')
        rospy.init_node("endeffector_teleop")
        fetch_robot = FetchRobot()
    elif robot == 'panda':
        # Panda
        print('[*] Connecting to Panda...')
        PORT_robot = 8080
        conn = connect2robot(PORT_robot)
    else:
        print('Type the robot name!')

    pos = []

    while True:

        if robot == 'fetch':
            robot_xyz = robotPos(fetch_robot, robot)
        elif robot == 'panda':
            robot_xyz = robotPos(conn, robot)

        savename_number = 'data/' + robot + '_' + block + ".pkl"
        reset_breaker()
        print("ready for a recording!")

        while True:

            stop, save = joystick.input()
            if stop:
                pickle.dump(pos, open(savename_number, "wb"))
                return True
            if save:
                pos.append(robot_xyz)
                print("[*] Position Recorded!")
                break



if __name__ == "__main__":
    main()
