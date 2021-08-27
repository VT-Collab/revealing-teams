import numpy as np
import kinpy as kp
import rospy
import pygame
from utils.fetch_kn import *
import sys
import pickle

from play_robots import readState, joint2pose, connect2robot

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
        start_pressed = self.gamepad.get_button(7)
        A_pressued = self.gamepad.get_button(0)
        return start_pressed, A_pressued

def robotPos(robot, name):
    if name == 'fetch':
        # Fetch's current end-effector position
        listener = JointStateListener()
        pose = robot.dirkin(listener.state)
        fetch_xyz = np.asarray(pose["gripper_link"].pos)
        return fetch_xyz
    elif name == 'panda':
        # Panda's current end-effector position
        state_panda = readState(robot)
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
    savename_number = 'data/' + robot + '_' + block + ".pkl"

    while True:
        if robot == 'fetch':
            robot_xyz = robotPos(fetch_robot, robot)
            reset_breaker()
            print("ready for a recording!")
        elif robot == 'panda':
            robot_xyz = robotPos(conn, robot)
            print(robot_xyz)

        while True:
            start_pressed, A_pressued = joystick.input()
            if start_pressed:
                pickle.dump(pos, open(savename_number, "wb"))
                return True
            if A_pressued:
                pos.append(robot_xyz)
                print("[*] Position Recorded!")
                break



if __name__ == "__main__":
    main()
