import numpy as np
import kinpy as kp
import rospy
import pygame
import sys
import pickle
import time

from utils.panda_demo import *

from sensor_msgs.msg import (
      JointState
)


class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()

    def input(self):
        pygame.event.get()
        A_pressed = self.gamepad.get_button(0)
        B_pressed = self.gamepad.get_button(1)
        Start_pressed = self.gamepad.get_button(7)
        return A_pressed, B_pressed, Start_pressed

def main():
    interface = Joystick()
    block = sys.argv[1]

    print('[*] Connected to Panda')
    savename_number = 'data/panda_' + block + ".pkl"
    last_time = 0.0
    sample_time = 0.2
    record = False
    positions = []

    while True:

        state = readState(conn)
        robot_xyz = joint2pose(state["q"].tolist())
        A_pressed, B_pressed, Start_pressed = interface.input()
        if Start_pressed:
            pickle.dump(positions, open(savename_number, "wb"))
            print("[*] Data Saved!")
            return True
        if A_pressed and not record:
            record = True
            last_time = time.time()
            print("---- ready for inputs!")
        if B_pressed and record:
            curr_time = time.time()
            if curr_time - last_time >= sample_time:
                print(robot_xyz)
                positions.append(robot_xyz)
                print("---- position Recorded!",'\n')
                last_time = curr_time
                record = False

if __name__ == "__main__":
    main()
