import numpy as np
import kinpy as kp
import rospy
import pygame
import sys
import pickle
import time

from utils.fetch_kn import *
# from utils.panda_demo import *
from teleop import TrajectoryClient

from sensor_msgs.msg import (
      JointState
)


HOME_POSITION = [0.3, 0.6937427520751953, -1.2402234077453613, 3.0553064346313477,
    -0.9848155975341797, -3.136991024017334, 1.8170002698898315, -0.11811652034521103]
HOMING_TIME = 4.0

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

    robot = sys.argv[1]
    block = sys.argv[2]

    if robot == 'fetch':
        home_xyz = np.array([1.128, 0, 0.786])
        print('[*] Connecting to Fetch...')
        rospy.init_node("endeffector_teleop")
        mover = TrajectoryClient()
        fetch_robot = FetchRobot()
        listener = JointStateListener()
        mover.open_gripper()
        mover.send_joint(HOME_POSITION, HOMING_TIME)
        print('[*] Home position...')
        positions = [home_xyz]
        reset_breaker()

    elif robot == 'panda':
        positions = []
        print('connected to panda')


    savename_number = 'data/' + robot + '_' + block + ".pkl"
    sample_time = 0.2
    last_time = None
    record = False

    while True:

        if robot == 'fetch':
            pose = fetch_robot.dirkin(listener.state)
            robot_xyz = np.asarray(pose["gripper_link"].pos)
        elif robot == 'panda':
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
            print('[*] Starting the demonstration...')
        if B_pressed and record:
            # print('[*] Pausing the demonstration...')
            curr_time = time.time()
            if curr_time - last_time >= sample_time:
                print(robot_xyz)
                positions.append(robot_xyz)
                print("[*] Position Recorded!",'\n')
                last_time = curr_time
                record = False



if __name__ == "__main__":
    main()
