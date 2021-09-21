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

from utils.fetch_kn import *
from utils.panda_home import main as send_panda_home

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
    GripperCommand
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import (
    JointState
)
from geometry_msgs.msg import(
    TwistStamped,
    Twist
)


fetch_home = [0.42, 0.0521550178527832,
-0.8318014144897461, 3.1197338104248047, -2.251500368118286, 3.128553867340088, -1.4852769374847412, 3.123185396194458]
fetch_home_t = 5.0


'''######################## Fetch ########################'''
class TrajectoryClient(object):

    def __init__(self):
        self.gripper = actionlib.SimpleActionClient(
                '/gripper_controller/gripper_action',
                GripperCommandAction)
        self.gripper.wait_for_server()
        self.publisher = rospy.Publisher(
                '/arm_controller/cartesian_twist/command',
                TwistStamped, queue_size=1)
        self.publish_rate = rospy.Rate(1)
        self.client = actionlib.SimpleActionClient(
                '/arm_with_torso_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        self.joint_names = ["torso_lift_joint", "shoulder_pan_joint",
                "shoulder_lift_joint", "upperarm_roll_joint",
                "elbow_flex_joint", "forearm_roll_joint",
                "wrist_flex_joint", "wrist_roll_joint"]

    def send(self, twist, time):
        xdot = TwistStamped()
        xdot.twist.linear.x = twist[0]
        xdot.twist.linear.y = twist[1]
        xdot.twist.linear.z = twist[2]
        xdot.twist.angular.x = twist[3]
        xdot.twist.angular.y = twist[4]
        xdot.twist.angular.z = twist[5]
        xdot.header.stamp = rospy.Time.now()
        xdot.header.frame_id = "shoulder_pan_joint"
        self.publisher.publish(xdot)
        # rospy.sleep(time)

    def send_joint(self, position, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = position
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        # rospy.sleep(time)

    def open_gripper(self):
        command = GripperCommand
        command.max_effort = 60
        command.position = 0.1
        waypoint = GripperCommandGoal
        waypoint.command = command
        self.gripper.send_goal(waypoint)

    def close_gripper(self):
        command = GripperCommand
        command.max_effort = 60
        command.position = 0.0
        waypoint = GripperCommandGoal
        waypoint.command = command
        self.gripper.send_goal(waypoint)
'''######################## Fetch ########################'''



'''######################## Panda ########################'''
def connect2robot(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('172.16.0.3', PORT))
    s.listen(10)
    conn, addr = s.accept()
    return conn

def connect2gripper(PORT):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('172.16.0.3', PORT))
    s.listen(10)
    conn, addr = s.accept()
    return conn

def send2gripper(conn, arg):
    send_msg = arg
    conn.send(send_msg.encode())

def send2robot(conn, qdot, limit=1.0):
    qdot = np.asarray(qdot)
    scale = np.linalg.norm(qdot)
    if scale > limit:
        qdot = np.asarray([qdot[i] * limit/scale for i in range(7)])
    send_msg = np.array2string(qdot, precision=5, separator=',',suppress_small=True)[1:-1]
    send_msg = "s," + send_msg + ","
    conn.send(send_msg.encode())

def listen2robot(conn):
    state_length = 7 + 7 + 7 + 42
    message = str(conn.recv(2048))[2:-2]
    state_str = list(message.split(","))
    for idx in range(len(state_str)):
        if state_str[idx] == "s":
            state_str = state_str[idx+1:idx+1+state_length]
            break
    try:
        state_vector = [float(item) for item in state_str]
    except ValueError:
        return None
    if len(state_vector) is not state_length:
        return None
    state_vector = np.asarray(state_vector)
    state = {}
    state["q"] = state_vector[0:7]
    state["dq"] = state_vector[7:14]
    state["tau"] = state_vector[14:21]
    state["J"] = state_vector[21:].reshape((7,6)).T
    return state

def readState(conn):
    while True:
        state = listen2robot(conn)
        if state is not None:
            break
    return state

def xdot2qdot(xdot, state):
    J_pinv = np.linalg.pinv(state["J"])
    return np.matmul(J_pinv, np.asarray(xdot))

def joint2pose(q):
    def RotX(q):
        return np.array([[1, 0, 0, 0], [0, np.cos(q), -np.sin(q), 0], [0, np.sin(q), np.cos(q), 0], [0, 0, 0, 1]])
    def RotZ(q):
        return np.array([[np.cos(q), -np.sin(q), 0, 0], [np.sin(q), np.cos(q), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    def TransX(q, x, y, z):
        return np.array([[1, 0, 0, x], [0, np.cos(q), -np.sin(q), y], [0, np.sin(q), np.cos(q), z], [0, 0, 0, 1]])
    def TransZ(q, x, y, z):
        return np.array([[np.cos(q), -np.sin(q), 0, x], [np.sin(q), np.cos(q), 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
    H1 = TransZ(q[0], 0, 0, 0.333)
    H2 = np.dot(RotX(-np.pi/2), RotZ(q[1]))
    H3 = np.dot(TransX(np.pi/2, 0, -0.316, 0), RotZ(q[2]))
    H4 = np.dot(TransX(np.pi/2, 0.0825, 0, 0), RotZ(q[3]))
    H5 = np.dot(TransX(-np.pi/2, -0.0825, 0.384, 0), RotZ(q[4]))
    H6 = np.dot(RotX(np.pi/2), RotZ(q[5]))
    H7 = np.dot(TransX(np.pi/2, 0.088, 0, 0), RotZ(q[6]))
    H_panda_hand = TransZ(-np.pi/4, 0, 0, 0.2105)
    H = np.linalg.multi_dot([H1, H2, H3, H4, H5, H6, H7, H_panda_hand])
    return H[:,3][:3]
'''######################## Panda ########################'''


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
        return A_pressed, Start_pressed


def robotAction(goal, cur_pos, action_scale, traj_length, waypoint, robot_working):
    if robot_working:
        if np.linalg.norm(goal - cur_pos) < 0.01 and waypoint > traj_length-25:
            robot_error = (goal - cur_pos)
        else:
            robot_error = (goal - cur_pos)/np.linalg.norm(goal - cur_pos)*action_scale
    else:
        robot_error = [0,0,0]
    robot_action = [robot_error[0], robot_error[1], robot_error[2], 0, 0, 0]
    return robot_action


def main(ALLOCATION_PANDA, ALLOCATION_FETCH, test, task):
    USER_TIME = {}
    USER_CHOICE = {}
    interface = Joystick()

    print('[*] Connecting to Fetch...')
    rospy.init_node("endeffector_velocity_control")
    fetch_robot = FetchRobot()
    mover = TrajectoryClient()
    listener = JointStateListener()
    mover.open_gripper()

    print('[*] Connecting to Panda...')
    PORT_robot = 8080
    conn = connect2robot(PORT_robot)

    print('[*] Connecting to Panda gripper...','\n')
    PORT_gripper = 8081
    conn_gripper = connect2gripper(PORT_gripper)
    send2gripper(conn_gripper, 'o')

    # this loops iterates 2 pairs of subtask allocations
    for alloc_idx in range(len(ALLOCATION_PANDA)):
        if (alloc_idx+1) % 2 == 0:
            pair_num = alloc_idx

        allocation_panda = ALLOCATION_PANDA[alloc_idx]
        allocation_fetch = ALLOCATION_FETCH[alloc_idx]

        # send robots to home positions
        mover.send_joint(fetch_home, fetch_home_t)
        send_panda_home(conn)

        # trajectories of robots
        trajectory_panda = allocation_panda[-1]
        trajectory_fetch = allocation_fetch[-1]
        panda_traj_length = len(trajectory_panda)
        fetch_traj_length = len(trajectory_fetch)

        # robot flags: paly vs. no-play
        flag_panda = allocation_panda[0]
        flag_fetch = allocation_fetch[0]

        # Panda paramteres
        panda_action_scale = 0.05
        panda_waypoint = 0
        panda_threshold = 0.01
        panda_goal = trajectory_panda[panda_waypoint]

        # Fetch paramteres
        fetch_action_scale = 0.05
        fetch_step_t = 0.1
        fetch_waypoint = 0
        fetch_threshold = 0.008
        fetch_goal = trajectory_fetch[fetch_waypoint]

        pause = False
        last_time = None
        sample_time = 0.5
        panda_working = True
        fetch_working = True

        print('[*] Starting the allocation ' + str(alloc_idx+1))
        alloc_start_time = time.time()
        while panda_working or fetch_working:

            # joystick control for the user to pause/continue the robots
            A_pressed, Start_pressed = interface.input()
            if A_pressed and not pause:
                pause = True
                last_time = time.time()
                user_paused_time = time.time() - alloc_start_time
                print('User paused time: ',user_paused_time)

                print('---Task paused!')
            if Start_pressed and pause:
                curr_time = time.time()
                if curr_time - last_time >= sample_time:
                    last_time = curr_time
                    pause = False
                    print("---Task continues!")

            # store the user time
            # image_name = '{}_{}_{}_{}.png'.format('alloc', str(idx+1), 'frame', str(step+1))
            # pygame.image.save(game.screen, '{}/{}'.format('screenshots/'+folder, image_name))

            time_name = '{}_{}_{}_{}'.format(test, test, pair_num, alloc_idx)
            try:
                USER_TIME[time_name] = user_paused_time
            except:
                user_paused_time = math.inf
                USER_TIME[time_name] = user_paused_time

            # read robot states
            panda_state = readState(conn)
            panda_xyz = joint2pose(panda_state["q"])
            fetch_state = fetch_robot.dirkin(listener.state)
            fetch_xyz = np.asarray(fetch_state["gripper_link"].pos)

            '''######################## Panda ########################'''
            if flag_panda:
                panda_working = False
            else:
                if np.linalg.norm(panda_goal - panda_xyz) < panda_threshold:
                    panda_waypoint += 1
                    if panda_waypoint == panda_traj_length-2 and panda_working:
                        send2gripper(conn_gripper, 'c')
                        time.sleep(1)
                    elif panda_waypoint == panda_traj_length-1:
                        panda_action_scale = 0.3
                    elif panda_waypoint == panda_traj_length:
                        panda_action_scale = 0.0
                        panda_waypoint -= 1
                        panda_working = False
                    else:
                        pass
                    panda_goal = trajectory_panda[panda_waypoint]
            '''######################## Panda ########################'''


            '''######################## Fetch ########################'''
            if flag_fetch:
                fetch_working = False
            else:
                if np.linalg.norm(fetch_goal - fetch_xyz) < fetch_threshold:
                    fetch_waypoint += 1
                    if fetch_waypoint == fetch_traj_length-3 and fetch_working:
                        fetch_action_scale = 0.02
                        fetch_threshold = 0.004
                    if fetch_waypoint == fetch_traj_length-2 and fetch_working:
                        fetch_action_scale = 0.06
                        mover.close_gripper()
                        time.sleep(0.6)
                    if fetch_waypoint == fetch_traj_length-1:
                        fetch_action_scale = 0.3
                        fetch_threshold = 0.1
                    if fetch_waypoint == fetch_traj_length:
                        fetch_action_scale = 0.0
                        fetch_waypoint -= 1
                        fetch_working = False
                    fetch_goal = trajectory_fetch[fetch_waypoint]
            '''######################## Fetch ########################'''

            # compute robot actions
            panda_action = robotAction(panda_goal, panda_xyz, panda_action_scale,
                                    panda_traj_length, panda_waypoint, panda_working)
            fetch_action = robotAction(fetch_goal, fetch_xyz, fetch_action_scale,
                                    fetch_traj_length, fetch_waypoint, fetch_working)

            # check if the pause requested by joystick
            if pause:
                panda_action = [0]*6
                fetch_action = [0]*6

            # send ee velocity commands to robots
            send2robot(conn, xdot2qdot(panda_action, panda_state))
            mover.send(fetch_action, fetch_step_t)

            # check if robots are done with the task
            if not panda_working and not fetch_working:
                send2gripper(conn_gripper, 'o')
                mover.open_gripper()

        alloc_idx += 1
        # record user's choice after viewing each pair of allocations
        valid_inputs = ['1', '2', '3']
        user_choice  = None
        if test == 'legible':
            while user_choice not in valid_inputs:
                user_choice = input("---Which tennis ball? ")
            choice_name = '{}_{}_{}_{}'.format(test, test, pair_num, alloc_idx)
            USER_CHOICE[choice_name] = user_choice
        if alloc_idx == 4:
            print('[*] Task is finished!')
            print('Please answer the survey question...')
        elif alloc_idx%2 == 0:
            print('Please answer the survey question...')
            show_next = input("---Next?")
        else:
            show_next = input("---Next?")
        print()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
