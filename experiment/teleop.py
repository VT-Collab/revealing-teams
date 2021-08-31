import socket
import time
import numpy as np
import pickle
import pygame
import sys
import rospy
import actionlib
import kinpy as kp
from threading import Thread


from utils.fetch_kn import *
from utils.panda_home import main as panda_return

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


fetch_home = [0.38, 0.15263128280639648, -1.2490439414978027,
            3.0549232959747314, -0.49547576904296875, 3.1063108444213867,
            2.279879093170166, 1.6206507682800293]
fetch_home_t = 5.0


'''-----------Fetch-----------'''
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
        rospy.sleep(time)

    def send_joint(self, position, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = position
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.sleep(time)

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
'''-----------Fetch-----------'''



'''-----------Panda-----------'''
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
'''-----------Panda-----------'''


def robotAtion(goal, cur_pos, action_scale):
    robot_error = (goal - cur_pos)/action_scale
    robot_action = [robot_error[0], robot_error[1], robot_error[2], 0, 0, 0]
    return robot_action


def fetchThread(waypoint, goal, listener, fetch_robot, mover, action_scale_fetch = 1.5):

    while True:
        fetch_step_t = 0.1
        # current end-effector position
        pose = fetch_robot.dirkin(listener.state)
        fetch_xyz = np.asarray(pose["gripper_link"].pos)
        # compute robot actions
        action_fetch = robotAtion(goal, fetch_xyz, action_scale_fetch)
        dist = np.linalg.norm(goal-fetch_xyz)
        if dist < 0.002:
            if waypoint == 3:
                mover.close_gripper()
                time.sleep(2)
            elif waypoint == 6:
                mover.open_gripper()
            break
        # send commands to the robot
        mover.send(action_fetch, fetch_step_t)


def pandaThread(waypoint, goal, conn, conn_gripper, action_scale_panda=1.5):

    while True:
        # current end-effector position
        state_panda = readState(conn)
        panda_xyz = joint2pose(state_panda["q"])
        # compute robot actions
        action_panda = robotAtion(goal, panda_xyz, action_scale_panda)
        dist = np.linalg.norm(goal-panda_xyz)
        if dist < 0.003:
            if waypoint == 3:
                send2gripper(conn_gripper, 'c')
                time.sleep(2)
            elif waypoint == 6:
                send2gripper(conn_gripper, 'o')
            break
        # send action commands to the robot
        send2robot(conn, xdot2qdot(action_panda, state_panda))


def main(trajectory1, trajectory2):

    print('[*] Connecting to Fetch...')
    rospy.init_node("endeffector_teleop")
    fetch_robot = FetchRobot()
    mover = TrajectoryClient()
    listener = JointStateListener()
    # mover.open_gripper()

    print('[*] Connecting to Panda...')
    PORT_robot = 8080
    conn = connect2robot(PORT_robot)

    print('[*] Connecting to Panda gripper...')
    PORT_gripper = 8081
    conn_gripper = connect2gripper(PORT_gripper)
    # send2gripper(conn_gripper, 'o')


    # send robots to home
    mover.send_joint(fetch_home, fetch_home_t)
    panda_return(conn)

    for idx in range(len(trajectory2)):
        waypoint = idx+1
        print('waypoint: ',waypoint)

        t1 = Thread(target = fetchThread, args=(waypoint, trajectory1[idx],
                                listener, fetch_robot, mover,))
        t2 = Thread(target = pandaThread, args= (waypoint, trajectory2[idx], conn, conn_gripper))

        t1.start()
        # time.sleep(0.01)
        t2.start()

        t1.join()
        t2.join()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
