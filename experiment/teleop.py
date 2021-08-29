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


fetch_home = [0.35, 0.6937427520751953, -1.2402234077453613,
        3.0553064346313477, -0.9848155975341797, -3.136991024017334,
        1.8170002698898315, -0.11811652034521103]
fetch_home_t = 4.0
fetch_step_t = 0.2

panda_home = [1.720000e-04, -7.853450e-01,  1.950000e-04, -2.356711e+00,
        3.260000e-04,  1.571298e+00,  7.857880e-01]

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
    robot_error = (goal - cur_pos)*action_scale
    robot_action = [robot_error[0], robot_error[1], robot_error[2], 0, 0, 0]
    return robot_action, np.linalg.norm(robot_error)

def fetchThread(idx, goal, listener, fetch_robot, mover):
    while True:
        # current end-effector position
        pose = fetch_robot.dirkin(listener.state)
        fetch_xyz = np.asarray(pose["gripper_link"].pos)
        action_scale_fetch = 0.5
        # compute robot actions
        action_fetch, error_fetch = robotAtion(goal, fetch_xyz, action_scale_fetch)
        if idx != 1:
            dist = 0.01
        else:
            dist = 0.1
        if 0.01 <= error_fetch < 0.02:
            action_scale_fetch = 0.3
        elif error_fetch < dist:
            break
        # mover.close_gripper()
        # send commands to the robot
        mover.send(action_fetch, fetch_step_t)
        time.sleep(0.01)


def pandaThread(idx, goal, conn, action_scale_panda=0.5):
    while True:
        # current end-effector position
        state_panda = readState(conn)
        panda_xyz = joint2pose(state_panda["q"])
        # compute robot actions
        action_panda, error_panda = robotAtion(goal, panda_xyz, action_scale_panda)
        dist = 0.01
        if error_panda < 0.02:
            action_scale_panda = 0.5*(error_panda)
        if error_panda < dist:
            break
        # send action commands to the robot
        send2robot(conn, xdot2qdot(action_panda, state_panda))
        time.sleep(0.01)



def main(trajectory1, trajectory2):

    print('[*] Connecting to Fetch...')
    rospy.init_node("endeffector_teleop")
    fetch_robot = FetchRobot()
    mover = TrajectoryClient()
    listener = JointStateListener()
    mover.send_joint(fetch_home, fetch_home_t)
    # mover.open_gripper()

    print('[*] Connecting to Panda...')
    PORT_robot = 8080
    conn = connect2robot(PORT_robot)
    panda_home_xyz = joint2pose(panda_home)
    pandaThread(0, panda_home_xyz, conn)


    for idx in range(len(trajectory2)):
        print('Goal: ',idx)

        # t1 = Thread(target = fetchThread, args=(idx, trajectory1[idx],
        #                         listener, fetch_robot, mover,))
        t2 = Thread(target = pandaThread, args= (idx, trajectory2[idx], conn,))

        # t1.start()
        # time.sleep(0.01)
        t2.start()

        # t1.join()
        t2.join()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
