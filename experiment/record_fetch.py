#!/usr/bin/env python

import rospy
import actionlib
import pickle
import sys
import time
import pygame
import numpy as np

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


HOME_POSITION = [0.45, 0.0521550178527832, -0.8318014144897461, 3.1197338104248047,
-2.251500368118286, 3.128553867340088, -1.4852769374847412, 3.123185396194458]
HOMING_TIME = 4.0

STEP_SIZE_L = 0.2
STEP_SIZE_A = np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1


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


class JoystickControl(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        z = [z1, z2, z3]
        for idx in range(len(z)):
            if abs(z[idx]) < DEADBAND:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        A_pressed = self.gamepad.get_button(0)
        B_pressued = self.gamepad.get_button(1)
        return tuple(z), (A_pressed, B_pressued), stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, STEP_SIZE_A * -z[1], STEP_SIZE_A * -z[0], STEP_SIZE_A * -z[2])
        else:
            self.action = (STEP_SIZE_L * -z[1], STEP_SIZE_L * -z[0], STEP_SIZE_L * -z[2], 0, 0, 0)


def main():
    task = sys.argv[1]
    block = sys.argv[2]

    rospy.init_node("endeffector_teleop")
    mover = TrajectoryClient()
    fetch_robot = FetchRobot()
    listener = JointStateListener()
    joystick = JoystickControl()
    mover.open_gripper()
    mover.send_joint(HOME_POSITION, HOMING_TIME)

    savename_number = 'data/'+task+'/fetch_' + block + ".pkl"
    start_time = time.time()
    last_time = 0.0
    sample_time = 0.1
    record = False
    count = 0
    positions = []


    while not rospy.is_shutdown():
        axes, buttons, stop = joystick.getInput()
        if stop:
            pickle.dump(positions, open(savename_number, "wb"))
            print("[*] Data Saved!")
            return True
        if buttons[0] and not record:
            record = True
            last_time = time.time()
            print("---- ready for inputs!")
            continue
        if buttons[1] and record:
            pose = fetch_robot.dirkin(listener.state)
            robot_xyz = np.asarray(pose["gripper_link"].pos)
            curr_time = time.time()
            if curr_time - last_time > sample_time:
                print(robot_xyz)
                positions.append(robot_xyz)
                count += 1
                print("---- position Recorded: ", count,'\n')
                last_time = curr_time
                record = False
            continue

        joystick.getAction(axes)
        action = joystick.action
        mover.send(action, STEP_TIME)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
