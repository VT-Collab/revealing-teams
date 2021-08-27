import numpy as np
import kinpy as kp
import rospy
import torch

from power_msgs.srv import BreakerCommand
from sensor_msgs.msg import (
      JointState
)


class JointStateListener(object):

  def __init__(self):
    rospy.Subscriber("/joint_states", JointState, self.recorder)
    self.state = {}
    self.position = [0]*8

  def recorder(self, msg):
    currtime = msg.header.stamp
    position = msg.position
    name  = msg.name
    if len(position) > 10:
        self.position[0] = position[2]
        self.position[1:] = position[6:]
        # print(self.position)
        state = {}
        for idx in range(len(name)):
            state[name[idx]] = position[idx]
        self.state = state


class FetchRobot:

    def __init__(self):
        self.chain = kp.build_chain_from_urdf((open("fetch.urdf")).read())

    def dirkin(self, q):

        pose = self.chain.forward_kinematics(q)
        return pose


def reset_breaker():
    rospy.wait_for_service('/arm_breaker')
    reset_arm_breaker = rospy.ServiceProxy('/arm_breaker', BreakerCommand)
    reset_arm_breaker(False)
    reset_arm_breaker(True)
    rospy.sleep(0.2)
