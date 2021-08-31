import time
import numpy as np
import pygame
import sys
import rospy
import kinpy as kp
from threading import Thread
import threads

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

    last_time = 0.0
    sample_time = 0.2
    record = False

    while True:
        A_pressed, B_pressed, Start_pressed = interface.input()
        threads.main()
        if Start_pressed:
            print("Start_pressed")
            time.sleep(10)
            return True
        if A_pressed and not record:
            record = True
        if B_pressed and record:
            curr_time = time.time()
            if curr_time - last_time >= sample_time:
                print("B_pressed",'\n')
                last_time = curr_time
                record = False

if __name__ == "__main__":
    main()
