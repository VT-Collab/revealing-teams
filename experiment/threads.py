import time
import numpy as np
import pygame
import sys
import rospy
import kinpy as kp
import threading


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




def fetchThread(e):
    while True:
        even_is_set = e.wait()
        print('[*] loop1 is running')

def pandaThread(e):
    while True:
        even_is_set = e.wait()
        print('[*] loop2 is running','\n')



def main():
    interface = Joystick()

    e = threading.Event()

    t1 = threading.Thread(target = fetchThread, args=(e,))
    t2 = threading.Thread(target = pandaThread, args=(e,))

    t1.start()
    t2.start()
    # t1.join()

    last_time = 0.0
    sample_time = 0.2
    t = 0.1
    e.set()
    # while True:
    A_pressed, B_pressed, Start_pressed = interface.input()
    if Start_pressed:
        print("Start_pressed")
    if A_pressed:
        e.clear()
    if B_pressed:
        curr_time = time.time()
        if curr_time - last_time >= sample_time:
            print("B_pressed",'\n')
            last_time = curr_time
            e.set()


if __name__ == "__main__":
    main()
