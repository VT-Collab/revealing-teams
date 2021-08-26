# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
from datetime import datetime


# class for the player joystick
class Joystick(object):

    def __init__(self):
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.DEADBAND = 0.1

    def input(self):
        pygame.event.get()
        z1 = self.gamepad.get_axis(0)
        if abs(z1) < self.DEADBAND:
            z1 = 0.0
        z2 = self.gamepad.get_axis(1)
        if abs(z2) < self.DEADBAND:
            z2 = 0.0
        start = self.gamepad.get_button(1)
        stop = self.gamepad.get_button(0)
        return [z1, z2], start, stop

class GamePlay(Game):
    def __init__(self, filename, world, sim_agents):
        Game.__init__(self, world, sim_agents, play=True)
        self.filename = filename
        self.save_dir = 'misc/game/screenshots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # tally up all gridsquare types
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            # Save current image
            if event.key == pygame.K_RETURN:
                image_name = '{}_{}.png'.format(self.filename, datetime.now().strftime('%m-%d-%y_%H-%M-%S'))
                pygame.image.save(self.screen, '{}/{}'.format(self.save_dir, image_name))
                print('just saved image {} to {}'.format(image_name, self.save_dir))
                return

            # Switch current agent
            if pygame.key.name(event.key) in "1234":
                try:
                    self.current_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                except:
                    pass
                return

            # Control current agent
            x, y = self.current_agent.location
            # if event.key in KeyToTuple.keys():
            #     action = KeyToTuple[event.key]
            #     self.current_agent.action = action
            #     interact(self.current_agent, self.world)
            action, start, stop = self.joystick.input()
            print(action)
            self.current_agent.action = action
            interact(self.current_agent, self.world)

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()
