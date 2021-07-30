import pygame
import sys
import os
import numpy as np
import time



# class for the objects on the screen (agents, goals, etc.)
class Object(pygame.sprite.Sprite):

    def __init__(self, position, color, size):

        # create square sprite
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((size, size))
        self.image.fill(color)
        self.rect = self.image.get_rect()

        # initial conditions
        self.x = position[0]
        self.y = position[1]
        self.state = np.asarray([self.x, self.y])
        self.rect.x = int((self.x * 500) + 100 - self.rect.size[0] / 2)
        self.rect.y = int((self.y * 500) + 100 - self.rect.size[1] / 2)

    # move the sprite to position s
    def update(self, s):
        self.rect = self.image.get_rect(center=self.rect.center)
        self.x = s[0]
        self.y = s[1]
        self.state = np.asarray([self.x, self.y])
        self.rect.x = int((self.x * 500) + 100 - self.rect.size[0] / 2)
        self.rect.y = int((self.y * 500) + 100 - self.rect.size[1] / 2)

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

# returns n x 1 vector containing the agent positions
def getState(group):
    s = []
    for agent in group:
        s += list(agent.state)
    return np.asarray(s)

# sets the location of the agents
def updateState(group, newstate):
    n_agents = len(group)
    for idx in range(n_agents):
        s_idx = [newstate[idx * 2], newstate[idx * 2 + 1]]
        group[idx].update(s_idx)



# bayes rule with boltzmann rational
def bayes(s, a, A, G, beta=20.0):
    P = []
    for g in G:
        num = np.exp(-beta * np.linalg.norm(g - (s + a)))
        den = 0
        for ap in A:
            den += np.exp(-beta * np.linalg.norm(g - (s + ap)))
        P.append(num / den)
    P = np.asarray(P)
    return P / sum(P)




# discretize the space of actions for the team
n_actions = 5
r = 0.05
single_action = []
angles = np.linspace(-np.pi/2, np.pi/2, n_actions)
for angle in angles:
    action = [r * np.cos(angle), r * np.sin(angle)]
    single_action.append(action)
A = []
for idx in range(n_actions):
    for jdx in range(n_actions):
        for kdx in range(n_actions):
            A.append(list(single_action[idx]) + list(single_action[jdx]))#+ list(single_action[kdx]))
A = np.asarray(A)

def Legible(sprite_list, team, world, gstar_idx, gstar, A, G):
    # create game
    clock = pygame.time.Clock()
    fps = 20
    # animate
    world.fill((255,255,255))
    sprite_list.draw(world)
    pygame.display.flip()
    clock.tick(fps)

    p_aloc = np.ones(len(G))
    iter = 1
    while True:
        # hyperparameter for optimization trade-off
        epsilon = 0.01
        # constrained optimization to find revealing but efficient action
        s = getState(team)

        Q = {}
        Qmax = -np.Inf
        for a in A:
            Q[str(a)] = np.linalg.norm(gstar - s) - np.linalg.norm(gstar - (s + a))
            if Q[str(a)] > Qmax:
                Qmax = Q[str(a)]
        value = -np.Inf
        astar = None
        for a in A:
            likelihood = bayes(s, a, A, G)
            if likelihood[gstar_idx] > value and Qmax - Q[str(a)] < epsilon:
                astar = np.copy(a)
                value = likelihood[gstar_idx]

        if iter <= 15:
            # print('Iteration: ', iter)
            p_aloc = np.multiply(p_aloc, bayes(s, astar, A, G))
            if iter == 15:
                print('Probability: ',p_aloc/np.sum(p_aloc))

        # update for next time step
        updateState(team, s + astar)

        # determine when to switch to the next allocation
        if np.all(abs(s - gstar) < 0.1):
            print("[*] Done!")
            # pygame.quit(); sys.exit()
            break

        # animate
        world.fill((255,255,255))
        sprite_list.draw(world)
        pygame.display.flip()
        clock.tick(fps)

        iter +=1

    return p_aloc/np.sum(p_aloc)

def main():

    # create game
    pygame.init()
    world = pygame.display.set_mode([700,700])

    # joystick = Joystick()
    # player = Object((0.1,0.8), [140,0,255], 25)

    # define the subtasks and the possible subtask allocations
    goal1 = Object((1.0, 0.4), [100, 100, 100], 50)
    goal2 = Object((1.0, 0.6), [100, 100, 100], 50)
    goal3 = Object((0.5, 1), [100, 100, 100], 50)
    tau1 = np.asarray(list(goal1.state) +  list(goal2.state))# + list(goal1.state))
    tau2 = np.asarray(list(goal1.state) +  list(goal1.state))# + list(goal3.state))
    tau3 = np.asarray(list(goal2.state) +  list(goal2.state))# + list(goal3.state))
    tau4 = np.asarray(list(goal1.state) +  list(goal3.state))# + list(goal1.state))
    G = [tau1, tau2, tau3, tau4]

    # joystick control output
    # action, start, stop = joystick.input()
    # s_p = getState([player])
    # updateState([player], s_p + np.asarray(action)*0.04)

    P_aloc = np.empty([len(G),len(G)])
    P = []
    for gstar_idx in range(len(G)):

        print('[*] Allocation: ', gstar_idx+1)

        # add as many agents as you want
        agent1 = Object((0.1, 0.4), [0, 0, 255], 25)
        agent2 = Object((0.1, 0.6), [0, 255, 0], 25)
        agent3 = Object((0.1, 0.5), [255, 0, 0], 25)
        team = [agent1, agent2]#, agent3]#, player]


        # the game will draw everything in the sprite list
        sprite_list = pygame.sprite.Group()
        sprite_list.add(goal1)
        sprite_list.add(goal2)
        sprite_list.add(goal3)
        sprite_list.add(agent1)
        sprite_list.add(agent2)
        # sprite_list.add(agent3)
        # sprite_list.add(player)

        # pick the desired allocation
        gstar = np.copy(G[gstar_idx])
        P_aloc[gstar_idx] = Legible(sprite_list, team, world, gstar_idx, gstar, A, G)

    tau_stars = np.max(P_aloc, axis=1)
    print(tau_stars)
    print(np.argmax(tau_stars)+1)

if __name__ == "__main__":
    main()
