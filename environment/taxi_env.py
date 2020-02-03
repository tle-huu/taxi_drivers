import tools
import pygame
import numpy as np
import random
import time
from renderer import Renderer

ACTIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3}

class TaxiEnv:

    class ActionError(Exception):
        pass

    def __init__(self, size):

        self.size = size
        self.map = np.zeros([size, size])


        self.car_position_ = [0, 0]
        self.destination_position_ = [int(size / 2), int(size / 2)]

        # self.parse()
        self.renderer = Renderer(size, self.map)

        self.ACTIONS = [ACTIONS["UP"], ACTIONS["RIGHT"], ACTIONS["DOWN"], ACTIONS["LEFT"]]

        self.reward = {'reached': 50, 'bad': -1}

    def update_map(self, current):
        self.map[current[1], current[0]] = 0
        self.map[self.car_position_[1], self.car_position_[0]] = 1

    def parse(self):
        self.map, self.car_position_, self.destination_position_ = tools.parser("map.txt")

    def info(self):
        pass

    def reset(self):

        self.map = np.zeros([self.size, self.size])
        self.car_position_ = [0, 0]
        # self.destination_position_ = [int(self.size / 2), int(self.size / 2)]


        # self.car_position_ = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]
        self.destination_position_ = [int(random.randint(0, self.size - 1)), int(random.randint(0, self.size - 1))]
        # self.destination_position_ = [2, 2]

        # while self.car_position_ == self.destination_position_:
        #     self.destination_position_ = [random.randint(0, self.size - 1), random.randint(0, self.size - 1)]

        self.map[self.car_position_[1], self.car_position_[0]] = 1
        self.map[self.destination_position_[1], self.destination_position_[0]] = 2

        self.renderer.update_map(self.map)

        return self.lol()

    def console_render(self):
        print("#" * (self.size + 2))

        for y in range(self.size):
            print("#", end='')
            for x in range(self.size):
                if self.map[y,x] == 1:
                    print("X", end='')
                elif self.map[y,x] == 2:
                    print("G", end='')
                else:
                    print(" ", end='')
            print("#")
        print("#" * (self.size + 2))

    def render(self):

        # self.console_render()
        self.renderer.render()

    def lol(self):

        return self.car_position_[1] * 8 + self.car_position_[0]

    def encode(self, taxi_row, taxi_col, destination):

        res = taxi_col
        res *= self.size

        res += taxi_row
        res *= self.size

        # Destination col
        res += destination[0]
        res *= self.size

        # Destination row
        res += destination[1]
        return res

    def decode(self, encoded_state):

        out = []

        # destination row
        out.append(encoded_state % self.size)
        encoded_state = encoded_state // self.size

        # destination col
        out.append(encoded_state % self.size)
        encoded_state = encoded_state // self.size

        # Row (y)
        encoded_state = encoded_state // self.size
        out.append(encoded_state % self.size)

        # Column (x)
        encoded_state = encoded_state // self.size
        out.append(encoded_state % self.size)

        assert 0 <= encoded_state < self.size

        out.reverse()

        # out is [col, row, dest_col, dest_row]
        return out


    def step(self, action):

        if action not in self.ACTIONS:
            raise ActionError("Actions does not exist")

        current = self.car_position_[0], self.car_position_[1]

        if action == ACTIONS["UP"] and self.car_position_[1] > 0:
            self.car_position_[1] -= 1
        elif action == ACTIONS["RIGHT"] and self.car_position_[0] < self.size - 1:
            self.car_position_[0] += 1
        elif action == ACTIONS["DOWN"] and self.car_position_[1] < self.size - 1:
            self.car_position_[1] += 1
        elif action == ACTIONS["LEFT"] and self.car_position_[0] > 0:
            self.car_position_[0] -= 1

        self.update_map(current)

        done = int(self.car_position_ == self.destination_position_)

        if done:
            reward = self.reward['reached']
        else:
            reward = self.reward['bad']

        self.renderer.update_map(self.map)



        return self.encode(self.car_position_[0], self.car_position_[1], (self.destination_position_[0], self.destination_position_[0])), reward, done, None

    def set_reward(new_reward):
        self.reward = new_reward


def main():

    env = TaxiEnv(10)
    env.reset()
    env.render()
    time.sleep(1)
    env.step(1)
    env.render()
    time.sleep(1)
    env.step(1)
    env.render()
    time.sleep(1)
    env.step(1)
    env.render()
    time.sleep(1)
    env.step(1)
    env.render()
    time.sleep(1)

    # env.renderer.start()

if __name__ == "__main__":
    main()
