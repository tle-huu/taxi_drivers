import environment.tools
#import pygame
import numpy as np
import random
import time
import torch

ACTIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "IDLE": 4}


class ActionSpace:

    def __init__(self, n):
        assert( n >= 0)
        self.n = n

    def sample(self):
        return np.random.randint(self.n)

class TaxiEnv:

    def __init__(self, map_size, number_of_cars, render = False):

        self.size = map_size
        self.number_of_cars = number_of_cars

        self.cars_positions = [{'x': 0, 'y': 0} for _ in range(number_of_cars)]

        self.map = np.zeros([map_size, map_size])

        self.action_space = ActionSpace(5 ** self.number_of_cars)
        self.state_space_size = self.size ** (2 * (number_of_cars + 1))

        self.destination_position_ = {'x': int(map_size / 2), 'y': int(map_size / 2)}

        self.parse()

        if render:
            # Lazy import
            from environment.renderer import Renderer
            self.renderer = Renderer(map_size, self.map, self.number_of_cars)
            self.renderer.set_cars_position(self.cars_positions)
            self.renderer.set_destination_position(self.destination_position_)

        ## Meant to be used by an external to change reward dynamically
        self.reward = {}


    ## TODO: Make the parsing Goal agnostic
    def parse(self):
        self.map, self.car_position_, self.destination_position_ = environment.tools.parser("environment/map_2.txt")

    def info(self):
        pass

    def reset(self):

        self.parse()

        for i in range(self.number_of_cars):
            self.cars_positions[i] = self.generate_random_correct_position_()

        ## Uncomment to randomize destination position
        # self.destination_position_ = self.generate_random_correct_position_()

        return self.encode_space(self.cars_positions, self.destination_position_)

    def console_render(self):
        # TODO
        pass

    def render(self):

        self.renderer.render()


    def step(self, action):

        action_array = self.decode_action(action)
        action_array.reverse()

        reward = 0
        number_of_cars_in_goal_square = 0

        for i in range(self.number_of_cars):


            car_position = self.cars_positions[i]
            action = action_array[i]

            # Used to move the car back to its first position if hitting a wall
            current = car_position.copy()
            hit_a_wall = False

            self.move_car_(car_position, action)

            ## If the car happens to have moved in a wall, putting it back to its road square
            if self.position_value_(car_position) == -1:
                self.cars_positions[i] = current
                hit_a_wall = True

            ## Gaining a point for being on the destination square
            if car_position == self.destination_position_:
                reward += 1
                number_of_cars_in_goal_square += 1

            ## A wall is very bad
            if hit_a_wall:
                reward -= 5
            ## Cars are burnt until reaching the goal
            else:
                reward -= self.number_of_cars

        ## Episode ends when all the cars have reached the goal
        done = (number_of_cars_in_goal_square == self.number_of_cars)

        return self.encode_space(self.cars_positions, self.destination_position_), reward, done, None


    def set_reward(new_reward):
        self.reward = new_reward


    def encode_space(self, cars_positions, destination):

        res = 0

        for i in range(len(cars_positions)):

            res += cars_positions[i]['y']
            res *= self.size

            res += cars_positions[i]['x']
            res *= self.size

        # Destination col
        res += destination['y']
        res *= self.size

        # Destination row
        res += destination['x']
        return res

    def decode_space(self, encoded_state):

        out = []

        destination_x = encoded_state % self.size
        encoded_state = encoded_state // self.size

        destination_y = encoded_state % self.size
        encoded_state = encoded_state // self.size

        map_out = np.zeros((self.size, self.size, 3))

        for i in range(map_out.shape[0]):
          for j in range(map_out.shape[1]):
            if self.map[i, j] == 0:
              map_out[i, j] = np.array([0, 100, 0])

        colors = [ np.array([0, 0, 200]), np.array([200, 0, 0]) ]

        # map_out[destination_y][destination_x] = np.array([100, 100, 66])

        stride = int(254 / self.number_of_cars)

        for i in range(self.number_of_cars):

            x = encoded_state % self.size
            encoded_state = encoded_state // self.size

            y = encoded_state % self.size
            encoded_state = encoded_state // self.size

            # map_out[y][x] = np.array([0, 0, (i + 1) * stride])
            map_out[y][x] += colors[i]

            # map_out[y][x] = np.array([])
            # map_out[y][x] += 1
            
        map_out[destination_y][destination_x] = np.array([100, 100, 66])

        return map_out.T

    def encode_action(self, actions_array):

        res = 0
        for i in range(len(actions_array) - 1):

            res += actions_array[i]
            res *= 5

        res += actions_array[-1]

        return res

    def decode_action(self, encoded_action):

        action_array = []

        for i in range(self.number_of_cars):

            action_array.append(encoded_action % 5)
            encoded_action  = encoded_action // 5

        return action_array
 

############################### PRIVATE ################################
    def position_value_(self, position_dict):

        return self.map[position_dict['y'], position_dict['x']]


    # Generating a position in a non-wall position
    def generate_random_correct_position_(self):

        position = {'x': int(random.randint(0, self.size - 1)), \
                    'y': int(random.randint(0, self.size - 1))}

        # Making sure the coordinates are not a wall
        while self.position_value_(position) == -1:
            position['x'] = int(random.randint(0, self.size - 1))
            position['y'] = int(random.randint(0, self.size - 1))

        return position

    def move_car_(self, car_position, action):

        if action == ACTIONS["UP"] and car_position['y'] > 0:
            car_position['y'] -= 1
        elif action == ACTIONS["RIGHT"] and car_position['x'] < self.size - 1:
            car_position['x'] += 1
        elif action == ACTIONS["DOWN"] and car_position['y'] < self.size - 1:
            car_position['y'] += 1
        elif action == ACTIONS["LEFT"] and car_position['x'] > 0:
            car_position['x'] -= 1



