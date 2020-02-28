import environment.tools
#import pygame
import numpy as np
import random
import time
import torch
import sys

ACTIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "IDLE": 4}

PIXEL_MAX_VALUE = 255

class ActionSpace:

    def __init__(self, n):
        assert( n >= 0)
        self.n = n

    def sample(self):
        return np.random.randint(self.n)

class TaxiEnv:

    def __init__(self, map_size, number_of_cars, map_file_path, render = False):

        self.map_file_path = map_file_path
        self.size = map_size
        self.number_of_cars = number_of_cars


        ## This is hardcoded to work with a number of cars <= 5
        STRIDE = 40

        if self.number_of_cars > 5:
            print("Error: Too many cars, can't discriminate")
            ## TODO: throw
            sys.exit(1)

        self.COLORS_ = []        
        # for i in range(1, int(self.number_of_cars / 2) + 2):
        #     self.COLORS_.append( np.array([255 - STRIDE * i, 0, 0]) )
        #     self.COLORS_.append( np.array([0, 0, 255 - STRIDE * i]) )

        # self.COLORS_.append( np.array([255, 0, 0]) )
        # self.COLORS_.append( np.array([0, 0, 255]) )
        # self.COLORS_.append( np.array([100, 0, 0]) )
        # self.COLORS_.append( np.array([0, 0, 100]) )
        # self.COLORS_.append( np.array([150, 150, 150]) )

        ## 3 cars colors test
        self.COLORS_.append( np.array([255, 0, 0]) )
        self.COLORS_.append( np.array([0, 0, 255]) )
        self.COLORS_.append( np.array([0, 255, 0]) )


        self.map = np.zeros([map_size, map_size])

        self.action_space = ActionSpace(5 ** self.number_of_cars)
        self.state_space_size = self.size ** (2 * (number_of_cars + 1))


        self.parse()

        ## Meant to be used by an external to change reward dynamically
        self.reward = {}

        ## Initializing to negative number every coordinate to catch early bug
        self.destination_position_ = {'x': -1337, 'y': -1337}
        self.cars_positions = [{'x': -1337, 'y': -1337} for _ in range(number_of_cars)]

        self.map_out = np.zeros((self.size, self.size, 3))

        if render:
            # Lazy import
            from environment.renderer import Renderer
            self.renderer = Renderer(map_size, self.map_out, self.number_of_cars, self.COLORS_)
            self.renderer.set_cars_position(self.cars_positions)
            self.renderer.set_destination_position(self.destination_position_)

    ## TODO: Make the parsing Goal agnostic
    def parse(self):
        self.map, self.car_position_, self.destination_position_ = environment.tools.parser(self.map_file_path)

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

        self.renderer.set_map(self.map_out)
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

            self.move_car_(i, car_position, action)

            ## If the car happens to have moved in a wall, putting it back to its road square
            if self.position_value_(car_position) == -1:
                self.cars_positions[i] = current
                hit_a_wall = True

            ## Gaining a point for being on the destination square
            if car_position == self.destination_position_:
                reward += 5
                number_of_cars_in_goal_square += 1

            ## A wall is very bad
            if hit_a_wall:
                reward -= 5
            ## Cars are burnt until reaching the goal
            else:
                reward -= self.number_of_cars

        ## Episode ends when all the cars have reached the goal
        done = (number_of_cars_in_goal_square == self.number_of_cars)
        # self.decode_space(self.encode_space(self.cars_positions, self.destination_position_))
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

        self.map_out = np.zeros((self.size, self.size, 3))

        for i in range(self.map_out.shape[0]):
            for j in range(self.map_out.shape[1]):
                # roads are green
                if self.map[i, j] == 0:
                    self.map_out[i, j] = np.array([255, 255, 255])

        for i in range(self.number_of_cars):

            ## Decoding coordinates
            x = encoded_state % self.size
            encoded_state = encoded_state // self.size

            y = encoded_state % self.size
            encoded_state = encoded_state // self.size

            if self.map_out[y, x][0] == 255 and self.map_out[y, x][1] == 255 and self.map_out[y, x][2] == 255:
                self.map_out[y, x] = np.array([0, 0, 0])

            ## Coloring car's square with the car's color
            self.map_out[y][x] += self.COLORS_[i]

        self.map_out[destination_y][destination_x] = np.array([235, 225, 52])

        return self.map_out.T

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

    def move_car_(self, car_id, car_position, action):

        collision = False

        for i in range(self.cars_positions):
            if i == car_id:
                continue

            if self.cars_positions[i] == car_position:
                collision = True
                break

        ## Jam
        if collision and np.random.random() < 0.4:
            return


        if action == ACTIONS["UP"] and car_position['y'] > 0:
            car_position['y'] -= 1
        elif action == ACTIONS["RIGHT"] and car_position['x'] < self.size - 1:
            car_position['x'] += 1
        elif action == ACTIONS["DOWN"] and car_position['y'] < self.size - 1:
            car_position['y'] += 1
        elif action == ACTIONS["LEFT"] and car_position['x'] > 0:
            car_position['x'] -= 1



