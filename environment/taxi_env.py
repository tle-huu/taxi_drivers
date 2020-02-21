import environment.tools
import pygame
import numpy as np
import random
import time
from environment.renderer import Renderer

ACTIONS = {"UP": 0, "RIGHT": 1, "DOWN": 2, "LEFT": 3, "IDLE": 4}


class ActionSpace:

    def __init__(self, n):
        assert( n >= 0)
        self.n = n

    def sample(self):
        return np.random.randint(self.n)

class TaxiEnv:

    MUR = -1

    def __init__(self, map_size, number_of_cars):

        self.size = map_size
        self.number_of_cars = number_of_cars

        self.cars_positions = [{'x': 0, 'y': 0} for _ in range(number_of_cars)]

        self.map = np.zeros([map_size, map_size])

        self.action_space = ActionSpace(5 ** self.number_of_cars)
        self.state_space_size = self.size ** (2 * (number_of_cars + 1))

        self.car_position_ = {'x': 0, 'y': 0}
        self.destination_position_ = {'x': int(map_size / 2), 'y': int(map_size / 2)}

        self.parse()

        self.renderer = Renderer(map_size, self.map, self.number_of_cars)
        self.renderer.set_cars_position(self.cars_positions)
        self.renderer.set_destination_position(self.destination_position_)

        self.ACTIONS = [ACTIONS["UP"], ACTIONS["RIGHT"], ACTIONS["DOWN"], ACTIONS["LEFT"], ACTIONS['IDLE']]

        self.reward = {'reached': 50, 'bad': -1}


    def parse(self):
        self.map, self.car_position_, self.destination_position_ = environment.tools.parser("environment/map_2.txt")

    def info(self):
        pass

    def coord_to_int(self, car_position):
        return car_position['y'] * self.size + car_position['x']

    def int_to_coord(self, car_position_number):
        return {'x': car_position_number % self.size, 'y': car_position_number // self.size}

    def reset(self):

        # self.map[self.car_position_['y'], self.car_position_['x']] = 1
        self.parse()

        # self.destination_position_['x'] = int(random.randint(0, self.size - 1))
        # self.destination_position_['y'] = int(random.randint(0, self.size - 1))

        for i in range(self.number_of_cars):
            car_position = self.cars_positions[i]

            car_position['x'] = int(random.randint(0, self.size - 1))
            car_position['y'] = int(random.randint(0, self.size - 1))

            while self.position_value(car_position) == -1:
                car_position['x'] = int(random.randint(0, self.size - 1))
                car_position['y'] = int(random.randint(0, self.size - 1))

        # while self.position_value(self.destination_position_) == -1:
        #     self.destination_position_['x'] = int(random.randint(0, self.size - 1))
        #     self.destination_position_['y'] = int(random.randint(0, self.size - 1))

        return self.encode_space(self.cars_positions, self.destination_position_)

    def console_render(self):

        for y in range(self.size):
            for x in range(self.size):
                if  {'x': x, 'y':y} == self.car_position_:
                    print("X", end='')
                elif self.map[y,x] == 2:
                    print("G", end='')
                elif self.map[y,x] == -1:
                    print("#", end='')
                elif self.map[y,x] == 50000:
                    print("S", end='')
                else:
                    print(" ", end='')
            print("")
        print("")

    def render(self):

        # self.console_render()
        self.renderer.render()



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

        # map_out = self.map.copy()

        map_out[destination_y][destination_x] = np.array([100, 100, 66])

        for i in range(self.number_of_cars):

            x = encoded_state % self.size
            encoded_state = encoded_state // self.size

            y = encoded_state % self.size
            encoded_state = encoded_state // self.size

            if self.map[y][x] == -1:
                map_out[y][x] = np.array([0, 0, 0])
            elif self.map[y][x] == 0:
                map_out[y][x] = np.array([65, 74, 0])
            elif self.map[y][x] > 0:
                map_out[y][x] = np.array([0, 0, 255])

            # map_out[y][x] = np.array([])
            # map_out[y][x] += 1

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

    def step(self, action):

        action_array = self.decode_action(action)
        action_array.reverse()

        reward = 0

        done = 0

        for i in range(self.number_of_cars):
            mur = 0

            car_position = self.cars_positions[i]
            action = action_array[i]

            current = car_position.copy()

            if action == ACTIONS["UP"] and car_position['y'] > 0:
                car_position['y'] -= 1
            elif action == ACTIONS["RIGHT"] and car_position['x'] < self.size - 1:
                car_position['x'] += 1
            elif action == ACTIONS["DOWN"] and car_position['y'] < self.size - 1:
                car_position['y'] += 1
            elif action == ACTIONS["LEFT"] and car_position['x'] > 0:
                car_position['x'] -= 1

            if self.position_value(car_position) == -1:
                self.cars_positions[i] = current
                mur += 1

            if car_position == self.destination_position_:
                reward += 100
                done += 1
            # elif current == self.destination_position_:
            #     reward -= 500
            # elif car_position == current:
            #     reward -= 50
            if mur > 0:
                reward -= 10
                # done = 1
            else:
                reward -= 1

        done = (done == self.number_of_cars)

        return self.encode_space(self.cars_positions, self.destination_position_), reward, done, None

    def set_reward(new_reward):
        self.reward = new_reward

    def position_value(self, position_dict):

        return self.map[position_dict['y'], position_dict['x']]


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
