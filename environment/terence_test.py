import gym
import math
import random
import numpy as np
import sys
import time

from taxi_env import TaxiEnv

from IPython.display import clear_output


SIZE = 8
env = TaxiEnv(SIZE, 2)

action_space_size = env.action_space.n
state_space_size = env.state_space_size

q_table = np.zeros([state_space_size, action_space_size])

NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 200


## Alpha
LEARNING_RATE = 0.01

## Gamma
DISCOUNT_RATE = 0.95

## Epsilon
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.05
exploration_decay_rate = 0.0001


rewards_all_episodes = []



UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

KEY_TO_ACTION = {"a":  LEFT, "w": UP, "d": RIGHT, "s": DOWN}

for episode in range(NUM_EPISODES):

    state = env.reset()
    done = False

    reward_current_episode = 0

    if episode % 1000 == 0:
        print("Processed %d" % int((episode / NUM_EPISODES) * 100))

    for step in range(MAX_STEPS_PER_EPISODE):

        exploration_rate_threshold = random.uniform(0, 1)

        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(q_table[new_state, :]))

        state = new_state
        env.decode_space(state)
        reward_current_episode += reward

        if done:
            break

    if exploration_rate >= min_exploration_rate:
        exploration_rate -= exploration_decay_rate

    rewards_all_episodes.append(reward_current_episode)

print("Traning done")
print(q_table)

print(np.mean(rewards_all_episodes), np.min(rewards_all_episodes), np.max(rewards_all_episodes))

done = False
state = env.reset()

r = []

wins = 0
lol = 1000
for i in range(lol):
    env.reset()
    current_reward = 0
    done = False
    i = 0
    while not done and i < 100:
        # env.render()
        action = np.argmax(q_table[state, :])
        state, reward, done, info = env.step(action)
        current_reward += reward
        # time.sleep(0.300)
        i+=1
    # env.render()
    # env.render()
    if done:
        print(" ############## GOAL ################3 ")
        wins += 1
    else:
        print(" FELL ")
    # time.sleep(2)

    r.append(current_reward)

print(wins / lol)
# env.close()






