import gym
import math
import random
import numpy as np
import sys
import time

# from IPython.display import clear_output

env = gym.make('FrozenLake-v0')
#env = gym.make('Taxi-v3')

action_space_size = env.action_space.n
state_space_size = env.observation_space.n


print(env.action_space)
print(env.observation_space)

q_table = np.zeros([state_space_size, action_space_size])



NUM_EPISODES = 40000
MAX_STEPS_PER_EPISODE = 80


## Alpha
LEARNING_RATE = 0.90

## Gamma
DISCOUNT_RATE = 0.90

## Epsilon
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.05
exploration_decay_rate = 0.001


rewards_all_episodes = []


#  0 ==> LEFT
#  1 ==> DOWN
#  2 ==> RIGHT
#  3 ==> UP

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

KEY_TO_ACTION = {"a":  LEFT, "w": UP, "d": RIGHT, "s": DOWN}

# state = env.reset()

# while True:
#     env.render()
#     print(state)
#     action = input()
#     state  = env.step(KEY_TO_ACTION[action])
# sys.exit(1)



for episode in range(NUM_EPISODES):

    state = env.reset()
    done = False

    reward_current_episode = 0

    for step in range(MAX_STEPS_PER_EPISODE):

        exploration_rate_threshold = random.uniform(0, 1)

        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        # print("xxx == >", reward)

        q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + LEARNING_RATE * (reward + DISCOUNT_RATE * np.max(q_table[new_state, :]))

        state = new_state

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
for i in range(100):
    env.reset()
    current_reward = 0
    done = False
    # print("####################################")
    # print("################  %d #################" % i)
    # print("####################################")
    while not done:
        env.render()
        action = np.argmax(q_table[state, :])
        state, reward, done, info = env.step(action)
        current_reward += reward
        time.sleep(0.300)
    r.append(current_reward)
    env.render()

lol = 0
for x in r:
    lol += x
print(np.mean(r), lol)
env.close()






