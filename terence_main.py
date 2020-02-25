import os
import sys
import numpy as np
import agent.Agent as Agents
import json
from environment.taxi_env import TaxiEnv
from IPython.display import clear_output

from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import time


with open('agent/config.json') as json_file:
    config = json.load(json_file)

env = TaxiEnv(config['input_dims'][-1], config['number_of_cars'])


if __name__ == '__main__':

    # print(agent.q_eval.state_dict().keys())
    # print(list(agent.q_eval.parameters()))

    # test = torch.rand(1, 3, 8, 8)

    # out = agent.q_eval(test).argmax(dim = 2)


    # print("ut : ", out.shape)
    # print("ut : ", out)
    # print("ut : ", out.squeeze(0))
    print("Ending")

    agent = Agents.DQNAgent(**config)

    print("Filling the replay memory buffer")
    while agent.memory.mem_cntr < agent.memory.mem_size:
        done = False
        observation = env.reset()
        it = 0
        while not done and it < 100:
            it += 1
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.store_transition(env.decode_space(observation),
                                   env.decode_action(action),
                                   reward,
                                   env.decode_space(observation_),
                                   done)
            observation = observation_
      
    print("Replay memory buffer filled")

    epsHistory = []
    total_rewards = []
    success = 0
    NUMGAMES = 200
    for i in tqdm(range(NUMGAMES), desc = 'games for training'):
        epsHistory.append(agent.epsilon)
        done = False
        observation = env.reset()
        iterations = 0

        episode_reward = 0
        while not done and iterations < 200:
            iterations += 1
            action_array = agent.choose_action(env.decode_space(observation))

            # Encoding action
            # TODO: removing in the env the encoding stuff
            action = env.encode_action(action_array)
            observation_, reward, done, info = env.step(action)

            agent.store_transition(env.decode_space(observation),
                                   action_array,
                                   reward,
                                   env.decode_space(observation_),
                                   done)
            observation = observation_
            agent.learn()

            episode_reward += reward

        print("[%d] reward [%d] " % (i, episode_reward))
        if done and reward > 0:
            success += 1

        total_rewards.append(episode_reward)

        # plt.plot(total_rewards)
        # plt.show()
    

    print("Training success rate %03f" % (success / NUMGAMES))

    print("Saving model to disk...")
    torch.save(agent.q_eval.state_dict(), "one_car_model.dat")

    print("Evaluation ...")
    success = 0
    iterations = []
    tests = 1000
    for i in tqdm(range(tests), desc='tests'):
        state = env.decode_space(env.reset())
        # env.render()
        done = False
        it = 0
        with torch.no_grad():
            while not done and it < 50:
                it += 1
                state = torch.tensor([state],dtype=torch.float).to(agent.q_eval.device)
                actions = agent.q_eval.forward(state)
                action = torch.argmax(actions).item()
                observation, reward, done, info = env.step(action)
                # time.sleep(0.3)
                # env.render()
                state = env.decode_space(observation)
            if done and reward > 0:
                success += 1
            iterations.append(it)
    print("Evaluation success rate %03f" % (success / tests))

    print("Average number of step: %d" % np.mean(iterations))
            
    
    
    
        
        
        
        
    