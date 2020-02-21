import os
# os.chdir("..")
import numpy as np
import agent.Agent as Agents
import json
from environment.taxi_env import TaxiEnv
from IPython.display import clear_output
import torch
import time

env = TaxiEnv(8, 1)



with open('agent/config.json') as json_file:
    config = json.load(json_file)


if __name__ == '__main__':

    agent = Agents.DQNAgent(**config)
    while agent.memory.mem_cntr < agent.memory.mem_size:
        done = False
        observation = env.reset()
        it = 0
        while not done and it < 100:
            it += 1
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.store_transition(env.decode_space(observation),
                                   action,
                                   reward,
                                   env.decode_space(observation_),
                                   done)
            observation = observation_
    epsHistory = []
    scores = []
    success = 0
    NUMGAMES = 100
    for i in range(NUMGAMES):
        epsHistory.append(agent.epsilon)
        done = False
        observation = env.reset()
        iterations = 0
        score = 0

        episod_reward = 0
        while not done and iterations < 200:
            iterations += 1
            action = agent.choose_action(env.decode_space(observation))
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(env.decode_space(observation),
                                   action,
                                   reward,
                                   env.decode_space(observation_),
                                   done)
            observation = observation_
            agent.learn()

            episod_reward += reward

        print("[%d] reward [%d] " % (i, episod_reward))
        if done and reward > 0:
            success += 1

        scores.append(score)
        # print(score)

    print("Training success rate %03f" % (success / NUMGAMES))

    print("Evaluation ...")
    success = 0
    iterations = []
    tests = 1000
    for i in range(tests):
        state = env.decode_space(env.reset())
        # env.render()
        done = False
        it = 0
        with torch.no_grad():
            while not done and it < 100:
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
            
    
    
    
        
        
        
        
    