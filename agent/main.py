import os
os.chdir("..")
import numpy as np
import agent.Agent as Agents
import json
from environment.taxi_env import TaxiEnv
from IPython.display import clear_output
import torch
import time





with open('agent/config.json') as json_file:
    config = json.load(json_file)


env = TaxiEnv(config["input_dims"][-1],config["number_of_cars"])



if __name__ == '__main__':
    agent = Agents.DQNAgent(**config)
    while agent.memory.mem_cntr < agent.memory.mem_size:
        done = False
        observation = env.reset()
        it = 0
        while not done and it <40:
            it += 1
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.store_transition(env.decode_space(observation),
                                   env.decode_action(action),
                                   reward,
                                   env.decode_space(observation_),
                                   done)
            observation = observation_
    print(agent.memory.state_memory)
    epsHistory = []
    scores = []
    numgames = 20
    for i in range(numgames):
        print("Essai number ",i)
        epsHistory.append(agent.epsilon)
        done = False
        observation = env.reset()
        iterations = 0
        score = 0
        while not done and iterations<200:
            iterations += 1
            action = env.encode_action(agent.choose_action(env.decode_space(observation)))
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(env.decode_space(observation),
                                   env.decode_action(action),
                                   reward,
                                   env.decode_space(observation_),
                                   done)
            observation = observation_
            agent.learn()
        scores.append(score)
        print(score)
    print("lets test")
    success = 0
    tests = 20
    for i in range(tests):
        state = env.decode_space(env.reset())
        env.render()
        done = False
        it = 0
        with torch.no_grad():
            while not done and it<40:
                it += 1
                state = torch.tensor([state],dtype=torch.float).to(agent.q_eval.device)
                actions = agent.q_eval.forward(state)
                action = torch.argmax(actions, dim=2)
                observation, reward, done, info = env.step(action)
                time.sleep(0.3)
                env.render()
                state = env.decode_space(observation)
            if done :
                success+=1
    print(success/tests)
            
    
    
    
        
        
        
        
    