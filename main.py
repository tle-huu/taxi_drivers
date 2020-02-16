import numpy as np
import Agent as Agents
import json
from environment.taxi_env import TaxiEnv
from IPython.display import clear_output


env = TaxiEnv(5)



with open('config.json') as json_file:
    config = json.load(json_file)




def launchtrain():
    agent = Agents.DQNAgent(**config)
    return agent.learn()



if __name__ == '__main__':
    agent = Agents.DQNAgent(**config)
    while agent.memory.mem_cntr < agent.memory.mem_size:
        done = False
        observation = env.reset()
        
        while not done :
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
    print("done initialising")
    epsHistory = []
    scores = []
    numgames = 50
    for i in range(numgames):
        print("Essai number ",i)
        epsHistory.append(agent.epsilon)
        done = False
        observation = env.reset()
        iterations = 0
        score = 0
        while not done and iterations<200:
            iterations += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            agent.learn()
        scores.append(score)
        print(score)
            
        
        
        
        
    