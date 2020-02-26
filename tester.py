import torch
import numpy as np
from agent.DRL import CarLeader
import json
import time
from environment.taxi_env import TaxiEnv
from tqdm import tqdm





with open('config.json') as json_file:
    config = json.load(json_file)


if __name__ == '__main__':
    weights, configmodel = torch.load(config["path_weights_to_test"], 
                                 map_location=torch.device("cpu")).values()
    
    model = CarLeader(configmodel["input_dims"][1:], configmodel["number_of_cars"])
    
    env = TaxiEnv(configmodel["input_dims"][-1],
                  configmodel["number_of_cars"],
                  render=True)

    model.load_state_dict(weights)
    model.eval()
    success = 0
    iterations = []
    tests = config["games_for_test"]
    for i in tqdm(range(tests), desc='tests'):
        state = env.decode_space(env.reset())
        env.render()
        done = False
        it = 0
        with torch.no_grad():
            while not done and it < config["depth_for_test"]:
                it += 1
                state = torch.tensor([state],dtype=torch.float)
                actions = model.forward(state)
                action = torch.argmax(actions, dim = 2).squeeze(0)
                observation, reward, done, info = env.step(action)
                time.sleep(0.3)
                env.render()
                state = env.decode_space(observation)
            if done and reward > 0:
                success += 1
            iterations.append(it)
    print("Evaluation success rate %03f" % (success / tests))

    print("Average number of step: %d" % np.mean(iterations))