import numpy as np
import agent.Agent as Agents
import json
from environment.taxi_env import TaxiEnv

from matplotlib import pyplot as plt
from tqdm import tqdm
import torch


with open('agent/config.json') as json_file:
    config = json.load(json_file)

if config["render_for_eval"]:
    import time

env = TaxiEnv(config["agent"]['input_dims'][-1],
              config["agent"]['number_of_cars'],
              config["render_for_eval"])


if __name__ == '__main__':

    agent = Agents.DQNAgent(**config["agent"])

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
    total_losses = []
    total_successes = []
    success = 0
    NUMGAMES = config["games_for_training"]
    for i in tqdm(range(NUMGAMES), desc = 'games for training'):
        epsHistory.append(agent.epsilon)
        done = False
        observation = env.reset()
        iterations = 0

        loss_mean_episode = 0
        episode_reward = 0
        while not done and iterations < config["depth_for_traintries"]:
            iterations += 1
            action_array = agent.choose_action(env.decode_space(observation))

            ## Encoding action
            ## TODO: removing in the env the encoding stuff
            action = env.encode_action(action_array)
            observation_, reward, done, info = env.step(action)

            agent.store_transition(env.decode_space(observation),
                                   action_array,
                                   reward,
                                   env.decode_space(observation_),
                                   done)

            observation = observation_
            loss = agent.learn()

            loss_mean_episode += loss

            episode_reward += reward

        print("[%d] reward [%d] " % (i, episode_reward))
        if done:
            success += 1
        
        total_successes.append(success)

        total_rewards.append(episode_reward)
        total_losses.append(loss_mean_episode / iterations)
        
        ##Plots
        fig, axes = plt.subplots(figsize=(15, 5),nrows = 1, ncols = 3)
        ax1,ax2,ax3 = axes.flatten()
        ax1.plot(total_rewards)
        ax1.set_title("Rewards")
        ax2.plot(total_losses)
        ax2.set_title("Losses")
        ax3.plot(total_successes)
        ax3.set_title("Successes")

        plt.show()
    

    print("Training success rate %03f" % (success / NUMGAMES))

    print("Saving model and config to disk...")
    torch.save({"weights" : agent.q_eval.state_dict(), 
                "config" : config["agent"]}, 
                config["path_to_save"]+"_with_{}cars_for_{}traingames".format(config["agent"]["number_of_cars"],
                                                                              config["games_for_training"]))

    agent.q_eval.eval()
    print("Evaluation ...")
    success = 0

    total_rewards = []
    iterations = []
    tests = config["games_for_evaluation"]
    for i in tqdm(range(tests), desc='tests'):
        state = env.decode_space(env.reset())
        if config["render_for_eval"]:
            env.render()
        done = False
        it = 0
        with torch.no_grad():
            current_reward = 0
            while not done and it < config["depth_for_testtries"]:
                it += 1
                state = torch.tensor([state],dtype=torch.float).to(agent.q_eval.device)

                actions = agent.q_eval.forward(state)
                action = torch.argmax(actions, dim = 2).squeeze(0)

                observation, reward, done, info = env.step(env.encode_action(action))

                current_reward += reward
                if config["render_for_eval"]:
                    time.sleep(0.3)
                    env.render()
                state = env.decode_space(observation)
            if done:
                success += 1
            total_rewards.append(current_reward)

            iterations.append(it)
    
    print("Evaluation success rate %03f" % (success / tests))

    print("Average number of step: %d" % np.mean(iterations))
    print("Average reward: %d" % np.mean(total_rewards))
            
    
    
    
        
        
        
        
    