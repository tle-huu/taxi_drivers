import gym
import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from torch.autograd import Variable

class DQNAgent(nn.Module):

    def __init__(self, env_id, path, episodes, max_env_steps, win_threshold, epsilon_decay,
                 state_size=None, action_size=None, epsilon=1.0, epsilon_min=0.01, 
                 gamma=1, alpha=.01, alpha_decay=.01, batch_size=16, prints=False):

        super(DQNAgent, self).__init__()
        self.memory = deque(maxlen = 100000)

        self.env = gym.make(env_id)

        if state_size is None: 
            self.state_size = self.env.observation_space.n 
        else: 
            self.state_size = state_size
 
        if action_size is None: 
            self.action_size = self.env.action_space.n 
        else: 
            self.action_size = action_size
 
        self.episodes = episodes
        self.env._max_episode_steps = max_env_steps
        self.win_threshold = win_threshold
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.batch_size = batch_size
        self.path = path                     #location where the model is saved to
        self.prints = prints                 #if true, the agent will print his scores
 
        self.model = self.build_model()
        self.build_model()


    def build_model(self):

        self.hidden_1 = nn.Linear(self.state_size, 24)
        self.hidden_2 = nn.Linear(24, 48)
        self.hidden_3 = nn.Linear(48, self.action_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.alpha, weight_decay = self.alpha_decay)
        self.loss_fn = nn.MSELoss()

    def forward(self, input):

        x = nn.Tanh(self.hidden_1(input))
        x = nn.Tanh(self.hidden_2(x))
        x = nn.ReLU(self.hidden_3(x))

    def act(self, state):

        if (np.random.random() <= self.epsilon):
            return self.env.action_space.sample()
        return np.argmax(self.forward(state))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(state, action, reward, next_state, done)

    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            y_target = self.model(state)
            y_target[0][action] = reward if done else reward + self.gamma * np.max(self.model(next_state)[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

            loss = self.loss_fn(y_target, y_batch)
            loss.backward()
            self.optimizer.step()
            
        # self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




model = DQNAgent("Taxi-v3", "./", 40000, 100, 2, 0.001)
model.build_model()

model.replay(10)
