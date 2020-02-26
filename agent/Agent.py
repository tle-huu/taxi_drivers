import numpy as np
import torch
from agent.Memory import ReplayBuffer
from agent.DRL import CarLeader

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, number_of_cars = 1):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.action_space = list(range(n_actions))
        self.learn_step_counter = 0
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.number_of_cars = number_of_cars

        self.memory = ReplayBuffer(mem_size, input_dims, self.number_of_cars)

    def store_transition(self, state, action, reward, state_, done):

        self.memory.store_transition(state, action, reward, state_, done)

    def choose_action(self, observation):
        raise NotImplementedError

    def replace_target_network(self):

        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):

        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min
    def sample_memory(self):
        state, action, reward, new_state, terminals = \
                                self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(state).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        dones = torch.tensor(terminals).to(self.q_eval.device)
        actions = torch.LongTensor(action).to(self.q_eval.device)
        states_ = torch.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def learn(self):
        raise NotImplementedError


class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQNAgent, self).__init__(*args, **kwargs)

        self.q_eval = CarLeader(self.input_dims, self.number_of_cars, self.lr)
        self.q_next = CarLeader(self.input_dims, self.number_of_cars, self.lr)
    
    
    def save_checkpoint(self, path):
        params= {"gamma" : self.gamma,
                 "epsilon" : self.epsilon,
                 "learn_step_counter" : self.learn_step_counter
                }
        weights = {"q_eval" : self.q_eval.state_dict(), 
                   "q_next" : self.q_next.state_dict()}
        torch.save({"params": params, "weights" : weights}, path)
        return
    
    def load_checkpoint(self, checkpoint):
        self.gamma = checkpoint["params"]["gamma"]
        self.epsilon = checkpoint["params"]["epsilon"]
        self.learn_step_counter = checkpoint["params"]["learn_step_counter"]
        self.q_eval.load_state_dict(checkpoint["weights"]["q_eval"])
        self.q_next.load_state_dict(checkpoint["weights"]["q_next"])
    
    def choose_action(self, observation):
      
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation],dtype=torch.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions, dim = 2).squeeze(0).cpu()
        else:
            action = torch.LongTensor([np.random.choice(self.action_space) for _ in range(self.number_of_cars)])

        return action

    def learn(self):

        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()


        device = self.q_eval.device

        ## Getting Q value of current state
        states_to_device = states.to(device)
        forward_states = self.q_eval.forward(states_to_device)

        gather_forward_states = torch.gather(forward_states, 2, actions.unsqueeze(-1))
        q_pred = gather_forward_states

        ## Getting Q value of target
        next_states_to_device = next_states.to(device)
        forward_next_states = self.q_next.forward(next_states_to_device)

        q_next = forward_next_states.max(dim = 2)[0]
        q_next[dones] = 0.0

        q_target = rewards.unsqueeze(-1) + self.gamma * q_next

        q_target = q_target.squeeze()
        q_pred = q_pred.squeeze()

        # q_target = q_target.sum(dim = 1)
        # q_pred = q_pred.sum(dim = 1)

        loss = self.q_eval.loss(q_target, q_pred).to(device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.q_eval.scheduler.step(loss)

        self.decrement_epsilon()

        return float(loss)
