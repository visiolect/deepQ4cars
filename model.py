#  libraries for self-driving-car
import numpy as np
import os
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# architecture of NN
class Network(nn.Module):
    def __init__(self, input_size, actions, h1=30):
        super(Network, self).__init__()
        self.x = input_size
        self.y = actions
        self.x_to_h1 = nn.Linear(input_size, h1)
        self.h1_to_y = nn.Linear(h1, actions)

    def forward(self, state):
        x = F.relu(self.x_to_h1(state))
        q = self.h1_to_y(x)
        return q


# experience replay implementation
class ReplayMemory:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []

    def add_memory(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:  # making sure that memory stays in bound of capacity
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x)), samples)


# deep q-learning implementation
class DeepQ:
    def __init__(self, input_size, actions, discount):
        self.discount = discount
        self.reward_window = []
        self.model = Network(input_size, actions)
        self.memory = ReplayMemory()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.previous_state = torch.Tensor(input_size).unsqueeze(0)
        self.previous_action = 0
        self.previous_reward = 0

    def select_action(self, state):
        with torch.no_grad():
            probs = F.softmax(self.model(Variable(state)) * 10)
            action = probs.multinomial(1)[0]
            return action

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        outputs = self.model(batch_state).gather(
            1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.discount*next_outputs + batch_reward
        loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

    def update(self, last_reward, last_signal):
        new_state = torch.Tensor(last_signal).float().unsqueeze(0)
        self.memory.add_memory((self.previous_state,
                                new_state,
                                torch.LongTensor([int(self.previous_action)]),
                                torch.Tensor([self.previous_reward]))
                               )
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(
                100)
            self.learn(batch_state, batch_next_state,
                       batch_action, batch_reward)
        self.previous_action = action
        self.previous_state = new_state
        self.previous_reward = last_reward
        self.reward_window.append(last_reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)

    def save(self):
        torch.save(
            {'state_dict': self.model.state_dict(),
             'optimizer': self.optimizer.state_dict(),
             }, 'model.pytorch'
        )

    def load(self):
        if os.path.isfile('model.pytorch'):
            print('-> loading model...')
            model = torch.load('model.pytorch')
            self.model.load_state_dict(model['state_dict'])
            self.optimizer.load_state_dict(model['optimizer'])
            print('Done.')
        else:
            print('No previous model found to load.')
