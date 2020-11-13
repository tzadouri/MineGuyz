try:
    from malmo import MalmoPython
except:
    import MalmoPython

import os
import sys
import time
import json
import random
from tqdm import tqdm
from collections import deque, namedtuple
import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import randint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math

Transition = namedtuple('Transition', ('state, action, nex_state', 'reward'))
LOSS = []
REWARD = []

# Hyperparameters
class Hyperparameters:
    SIZE = 50
    REWARD_DENSITY = .1
    PENALTY_DENSITY = .02
    OBS_SIZE = 5
    MAX_EPISODE_STEPS = 100
    MAX_GLOBAL_STEPS = 10000
    REPLAY_BUFFER_SIZE = 10000
    EPSILON_DECAY = .999
    MIN_EPSILON = .1
    BATCH_SIZE = 128
    GAMMA = .9
    TARGET_UPDATE = 10
    LEARNING_RATE = 1e-4
    START_TRAINING = 500
    LEARN_FREQUENCY = 1
    ACTION_DICT = {
        0: 'move 1', 
        1: 'turn 1',  
        2: 'turn -1',  
        3: 'attack 1' 
    }

#Ignore for now
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self,*args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class QNetwork(nn.Module):
    
    def __init__(self, obs_size, action_size, hidden_size=128):
        super().__init__()

        self.cnn = nn.Sequential(
                nn.Conv2d(obs_size[0], hidden_size, kernel_size=3,padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3,padding=1),
        ) 

        self.linear = nn.Linear(obs_size[1] * obs_size[2] * hidden_size, action_size)
        
        
    def forward(self, obs):
        batch_size = obs.shape[0]
        obs = self.cnn(obs)
        obs_flat = obs.view(batch_size, -1)
        return self.linear(obs_flat)



