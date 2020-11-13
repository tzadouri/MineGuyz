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

'''
Transition = namedtuple('Transition', ('state, action, nex_state', 'reward'))
LOSS = []
REWARD = []
'''

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
        1: 'turn 0.15',  
        2: 'turn -0.15',  
        3: 'attack 1' 
    }


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


def get_action(obs, q_network, epsilon, allow_break_action):
    
    obs_torch = torch.tensor(obs.copy(), dtype=torch.float).unsqueeze(0)
    action_values = q_network(obs_torch)
    if random.random() > epsilon:
        with torch.no_grad(): 
            if not allow_break_action:
                action_values[0, 3] = -float('inf')  
            action_idx = torch.argmax(action_values).item()
    else: 
        if not allow_break_action:
            action_idx = random.randint(0,Hyperparameters.OBS_SIZE -3) 
        else: 
            action_idx = random.randint(0,Hyperparameters.OBS_SIZE -2 ) 
    
    return action_idx

'''    
def get_observation(world_state):
    
    obs = np.zeros((2, Hyperparameters.OBS_SIZE, Hyperparameters.OBS_SIZE))
    while world_state.is_mission_running:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        if len(world_state.errors) > 0:
            raise AssertionError('Could not load grid.')

        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            observations = json.loads(msg)

            grid = observations['floorAll']
            grid_binary = [1 if x == 'diamond_ore' or x == 'lava' else 0 for x in grid]
            obs = np.reshape(grid_binary, (2, Hyperparameters.OBS_SIZE, Hyperparameters.OBS_SIZE))

            yaw = observations['Yaw']
            if yaw == 270:
                obs = np.rot90(obs, k=1, axes=(1, 2))
            elif yaw == 0:
                obs = np.rot90(obs, k=2, axes=(1, 2))
            elif yaw == 90:
                obs = np.rot90(obs, k=3, axes=(1, 2))
            
            break

    return obs

'''

def prepare_batch(replay_buffer):

    batch_data = random.sample(replay_buffer, Hyperparameters.BATCH_SIZE)
    obs = torch.tensor([x[0] for x in batch_data], dtype=torch.float)
    action = torch.tensor([x[1] for x in batch_data], dtype=torch.long)
    next_obs = torch.tensor([x[2] for x in batch_data], dtype=torch.float)
    reward = torch.tensor([x[3] for x in batch_data], dtype=torch.float)
    done = torch.tensor([x[4] for x in batch_data], dtype=torch.float)
    
    return obs, action, next_obs, reward, done
  

def learn(batch, optim, q_network, target_network):
    obs, action, next_obs, reward, done = batch

    optim.zero_grad()
    values = q_network(obs).gather(1, action.unsqueeze(-1)).squeeze(-1)
    target = torch.max(target_network(next_obs), 1)[0]
    target = reward + Hyperparameters.GAMMA * target * (1 - done)
    loss = torch.mean((target - values) ** 2)
    loss.backward()
    optim.step()

    return loss.item()

def log_returns(steps, returns):
    box = np.ones(10) / 10
    returns_smooth = np.convolve(returns, box, mode='same')
    plt.clf()
    plt.plot(steps, returns_smooth)
    plt.title('Diamond Collector')
    plt.ylabel('Return')
    plt.xlabel('Steps')
    plt.savefig('returns.png')

    with open('returns.txt', 'w') as f:
        for value in returns:
            f.write("{}\n".format(value)) 








