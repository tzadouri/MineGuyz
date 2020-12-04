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
from collections import deque
import matplotlib.pyplot as plt 
import numpy as np
from numpy.random import randint
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from map_generator import GetMissionXML
from RL_DQN import QNetwork, Hyperparameters, get_action, prepare_batch, learn, log_returns
from get_observation import get_observation
from init_malmo import init_malmo

def main(agent_host):
    
    q_network = QNetwork((2, Hyperparameters.OBS_SIZE, Hyperparameters.OBS_SIZE), len(Hyperparameters.ACTION_DICT))
    target_network = QNetwork((2, Hyperparameters.OBS_SIZE, Hyperparameters.OBS_SIZE), len(Hyperparameters.ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    optim = torch.optim.Adam(q_network.parameters(), lr= Hyperparameters.LEARNING_RATE)

    replay_buffer = deque(maxlen=Hyperparameters.REPLAY_BUFFER_SIZE)

    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []
    loss_array = []

    loop = tqdm(total=Hyperparameters.MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < Hyperparameters.MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False

        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs = get_observation(world_state, agent_host)

        while world_state.is_mission_running:
            action_idx = get_action(obs, q_network, epsilon)
            command = Hyperparameters.ACTION_DICT[action_idx]

            agent_host.sendCommand(command)

            time.sleep(.1)

            episode_step += 1
            if episode_step >= Hyperparameters.MAX_EPISODE_STEPS or \
                    (obs[0, int(Hyperparameters.OBS_SIZE/2)+1, int(Hyperparameters.OBS_SIZE/2)] == -1 and \
                    command == 'movesouth 1'):
                done = True
                time.sleep(2)  

            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs = get_observation(world_state, agent_host) 

            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            episode_return += reward

            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            global_step += 1
            if global_step > Hyperparameters.START_TRAINING and global_step % Hyperparameters.LEARN_FREQUENCY == 0:
                batch = prepare_batch(replay_buffer)
                loss = learn(batch, optim, q_network, target_network)
                episode_loss += loss

                if epsilon > Hyperparameters.MIN_EPSILON:
                    epsilon *= Hyperparameters.EPSILON_DECAY

                if global_step % Hyperparameters.TARGET_UPDATE == 0:
                    target_network.load_state_dict(q_network.state_dict())

        num_episode += 1
        returns.append(episode_return)
        loss_array.append(episode_loss)
        steps.append(global_step)
        avg_return = sum(returns[-min(len(returns), 10):]) / min(len(returns), 10)
        loop.update(episode_step)
        loop.set_description('Episode: {} Steps: {} Time: {:.2f} Loss: {:.2f} Last Return: {:.2f} Avg Return: {:.2f}'.format(
            num_episode, global_step, (time.time() - start_time) / 60, episode_loss, episode_return, avg_return))

        if num_episode > 0 and num_episode % 10 == 0:
            log_returns(steps, loss_array)
            print()

if __name__ == '__main__':
    
    agent_host = MalmoPython.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:', e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)

    main(agent_host)

