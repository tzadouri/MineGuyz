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
            grid_binary = [1 if x == 'glass' or x == 'gold_block'  or x == 'emerald_block' else -1 if x=='redstone_block' else 0 for x in grid]
            obs = np.reshape(grid_binary, (2, Hyperparameters.OBS_SIZE, Hyperparameters.OBS_SIZE))
            
            break

    return obs


def init_malmo(agent_host):
    """
    Initialize new malmo mission.
    """
    my_mission = MalmoPython.MissionSpec(GetMissionXML(Hyperparameters.SIZE, Hyperparameters.OBS_SIZE, Hyperparameters.MAX_EPISODE_STEPS), True)
    my_mission_record = MalmoPython.MissionRecordSpec()
    my_mission.requestVideo(800, 500)
    my_mission.setViewpoint(1)

    max_retries = 3
    my_clients = MalmoPython.ClientPool()
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10000)) # add Minecraft machines here as available
    my_clients.add(MalmoPython.ClientInfo('127.0.0.1', 10001))
    for retry in range(max_retries):
        try:
            agent_host.startMission( my_mission, my_clients, my_mission_record, 0, "MineGuyz" )
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:", e)
                exit(1)
            else:
                time.sleep(2)

    return agent_host


def main(agent_host):
    
    # Init networks
    q_network = QNetwork((2, Hyperparameters.OBS_SIZE, Hyperparameters.OBS_SIZE), len(Hyperparameters.ACTION_DICT))
    target_network = QNetwork((2, Hyperparameters.OBS_SIZE, Hyperparameters.OBS_SIZE), len(Hyperparameters.ACTION_DICT))
    target_network.load_state_dict(q_network.state_dict())

    # Init optimizer
    optim = torch.optim.Adam(q_network.parameters(), lr= Hyperparameters.LEARNING_RATE)

    # Init replay buffer
    replay_buffer = deque(maxlen=Hyperparameters.REPLAY_BUFFER_SIZE)

    # Init vars
    global_step = 0
    num_episode = 0
    epsilon = 1
    start_time = time.time()
    returns = []
    steps = []
    loss_array = []

    # Begin main loop
    loop = tqdm(total=Hyperparameters.MAX_GLOBAL_STEPS, position=0, leave=False)
    while global_step < Hyperparameters.MAX_GLOBAL_STEPS:
        episode_step = 0
        episode_return = 0
        episode_loss = 0
        done = False

        # Setup Malmo
        agent_host = init_malmo(agent_host)
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("\nError:",error.text)
        obs = get_observation(world_state)

        # Run episode
        while world_state.is_mission_running:
            # Get action
            action_idx = get_action(obs, q_network, epsilon)
            command = Hyperparameters.ACTION_DICT[action_idx]

            # Take step
            agent_host.sendCommand(command)

            # If your agent isn't registering reward you may need to increase this
            time.sleep(.1)

            # We have to manually calculate terminal state to give malmo time to register the end of the mission
            # If you see "commands connection is not open. Is the mission running?" you may need to increase this
            episode_step += 1
            if episode_step >= Hyperparameters.MAX_EPISODE_STEPS or \
                    (obs[0, int(Hyperparameters.OBS_SIZE/2)+1, int(Hyperparameters.OBS_SIZE/2)] == -1 and \
                    command == 'movesouth 1'):
                done = True
                time.sleep(2)  

            # Get next observation
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:", error.text)
            next_obs = get_observation(world_state) 

            # Get reward
            reward = 0
            for r in world_state.rewards:
                reward += r.getValue()
            episode_return += reward

            # Store step in replay buffer
            replay_buffer.append((obs, action_idx, next_obs, reward, done))
            obs = next_obs

            # Learn
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
