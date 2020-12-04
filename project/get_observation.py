try:
    from malmo import MalmoPython
except:
    import MalmoPython
from RL_DQN import QNetwork, Hyperparameters, get_action, prepare_batch, learn, log_returns
import numpy as np
import time
import json

def get_observation(world_state, agent_host):
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

