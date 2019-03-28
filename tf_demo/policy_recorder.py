"""
The expert recorder.
"""
import argparse
import getch
from getkey import getkey
import random
import gym
import numpy as np
import time
import os

SHARD_SIZE = 2000

def get_options():
    parser = argparse.ArgumentParser(description='Records an expert..')
    parser.add_argument('data_directory', type=str,
        help="The main datastore for this particular expert.")
    parser.add_argument('model_name', type=str, help='String representing the model')
    parser.add_argument('model_weights', type=str, help='Path to weights to load')

    args = parser.parse_args()

    return args


def run_recorder(opts):
    """
    Runs the main recorder by binding certain discrete actions to keys.
    """
    ddir = opts.data_directory

    record_history = [] # The state action history buffer.

    #env = gym.make('MountainCar-v0')
    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 1200

    shard_suffix = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    sarsa_pairs = []

    print("Welcome to the expert recorder")
    print("To record press either a or d to move the agent left or right.")
    print("Once you're finished press + to save the data.")
    print("NOTE: Make sure you've selected the console window in order for the application to receive your input.")

    for _ in range(1000):
        done = False
        _last_obs = env.reset()
        while not done:
            env.render()

            obs, reward, done, info = env.step(action)
            print(reward)
            
            no_action = False
            sarsa = (_last_obs, action, reward, done)
            _last_obs = obs
            sarsa_pairs.append(sarsa)

    print("SAVING")
    # Save out recording data.
    num_shards = int(np.ceil(len(sarsa_pairs)/SHARD_SIZE))
    for shard_iter in range(num_shards):
        shard = sarsa_pairs[
            shard_iter*SHARD_SIZE: min(
                (shard_iter+1)*SHARD_SIZE, len(sarsa_pairs))]

        shard_name = "{}_{}.npy".format(str(shard_iter), shard_suffix)
        if not os.path.exists(ddir):
            os.makedirs(ddir)
        with open(os.path.join(ddir, shard_name), 'wb') as f:
            np.save(f, sarsa_pairs)

if __name__ == "__main__":
    opts = get_options()
    run_recorder(opts)
