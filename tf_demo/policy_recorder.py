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

import tensorflow as tf

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

SHARD_SIZE = 10000

def get_options():
    parser = argparse.ArgumentParser(description='Records an expert..')
    parser.add_argument('data_directory', type=str,
        help="The main datastore for this particular expert.")
    parser.add_argument('model_name', type=str, help='String representing the model')
    parser.add_argument('model_weights', type=str, help='Path to weights to load')

    args = parser.parse_args()

    return args

def init_model(opts, env):
    sess = get_session()

    q_func = build_q_func('mlp')

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act = deepq.build_act(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n
    )

    load_variables(opts.model_weights)
    return act

def run_recorder(opts):
    """
    Runs the main recorder by binding certain discrete actions to keys.
    """
    ddir = opts.data_directory

    record_history = [] # The state action history buffer.

    #env = gym.make('MountainCar-v0')
    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 1200

    act = init_model(opts, env)

    shard_suffix = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
    sarsa_pairs = []

    print("Welcome to the expert recorder")
    print("To record press either a or d to move the agent left or right.")
    print("Once you're finished press + to save the data.")
    print("NOTE: Make sure you've selected the console window in order for the application to receive your input.")

    num_episodes = 0
    num_shards = 0

    for _ in range(100000):
        done = False
        obs = _last_obs = env.reset()
        rewards = 0
        while not done:
            env.render()

            action = act(np.array(obs)[None])[0]
            obs, reward, done, info = env.step(action)
            rewards += reward
            
            sarsa = (_last_obs, action, reward, done)
            _last_obs = obs
            sarsa_pairs.append(sarsa)
        num_episodes += 1
        print("Episode num: {}".format(num_episodes))
        print("Rewards for episode: {}".format(rewards))

        if len(sarsa_pairs) > SHARD_SIZE:
            shard = sarsa_pairs[:SHARD_SIZE]
            sarsa_pairs = sarsa_pairs[SHARD_SIZE:]
            shard_name = "{}_{}.npy".format(str(num_shards), shard_suffix)
            if not os.path.exists(ddir):
                os.makedirs(ddir)
            with open(os.path.join(ddir, shard_name), 'wb') as f:
                np.save(f, shard)
            num_shards += 1
            print("Saved shard number {}".format(num_shards))
    
if __name__ == "__main__":
    opts = get_options()
    run_recorder(opts)
