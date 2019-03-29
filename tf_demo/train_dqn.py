"""
The expert recorder.
"""
import argparse
import random
import gym
import numpy as np
import time
import os

import tensorflow as tf

import baselines.common.tf_util as U
from baselines.common.tf_util import load_variables, save_variables
from baselines.common.schedules import LinearSchedule
from baselines.common import set_global_seeds

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput

from baselines.common.tf_util import get_session
from baselines.deepq.models import build_q_func

SHARD_SIZE = 2000

def get_options():
    parser = argparse.ArgumentParser(description='Trains DQN..')
    parser.add_argument('model_weights', type=str, help='Path to weights to load')

    args = parser.parse_args()

    return args


def train_dqn(opts,
              seed=None,
              lr=5e-4,
              total_timesteps=5000000,
              buffer_size=10000,
              exploration_fraction=0.1,
              exploration_final_eps=0.02,
              train_freq=4,
              batch_size=64,
              checkpoint_freq=10000,
              learning_starts=1000,
              gamma=0.995,
              target_network_update_freq=500,
              load_path=None):
    """
    Runs the main recorder by binding certain discrete actions to keys.
    """
    #env = gym.make('MountainCar-v0')
    env = gym.make('LunarLander-v2')
    env._max_episode_steps = 1200

    sess = get_session()
    set_global_seeds(seed)

    q_func = build_q_func('mlp')

    # capture the shape outside the closure so that the env object is not serialized
    # by cloudpickle when serializing make_obs_ph

    observation_space = env.observation_space
    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = deepq.build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10
    )
    replay_buffer = ReplayBuffer(buffer_size)

    # Create the schedule for exploration starting from 1.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    # Initialize the parameters and copy them to the target network.
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    obs = env.reset()

    for t in range(total_timesteps):
        # Take action and update exploration to the newest value
        env.render()
        update_eps = exploration.value(t)
        action = act(np.array(obs)[None], update_eps=update_eps)[0]
        new_obs, rew, done, _ = env.step(action)
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs

        episode_rewards[-1] += rew
        if done:
            print("Exploration value: {}".format(exploration.value(t)))
            print(episode_rewards)
            obs = env.reset()
            episode_rewards.append(0.0)

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None
            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
            
        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target()

        if t > learning_starts and t % checkpoint_freq == 0:
            save_variables(opts.model_weights)

if __name__ == "__main__":
    opts = get_options()
    train_dqn(opts)
