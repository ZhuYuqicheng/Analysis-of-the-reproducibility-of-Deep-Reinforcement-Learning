import os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch as th
from torch import nn
import scipy.stats as stats

import gym
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

from EvaluationClasses import SaveOnBestTrainingRewardCallback
from ModificationClasses import ActivationActorCriticPolicy

def SaveGIF(env_id, length, model, algo):
    # save GIF
    gif_path = f"record/gif/{algo}_{env_id}.gif"

    images = []
    obs = model.env.reset()
    img = model.env.render(mode='rgb_array')
    for i in range(length):
        images.append(img)
        action, _ = model.predict(obs)
        obs, _, _ ,_ = model.env.step(action)
        img = model.env.render(mode='rgb_array')
    imageio.mimsave(gif_path, [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)

def SaveVideo(env_id, video_length, model, algo):
    video_folder = 'record/videos/'
    env = DummyVecEnv([lambda: gym.make(env_id)])
    obs = env.reset()
    # Record the video starting at the first step
    env = VecVideoRecorder(env, video_folder, \
        record_video_trigger=lambda x: x == 0, \
            video_length=video_length, name_prefix=f"{algo}_{env_id}")
    env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _ = env.step(action)
    # Save the video
    env.close()

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def plot_results(log_folder, title):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # Create log dir
    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)

    # create a gym environment, CartPole-v1 (5e5), HalfCheetah-v2 (2e6)
    env_id = "HalfCheetah-v2"
    algo = "PPO"
    timesteps = 2e6
    episode = 5

    # create environment
    env = gym.make(env_id)
    rewards = []
    min_len = timesteps

    opt_str = ['Adam', 'RMSprop', 'SGD']
    optimizers = [th.optim.Adam, th.optim.RMSprop, th.optim.SGD]

    for index, optimizer in enumerate(optimizers):
        for _ in range(episode):
            # reset the environment
            env.reset()
            # record the reward for every update
            env = Monitor(env, log_dir)

            # model definition
            # modification of optimizer (th.optim.Adam, th.optim.RMSprop, th.optim.SGD)
            model = PPO("MlpPolicy", env, verbose=1, policy_kwargs={'optimizer_class': optimizer})
            # modification of activation function (nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.LeakyReLU())
            # model = PPO(ActivationActorCriticPolicy, env, verbose=1)

            # model training
            model.learn(total_timesteps=timesteps)

            # save the reward result in DataFrame
            log = load_results(log_dir)
            rewards.append(log['r'])
            os.remove(f"{log_dir}monitor.csv")
            # in order to keep the same column length
            if len(log['r']) < min_len:
                min_len = len(log['r'])

        # merge the results
        reward_df = pd.DataFrame()
        for i in range(episode):         
            reward_df = pd.concat([reward_df, rewards[i][:min_len]], axis=1, ignore_index=True)

        # save the results as pickle
        pkl_file = f"./logs/{algo}_{env_id}_{opt_str[index]}.pkl"
        reward_df.to_pickle(pkl_file)
    
    # # save section
    # SaveGIF(env_id, 350, model, algo)
    # SaveVideo(env_id, 350, model, algo)