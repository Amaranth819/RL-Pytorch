import numpy as np
import os
import time
import gym
import pybulletgym
import pybullet as pb
from algo.ppo import PPO
from net.network import MlpActor, MlpCritic
from common.env_utils import make_multiprocessing_env, get_space_dim


def evaluate_policy(env, policy = None, eval_freq = 50, sample_action = False):
    epi_rewards = []
    epi_steps = []

    for _ in range(eval_freq):
        obs = env.reset()
        done = False
        curr_s = 0
        curr_r = 0

        while not done:
            if sample_action:
                action = env.action_space.sample()
            else:
                action = policy.predict(obs, deterministic = True)
            obs, r, done, info = env.step(action)
            curr_s += 1
            curr_r += r

        epi_rewards.append(curr_r)
        epi_steps.append(curr_s)

    epi_rewards_mean, epi_rewards_std = np.mean(epi_rewards), np.std(epi_steps)
    epi_steps_mean, epi_steps_std = np.mean(epi_steps), np.std(epi_steps)
    return (epi_rewards_mean, epi_rewards_std), (epi_steps_mean, epi_steps_std)


class Config(object):
    def __init__(self) -> None:
        self.env_name = 'HopperPyBulletEnv-v0'
        self.n_cpus = 4
        self.lr = 0.0001
        self.gamma = 0.99
        self.batch_size = 128
        self.device = 'auto'
        self.epochs = 100000
        self.log_path = './PPO/log/'
        self.model_path = './exp/HopperPyBulletEnv-v0/best/' # './PPO/best/'


if __name__ == '__main__':
    config = Config()

    env = gym.make(config.env_name)
    obs_dim = get_space_dim(env.observation_space)
    act_dim = get_space_dim(env.action_space)

    actor = MlpActor(obs_dim, act_dim)
    critic = MlpCritic(obs_dim)

    model = PPO(
        env_name = config.env_name,
        actor = actor,
        critic = critic,
        # lr = config.lr,
        gamma = config.gamma,
        batch_size = config.batch_size,
        device = 'auto'
    )
    model.load_actor(os.path.join(config.model_path, 'actor.pkl'))
    model.load_critic(os.path.join(config.model_path, 'critic.pkl'))
    
    # print('Random')
    # (epi_r_mean, epi_r_std), (epi_s_mean, epi_s_std) = evaluate_policy(env, sample_action = True)
    # print('Episode Rewards: {:.2f} +- {:.2f} | Episode Steps: {:.2f} +- {:.2f}'.format(epi_r_mean, epi_r_std, epi_s_mean, epi_s_std))
    # print('After training')
    # (epi_r_mean, epi_r_std), (epi_s_mean, epi_s_std) = evaluate_policy(env, policy = model, sample_action = False)
    # print('Episode Rewards: {:.2f} +- {:.2f} | Episode Steps: {:.2f} +- {:.2f}'.format(epi_r_mean, epi_r_std, epi_s_mean, epi_s_std))

    epi_rewards, epi_steps = [], []
    env.render()
    for _ in range(10):
        obs = env.reset()
        done = False
        epi_reward, epi_step = 0, 0

        while not done:
            action = model.predict(obs, deterministic = True)
            # action = env.action_space.sample()
            obs, r, done, info = env.step(action)
            env.render()
            epi_step += 1
            epi_reward += r
            time.sleep(0.01)

        print(epi_reward, epi_step)
        epi_rewards.append(epi_reward)
        epi_steps.append(epi_step)

    print('{:.2f} +- {:.2f} | {:.2f} +- {:.2f}'.format(
        np.mean(epi_rewards),
        np.std(epi_rewards),
        np.mean(epi_steps),
        np.std(epi_steps)
    ))