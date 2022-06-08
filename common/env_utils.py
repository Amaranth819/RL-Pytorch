import gym
import pybulletgym
import numpy as np
from gym.core import Wrapper
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

def make_singleprocessing_env(env_id):
    return gym.make(env_id)


def make_multiprocessing_env(env_id, n_cpus):
    return gym.vector.make(env_id, num_envs = n_cpus)


def get_space_dim(space):
    if type(space) == Box:
        return space.shape[0]
    elif type(space) == Discrete:
        return space.n
    else:
        raise TypeError


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

    epi_rewards_mean, epi_rewards_std = np.mean(epi_rewards), np.std(epi_rewards)
    epi_steps_mean, epi_steps_std = np.mean(epi_steps), np.std(epi_steps)
    return (epi_rewards_mean, epi_rewards_std), (epi_steps_mean, epi_steps_std)



# if __name__ == '__main__':
#     # https://tristandeleu.github.io/gym/vector/
#     # env = gym.vector.make('HumanoidPyBulletEnv-v0', num_envs = 5)
#     env = gym.vector.make('CartPole-v1', num_envs = 5)
#     print(env.action_space)
#     obs = env.reset()
#     actions = np.zeros(5, dtype = int)
#     obs, rewards, dones, infos = env.step(actions)
#     print(obs.shape)