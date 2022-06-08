import sys
sys.path.append('./common/')

import numpy as np
import torch
from gym.spaces import Box, Discrete
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from common_utils import np_to_tensor


'''
    Reference: https://github.com/UBCMOCCA/SymmetricRL/blob/master/algorithms/storage.py
'''


'''
    Base class
'''
class BaseBuffer(object):
    def __init__(self, obs_space, act_space, n_cpus):
        self.obs_shape = obs_space.shape if isinstance(obs_space, Box) else [obs_space.n]
        self.act_shape = act_space.shape if isinstance(act_space, Box) else [act_space.n]

        self.n_cpus = n_cpus

        # print(self.obs_shape, self.act_shape)


    def add(self):
        raise NotImplementedError


# '''
#     Replay buffer for off-policy algorithms
# '''
# class ReplayBuffer(BaseBuffer):
#     def __init__(self, obs_space, act_space, n_cpus, capacity = 1000000):
#         super().__init__(obs_space, act_space, n_cpus)

#         self.capacity = capacity
#         self.obs_buffer = np.zeros((self.capacity, *self.obs_shape))
#         self.act_buffer = np.zeros((self.capacity, *self.act_shape))
#         self.reward_buffer = np.zeros((self.capacity))
#         self.done_buffer = np.zeros((self.capacity))

#         self.pos = 0



'''
    Rollout buffer for PPO
'''
class RolloutBuffer(BaseBuffer):
    def __init__(self, obs_space, act_space, n_cpus, n_episode_steps):
        super().__init__(obs_space, act_space, n_cpus)

        self.n_episode_steps = n_episode_steps
        self.pos = 0

        self.obs_shape = (47,)
        self.obs_buffer = torch.zeros((self.n_episode_steps, self.n_cpus, *self.obs_shape))
        self.act_buffer = torch.zeros((self.n_episode_steps, self.n_cpus, *self.act_shape))
        self.reward_buffer = torch.zeros((self.n_episode_steps, self.n_cpus))
        self.done_buffer = torch.zeros((self.n_episode_steps, self.n_cpus))
        self.value_pred_buffer = torch.zeros((self.n_episode_steps, self.n_cpus))
        self.action_log_prob_buffer = torch.zeros((self.n_episode_steps, self.n_cpus))
        self.return_buffer = torch.zeros((self.n_episode_steps, self.n_cpus))


    def to_device(self, device):
        self.obs_buffer = self.obs_buffer.to(device)
        self.act_buffer = self.act_buffer.to(device)
        self.reward_buffer = self.reward_buffer.to(device)
        self.done_buffer = self.done_buffer.to(device)
        self.value_pred_buffer = self.value_pred_buffer.to(device)
        self.action_log_prob_buffer = self.action_log_prob_buffer.to(device)
        self.return_buffer = self.return_buffer.to(device)


    def add(self, 
        obs : torch.Tensor, 
        act : torch.Tensor, 
        reward : torch.Tensor, 
        done : torch.Tensor,
        value_pred : torch.Tensor,
        action_log_prob : torch.Tensor
    ):
        self.obs_buffer[self.pos].copy_(obs)
        self.act_buffer[self.pos].copy_(act)
        self.reward_buffer[self.pos].copy_(reward)
        self.done_buffer[self.pos].copy_(done)
        self.value_pred_buffer[self.pos].copy_(value_pred)
        self.action_log_prob_buffer[self.pos].copy_(action_log_prob)

        self.pos = (self.pos + 1) % self.n_episode_steps


    # def add_np(self,
    #     obs : np.ndarray, 
    #     act : np.ndarray, 
    #     reward : np.ndarray,  
    #     # done : np.ndarray, 
    #     value_pred : np.ndarray,
    #     action_log_prob : np.ndarray, 
    # ):
    #     self.obs_buffer[self.pos].copy_(np_to_tensor(obs))
    #     self.act_buffer[self.pos].copy_(np_to_tensor(act))
    #     self.reward_buffer[self.pos].copy_(np_to_tensor(reward))
    #     # self.done_buffer[self.pos].copy_(np_to_tensor(done))
    #     self.value_pred_buffer[self.pos].copy_(np_to_tensor(value_pred))
    #     self.action_log_prob_buffer[self.pos].copy_(np_to_tensor(action_log_prob))

    #     self.pos += 1


    def reset(self):
        self.pos = 0



    # def _compute_return(self, final_return : torch.Tensor, gamma = 0.99):
    #     curr_return = final_return
    #     for i in reversed(range(self.pos)):
    #         curr_return = self.reward_buffer[i] + gamma * curr_return
    #         self.return_buffer[i].copy_(curr_return)


    def compute_return(self, final_V : torch.Tensor, gamma = 0.99):
        curr_return = final_V
        for i in reversed(range(self.n_episode_steps)):
            # curr_return = self.done_buffer[i] * self.value_pred_buffer[i] + (1 - self.done_buffer[i]) * (self.reward_buffer[i] + gamma * curr_return)
            curr_return = (self.reward_buffer[i] + gamma * curr_return) * (1 - self.done_buffer[i])
            self.return_buffer[i].copy_(curr_return)


    def generate(self, mini_batch_size = 128):
        n_total_samples = self.n_episode_steps * self.n_cpus
        sampler = BatchSampler(SubsetRandomSampler(range(n_total_samples)), mini_batch_size, drop_last = False)

        tmp_obs_buffer = self.obs_buffer.view(-1, *self.obs_shape).float()
        tmp_act_buffer = self.act_buffer.view(-1, *self.act_shape)
        tmp_value_pred_buffer = self.value_pred_buffer.view(-1)
        tmp_old_action_log_prob_buffer = self.action_log_prob_buffer.view(-1)
        tmp_return_buffer = self.return_buffer.view(-1)

        for indices in sampler:
            obs_batch = tmp_obs_buffer[indices]
            act_batch = tmp_act_buffer[indices]
            value_pred_batch = tmp_value_pred_buffer[indices]
            old_action_log_prob_batch = tmp_old_action_log_prob_buffer[indices]
            return_batch = tmp_return_buffer[indices]

            yield obs_batch, act_batch, value_pred_batch, old_action_log_prob_batch, return_batch



# if __name__ == '__main__':
#     from gym.spaces import Box
#     space = Box(-1, 1, (3,))
#     buffer = RolloutBuffer(space, space, 2, 10)

#     buffer.reward_buffer.fill_(1.0)
#     buffer.value_pred_buffer.fill_(2.0)

#     buffer.done_buffer[2][0] = 1
#     buffer.done_buffer[-1][0] = 1
#     buffer.done_buffer[7][1] = 1
#     buffer.done_buffer[3][1] = 1

#     buffer.compute_return(torch.tensor([3.0, 0.0]), gamma = 1)
#     print(buffer.reward_buffer)
#     print(buffer.done_buffer)
#     print(buffer.return_buffer)