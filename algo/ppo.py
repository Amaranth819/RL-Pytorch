import torch
import torch.nn as nn
import numpy as np
import os
from collections import defaultdict
from torch.distributions.normal import Normal
from common.common_utils import get_device, np_to_tensor, tensor_to_np, linear_lr_decay, set_optimizer_lr
from common.log_utils import Logger
from common.env_utils import evaluate_policy
from common.data_memory import RolloutBuffer
from net.network import BaseActor, BaseCritic
from common.env_utils import make_multiprocessing_env, get_space_dim, make_singleprocessing_env


class PPO(object):
    def __init__(self, 
        env_name, 
        actor : BaseActor, 
        critic : BaseCritic, 
        actor_lr = 0.0001, 
        critic_lr = 0.001, 
        gamma = 0.99,
        batch_size = 128,
        n_cpus = 4,
        device = 'auto', 
    ):
        # Environment
        self.env = make_multiprocessing_env(env_name, n_cpus)
        self.eval_env = make_singleprocessing_env(env_name)
        self.n_cpus = n_cpus
        self.n_episode_steps = 1000  # 50000 // self.n_cpus


        # Network
        self.device = get_device(device)

        self.actor = actor
        self.actor.to(self.device)

        self.critic = critic
        self.critic.to(self.device)

        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr = critic_lr)
        # parameter_group = list(self.actor.parameters()) + list(self.critic.parameters())
        # self.optimizer = torch.optim.Adam(parameter_group, lr = lr)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': actor_lr},
            {'params': self.critic.parameters(), 'lr': critic_lr}
        ])


        # Rollout buffer
        self.rollout = RolloutBuffer(self.env.single_observation_space, self.env.single_action_space, self.n_cpus, self.n_episode_steps)
        self.rollout.to_device(self.device)


        # Parameters
        self.epsilon = 0.2
        self.clip_loss_coef = 1.0
        self.value_loss_coef = 1.0
        self.action_entropy_loss_coef = 0.1

        self.gamma = gamma
        self.batch_size = batch_size


        # Logging
        self.logging = None



    def predict(self, obs : np.ndarray, deterministic = False):
        with torch.no_grad():
            action, _ = self.actor.act(np_to_tensor(obs, self.device), deterministic)
            return action.detach().cpu().numpy()


    def get_value(self, batch_obs):
        return self.critic(batch_obs).squeeze()



    def update(self, batch_obs, batch_act, batch_old_a_log_prob, batch_V, batch_return):
        # Evaluate actions
        new_a_log_prob, action_entropy = self.actor.evaluate_act(batch_obs, batch_act)
        V_pred = self.critic.get_value(batch_obs)

        # Compute the advantage value function
        batch_A_hat = batch_return - V_pred.detach()
        batch_A_hat = (batch_A_hat - batch_A_hat.mean()) / (batch_A_hat.std() + 1e-8)

        # Compute the surrogate loss
        ratio = (new_a_log_prob - batch_old_a_log_prob).exp()
        surr1 = ratio * batch_A_hat
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_A_hat
        clip_loss = -torch.min(surr1, surr2).mean()
        clip_loss *= self.clip_loss_coef

        # Compute the prediction loss for value function
        value_func_loss = 0.5 * (V_pred - batch_return).pow(2).mean()
        value_func_loss *= self.value_loss_coef
        # # Clipped
        # value_pred_clipped = batch_V + (V_pred - batch_V).clamp(-self.epsilon, self.epsilon)
        # value_losses = (V_pred - batch_return).pow(2)
        # value_losses_clipped = (value_pred_clipped - batch_return).pow(2)
        # value_func_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        # Action entropy loss (for better exploration)
        action_entropy_loss = -action_entropy.mean()
        action_entropy_loss *= self.action_entropy_loss_coef

        # Symmetry loss
        symmetry_loss = 0.0

        # Total loss
        actor_loss = clip_loss + action_entropy_loss + symmetry_loss
        critic_loss = value_func_loss
        total_loss = actor_loss + critic_loss

        training_info = {
            'Training/clip_loss' : clip_loss,
            'Training/action_entropy_loss' : action_entropy_loss.item(),
            # 'Training/symmetry_loss' : symmetry_loss.item(),
            'Training/actor_loss' : actor_loss.item(),
            'Training/value_func_loss' : value_func_loss.item(),
            'Training/critic_loss' : critic_loss.item(),
            'Training/total_loss' : total_loss.item()
        }

        # Update the critic
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()


        return training_info
        


    def learn(self, epochs, log_path = None, eval_freq = None, best_model_dir = None):
        # self.actor.train()
        # self.critic.train()

        max_episode_rewards_mean = float('-inf')

        if log_path is not None:
            self.logging = Logger(log_path)


        curr_obs = self.env.reset()
        curr_obs_tensor = np_to_tensor(curr_obs, self.device).float()

        curr_epoch = 0
        eval_every_episode = int(eval_freq * self.n_cpus * self.n_episode_steps)

        # lr_decay = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [40000, 80000], gamma = 0.1)

        while curr_epoch < epochs:
            self.rollout.reset()

            for _ in range(self.n_episode_steps):
                # Sample trajectories
                with torch.no_grad():
                    action_tensor, action_log_prob_tensor = self.actor.act(curr_obs_tensor)
                    cpu_action = tensor_to_np(action_tensor)
                    value_pred_tensor = self.critic.get_value(curr_obs_tensor)
                    
                next_obs, reward, done, _ = self.env.step(cpu_action)

                self.rollout.add(
                    curr_obs_tensor,
                    action_tensor,
                    np_to_tensor(reward, self.device),
                    np_to_tensor(done, self.device),
                    value_pred_tensor,
                    action_log_prob_tensor
                )
                curr_obs_tensor.copy_(np_to_tensor(next_obs, self.device))

                curr_epoch += self.n_cpus


            # Compute returns
            with torch.no_grad():
                final_V = self.critic.get_value(curr_obs_tensor)
                self.rollout.compute_return(final_V, self.gamma)


            # Training
            epoch_training_info = defaultdict(lambda: 0.0)
            batch_sampler = self.rollout.generate(self.batch_size)
            n_batches = 0
            for sample in batch_sampler:
                n_batches += 1
                batch_obs, batch_act, batch_value_pred, batch_old_action_log_prob, batch_return = sample

                loss_info = self.update(
                    batch_obs, 
                    batch_act, 
                    batch_old_action_log_prob, 
                    batch_value_pred,
                    batch_return
                )

                for loss_name, loss_val in loss_info.items():
                    epoch_training_info[loss_name] += loss_val

            # lr_decay.step()
                    
                    
            # Logging
            for loss_name, loss_val in loss_info.items():
                epoch_training_info[loss_name] /= n_batches

            epoch_training_info['actor_lr'] = self.optimizer.param_groups[0]['lr']
            epoch_training_info['critic_lr'] = self.optimizer.param_groups[1]['lr']

            self.logging.add(curr_epoch, epoch_training_info)


            # Evaluation
            # curr_episode += 1
            if eval_freq is not None and curr_epoch % eval_every_episode == 0:
                (epi_rewards_mean, epi_rewards_std), (epi_steps_mean, epi_steps_std) = evaluate_policy(self.eval_env, self, 50)
                
                eval_info = {
                    'Eval/epi_rewards_mean' : epi_rewards_mean,
                    'Eval/epi_rewards_std' : epi_rewards_std,
                    'Eval/epi_steps_mean' : epi_steps_mean,
                    'Eval/epi_steps_std' : epi_steps_std
                }
                self.logging.add(curr_epoch, eval_info)

                better = True if epi_rewards_mean >= max_episode_rewards_mean else False
                print('Evaluation: Episode Rewards = {:.2f} +- {:.2f} | Episode Steps = {:.2f} +- {:.2f}'.format(epi_rewards_mean, epi_rewards_std, epi_steps_mean, epi_steps_std))
                if better and best_model_dir is not None:
                    print('Get a better model!')
                    max_episode_rewards_mean = epi_rewards_mean
                    self.save(best_model_dir)
                else:
                    print('Don\'t get a better model!')
                    
        

        if self.logging is not None:
            self.logging.close()



    def save(self, root_dir):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        # Save actor & critic for independent usage
        self.actor.save(os.path.join(root_dir, 'actor.pkl'))
        self.critic.save(os.path.join(root_dir, 'critic.pkl'))

        # Save PPO state_dict for tweaking
        state_dict = {
            'actor' : self.actor.state_dict(),
            'critic' : self.critic.state_dict(),
            'optimizer' : self.optimizer.state_dict()
        }
        torch.save(state_dict, os.path.join(root_dir, 'PPO.pkl'))



    def load(self, PPO_pkl_path):
        state_dict = torch.load(PPO_pkl_path)
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.optimizer.load_state_dict(state_dict['optimizer'])


    def load_actor(self, actor_pkl_path):
        self.actor.load(actor_pkl_path, self.device)
        self.actor.to(self.device)


    def load_critic(self, critic_pkl_path):
        self.critic.load(critic_pkl_path, self.device)
        self.critic.to(self.device)
