# import roboschool
import gym
import pybulletgym
from algo.ppo import PPO
from net.network import MlpActor, MlpCritic
from common.env_utils import make_multiprocessing_env, get_space_dim

class Config(object):
    def __init__(self) -> None:
        self.env_name = 'Humanoid-v4'
        self.n_cpus = 4
        self.lr = 0.0001
        self.actor_lr = 0.0001
        self.critic_lr = 0.0001
        self.gamma = 0.99
        self.batch_size = 128
        self.device = 'auto'
        self.epochs = 120000000
        self.log_path = './PPO/log/'
        self.model_path = './PPO/'
        self.best_model_dir = './PPO/best/'
        self.eval_freq = 100


if __name__ == '__main__':
    config = Config()

    env = make_multiprocessing_env(config.env_name, config.n_cpus)
    # obs_dim = get_space_dim(env.single_observation_space)
    obs_dim = 47
    act_dim = get_space_dim(env.single_action_space)

    actor = MlpActor(obs_dim, act_dim)
    critic = MlpCritic(obs_dim)

    model = PPO(
        env_name = config.env_name,
        actor = actor,
        critic = critic,
        # lr = config.lr,
        actor_lr = config.actor_lr,
        critic_lr = config.critic_lr,
        gamma = config.gamma,
        batch_size = config.batch_size,
        n_cpus = config.n_cpus,
        device = config.device
    )
    # model.load('./exp/Humanoid-v4/best/PPO.pkl')
    model.learn(epochs = config.epochs, log_path = config.log_path, eval_freq = config.eval_freq, best_model_dir = config.best_model_dir)
    model.save(config.model_path)
