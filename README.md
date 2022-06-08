# RL Framework

​	My implementation of Reinforcement Learning algorithms.



#### Folders

- algo: Training algorithms.
- common: Some helper functions.
- exp: Save experimental results, including models and plots.
- net: Neural network architecture.



#### Usage

- Install the prerequisites `pip install -r requirements.txt`
- To train a model, modify train.py.
- To evaluate a model, modify eval.py.



#### Update on 06/08/2022

- Implement [PPO](https://arxiv.org/abs/1707.06347) and train on BipedWalker-v3, HopperPyBulletEnv-v0 ([pybullet](https://pybullet.org/wordpress/) and [pybulletgym](https://github.com/benelot/pybullet-gym) required) and Humanoid-v4 ([mujoco](https://www.gymlibrary.ml/environments/mujoco/) required).

- For Humanoid-v4, the first 47 dimensions of the observation space is fed into the network.

  