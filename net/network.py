import torch
import torch.nn as nn
from torch.distributions.normal import Normal


'''
    Base functions
'''
def create_mlp(in_dim : int, out_dim : int, hidden_units : list):
    net = [nn.Linear(in_dim, hidden_units[0]), nn.Tanh()]
    for i in range(len(hidden_units) - 1):
        net.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
        net.append(nn.Tanh())
    net.append(nn.Linear(hidden_units[-1], out_dim))

    net = nn.Sequential(*net)

    return net


'''
    Classes
'''
class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()


    def init_weight(self):
        for module in self.parameters():
            nn.init.orthogonal_(module.weight.data)
            if module.bias.data is not None:
                nn.init.zeros_(module.bias.data)


    def save(self, path):
        torch.save(self.state_dict(), path)


    def load(self, path, device):
        self.load_state_dict(torch.load(path, map_location = device))
        
        
# '''
#     Reference: https://github.com/UBCMOCCA/SymmetricRL/blob/master/common/controller.py
# '''
# class AddBias(nn.Module):
#     def __init__(self, bias):
#         super(AddBias, self).__init__()
#         self._bias = nn.Parameter(bias.unsqueeze(1))

#     def forward(self, x):
#         if x.dim() == 2:
#             bias = self._bias.t().view(1, -1)
#         else:
#             bias = self._bias.t().view(1, -1, 1, 1)

#         return x + bias


# class DiagGaussian(nn.Module):
#     def __init__(self, num_outputs) -> None:
#         super().__init__()
#         self.logstd = AddBias(torch.zeros(num_outputs))

#     def forward(self, action_mean):
#         zeros = torch.zeros(action_mean.size())
#         if action_mean.is_cuda:
#             zeros = zeros.cuda()
#         # action_logstd = self.logstd(zeros) - 1
#         action_logstd = zeros - 1

#         return torch.distributions.Normal(action_mean, action_logstd.exp())


''' 
    Actor
'''
class BaseActor(BaseNet):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        self.main = None
        self.cov_mat = torch.diag(torch.full(size = (act_dim,), fill_value = 0.5))


    def forward(self, obs):
        return self.main(obs.float())
        # output = self.main(obs)
        # mean, logstd = torch.split(output, output.size(-1)// 2, -1)
        # return mean, logstd


    def act(self, obs, deterministic = False):
        # mean = self.forward(obs)
        # action_dist = self.dist(mean)
        # action = mean if deterministic else action_dist.sample()
        # action_log_prob = action_dist.log_prob(action).sum(-1)
        # return action, action_log_prob

        # mean, logstd = self.forward(obs)
        # action_dist = torch.distributions.Normal(mean, logstd.exp())
        # action = mean if deterministic else action_dist.sample()
        # action_log_prob = action_dist.log_prob(action).sum(-1)
        # return action, action_log_prob

        mean = self.forward(obs)
        action_dist = torch.distributions.MultivariateNormal(mean, self.cov_mat.to(mean.device))
        action = mean if deterministic else action_dist.sample()
        action_log_prob = action_dist.log_prob(action)
        return action, action_log_prob


    def evaluate_act(self, batch_obs, batch_act):
        # mean = self.forward(batch_obs)
        # action_dist = self.dist(mean)
        # action_log_prob = action_dist.log_prob(batch_act).sum(-1)
        # action_entropy = action_dist.entropy().mean(-1)
        # return action_log_prob, action_entropy

        # mean, logstd = self.forward(batch_obs)
        # action_dist = torch.distributions.Normal(mean, logstd.exp())
        # action_log_prob = action_dist.log_prob(batch_act).sum(-1)
        # action_entropy = action_dist.entropy()
        # return action_log_prob, action_entropy

        mean = self.forward(batch_obs)
        action_dist = torch.distributions.MultivariateNormal(mean, self.cov_mat.to(mean.device))
        action_log_prob = action_dist.log_prob(batch_act)
        action_entropy = action_dist.entropy()
        return action_log_prob, action_entropy


class MlpActor(BaseActor):
    def __init__(self, obs_dim, act_dim):
        super().__init__(obs_dim, act_dim)

        # self.main = create_mlp(obs_dim, act_dim, [256, 256])
        # self.main.append(nn.Tanh())

        self.main = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )





'''
    Critic
'''
class BaseCritic(BaseNet):
    def __init__(self, obs_dim):
        super().__init__()

        self.main = None


    # def forward(self, obs):
    #     return self.main(obs)


    def get_value(self, obs):
        return self.main(obs.float()).squeeze()



class MlpCritic(BaseCritic):
    def __init__(self, obs_dim):
        super().__init__(obs_dim)

        # self.main = create_mlp(obs_dim, 1, [256, 256])

        self.main = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )