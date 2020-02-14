import math
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from model_gnn import MultiLinear
import transformer
import  random
import define
import path_obj
import matplotlib.pyplot as plt
import gen_data
import time
import schedule_greedy
import argparse
torch.set_default_tensor_type('torch.FloatTensor')


class GraphNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, n_head=6, nlayers=3,  dropout=0.1):
        super(GraphNet, self).__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.leakyReLu = nn.LeakyReLU()
        self.linear1 = nn.Linear(input_size, self.hidden_size)

        self.multilinear1 = MultiLinear(self.hidden_size, 1, self.n_head)

        encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers)
        self.policy_encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.value_encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.outputlayer = nn.Linear(hidden_size, 1)
        self.policylinear = nn.Linear(hidden_size, 1)

    def repeat(self, feat):
        feat = feat.unsqueeze(1)
        feat_trans = torch.transpose(feat, 1, 2)
        feat_repeat = feat.repeat(1, feat.size(2), 1, 1)
        feat_trans_repeat = feat_trans.repeat(1, 1, feat.size(2), 1)
        return feat_repeat, feat_trans_repeat

    def repeat_matrix(self, feat, length):
        feat = feat.unsqueeze(1).unsqueeze(2)
        feat_repeat = feat.repeat(1, length, length, 1)
        return feat_repeat

    def GAT(self, multilinear, feat, valid):
        feat_view = feat.view(feat.size(0) * feat.size(1) * feat.size(2), feat.size(3))

        attention_weight = multilinear(feat_view) # H, B * N * N, 1
        attention_weight = attention_weight.view(self.n_head, feat.size(0), feat.size(1), feat.size(2), 1)

        weight = torch.exp(self.leakyReLu(attention_weight))
        weight = weight * valid.unsqueeze(0)  # B * N * N * 1

        weight_sum = torch.sum(weight, dim=3).unsqueeze(3) + 1e-9
        weight = weight / weight_sum

        hidden1 = feat * weight
        hidden_sum = torch.sum(hidden1, dim=2)
        hidden_sum = torch.mean(hidden_sum, dim=0)
        return hidden_sum

    def forward(self, feature, package_valid_matrix):
        feat = self.linear1(feature)
        hidden_sum = self.GAT(self.multilinear1, feat, package_valid_matrix)
        hidden_sum = hidden_sum.transpose(0,1)
        mask = torch.diagonal(package_valid_matrix[:, :, :, 0],dim1=1,dim2=2)
        src_key_padding_mask = (1 - mask).bool()
        output = self.transformer_encoder(hidden_sum, src_key_padding_mask=src_key_padding_mask)
        policy_output = self.policy_encoder_layers(output, src_key_padding_mask=src_key_padding_mask).transpose(1,0)
        value_output = self.value_encoder_layers(output, src_key_padding_mask=src_key_padding_mask).transpose(1,0)

        policy_output = self.policylinear(policy_output).squeeze(-1)
        # policy_output = 10 * torch.tanh(policy_output)
        policy_output -= 9999999999 * (1-mask)
        probs = torch.softmax(policy_output, dim=1)

        output = torch.sum(value_output * mask.unsqueeze(-1), dim=1)
        output = self.outputlayer(output)
        return probs, output.squeeze(-1)

class Env:
    def __init__(self, seed_start, seed_end):
        self.range_start = seed_start
        self.range_end = seed_end
        self.count = 0
        self.device = torch.device('cpu')
    def reset(self):
        self.packages, self.resources = gen_data.wrapper(self.range_start + self.count)
        self.count = (self.count + 1) % (self.range_end - self.range_start)
        self.path = path_obj.Path(self.resources[0], self.packages, lambda x:0, True, self.device)
        self.done = 0
        return self.path.to_state()
    def state(self):
        return self.path.to_state()
    def step(self, action, reset=True):
        package = self.packages[action]
        times = self.path.getResourceNeedTime(package) + self.path.getResourceWorkingTime()
        if self.done == 1 or times + self.path.getReturnTime(package) > define.get_value('time_limit'):
            self.done = 1
            reward = 0
            if reset:
                self.reset()
        else:
            self.done = 0
            reward = package.getUrgency()
            self.path.addWorkPackage(package)
            self.path.setResourceWorkingTime(times)
            self.path.setResourcePosition(package.getX(), package.getY(), package.getId())
        return self.path.to_state(), torch.FloatTensor([reward]).to(self.device), torch.FloatTensor([self.done]).to(self.device)


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def evaluate(model, seed):
    returns = 0
    packages, resources = gen_data.wrapper(seed)
    path = path_obj.Path(resources[0], packages, lambda x:0, True, device)
    while True:
        state = path.to_state()
        prob, value = model(*state)
        action = torch.argmax(prob, dim=1)
        action = action.cpu().numpy()[0]
        package = packages[action]
        times = path.getResourceNeedTime(package) + path.getResourceWorkingTime()

        if times + path.getReturnTime(package) > define.get_value('time_limit'):
            return returns
        reward = package.getUrgency()
        path.addWorkPackage(package)
        path.setResourceWorkingTime(times)
        path.setResourcePosition(package.getX(), package.getY(), package.getId())
        returns += reward

    return returns

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.99, help="number of sample")
parser.add_argument("--num-step", type=int, default=16, help="limit of feature")
parser.add_argument("--max-step", type=int, default=1000000, help='the threshold of random sampling')
parser.add_argument("--num-env", type=int, default=16, help='the threshold of random sampling')
parser.add_argument("--lr", type=float, default=1e-3, help='the threshold of random sampling')
parser.add_argument("--device", type=str, default='cuda:2', help='generate RL sample or IL sample')
parser.add_argument("--hidden-size", type=int, default=32, help='generate RL sample or IL sample')
parser.add_argument("--nhead", type=int, default=8, help='generate RL sample or IL sample')
parser.add_argument("--nlayer", type=int, default=4, help='generate RL sample or IL sample')
parser.add_argument("--package-num", type=int, help='generate RL sample or IL sample')
parser.add_argument("--time-limit", type=float, help='generate RL sample or IL sample')
parser.add_argument("--test", action='store_true', help='generate RL sample or IL sample')
parser.add_argument("--sample", action='store_true', help='generate RL sample or IL sample')
parser.add_argument("--func-type", type=str, default='uniform', help='generate RL sample or IL sample')
parser.add_argument("--fn", type=str, default='')
args = parser.parse_args()

define.init()
define.set_value('package_num', args.package_num)
define.set_value('time_limit', args.time_limit)
define.set_value('func_type', args.func_type)
gen_data.generate_data(100000,args.package_num,args.func_type)

# if args.device == 'cpu':
#     torch.set_num_threads(50)

device  = torch.device(args.device)
model = GraphNet( hidden_size=args.hidden_size, n_head=args.nhead, nlayers=args.nlayer).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
gamma = args.gamma
num_steps = args.num_step
max_step = args.max_step
num_env = args.num_env
time_last = time.time()

def step_func(args):
    env, action = args
    ret = env.step(action)
    return env, ret

class Envs:
    def __init__(self, seed_start, seed_end, num_env):
        num_seed = (seed_end - seed_start) // num_env
        self.envs = [Env(seed_start + i * num_seed, seed_start + (i+1) * num_seed) for i in range(0, num_env)]
    def reset(self):
        ret = []
        for env in self.envs:
            ret.append(env.reset())
        return [torch.cat(x, dim=0).to(device) for x in zip(*ret)]
    def step(self, actions):
        result = []
        for line in zip(self.envs, actions):
            result.append(step_func(line))
        # result = pool.map(step_func,zip(self.envs, actions))
        self.envs, ret = list(zip(*result))
        states, rewards, dones = list(zip(*ret))

        return [torch.cat(x, dim=0).to(device) for x in zip(*states)], torch.cat(rewards, dim=0).to(device), torch.cat(dones, dim=0).to(device)
    def to_state(self):
        ret = []
        for env in self.envs:
            ret.append(env.to_state())
        return [torch.cat(x, dim=0).to(device) for x in zip(*ret)]

performance = []

envs = Envs(10000,100000, num_env)


state = envs.reset()

for _i in range(max_step):
    log_probs = []
    values = []
    rewards = []
    masks   = []
    entropy = 0
    for _ in range(num_steps):
        prob, value = model(*state)
        dist = Categorical(prob)
        action = dist.sample()
        next_state, reward, done = envs.step(action.cpu().numpy())
        log_prob = dist.log_prob(action)
        entropy += dist.entropy().mean()

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        masks.append(1-done)

        state = next_state

    _, next_value = model(*next_state)
    # print(len(rewards), len(masks))
    returns = compute_returns(next_value, rewards, masks)

    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)
    advantage = returns - values

    actor_loss  = -(log_probs * advantage.detach()).mean()
    critic_loss = advantage.pow(2).mean()

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if _i%20 == 0:
    # if _i % 10 == 0:
        model.eval()
        result = []
        for _k in range(0, 50):
            ret = evaluate(model.train(), _k)
            result.append(ret)
        time_now = time.time()
        performance.append(np.mean(result))
        print('average performance{}{} {}: {:.4f}, time: {:.4f}'.format(args.func_type, args.package_num, _i, performance[-1], time_now - time_last))
        time_last = time_now
        model.train()
    if (_i) % 500 == 0:
        torch.save(model.state_dict(), 'model/model_a2c2_{}_{}_{}_{}_{}_{}_{}_{}.ckpt'.format(_i+1, args.package_num, args.func_type,num_steps, num_env, args.hidden_size, args.nhead, args.nlayer))
        torch.save(model.state_dict(), 'model/model_a2c2_{}_{}_{}_{}_{}_{}_{}.ckpt'.format(args.package_num, args.func_type,  num_steps, num_env, args.hidden_size, args.nhead, args.nlayer))
        np.save('result/performance2_{}_{}_{}_{}_{}_{}'.format(args.package_num, args.func_type, num_env, args.hidden_size, args.nhead, args.nlayer), performance)
        print('#####dump######')