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
from scipy.stats import ttest_rel
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_num_threads(1)




class Env:
    def __init__(self, seed_start, seed_end):
        self.range_start = seed_start
        self.range_end = seed_end
        self.count = 0
        self.device = torch.device('cpu')
    def reset(self, i):
        self.seed = self.range_start + i % (self.range_end - self.range_start)
        self.count = (self.count + 1) % (self.range_end - self.range_start)
        self.packages, self.resources = gen_data.wrapper(self.seed)
        self.mask = torch.zeros(1, define.get_value('package_num')+ 1)
        self.mask[0, -1] =1
        self.path = path_obj.Path(self.resources[0], self.packages, None, False, self.device)
        self.reward = 0
        self.times = 0
    def step(self, action):
        if self.mask[0, action] == 1:
            done = 1
            reward = 0
        else:
            package = self.packages[action]
            self.times = self.path.getResourceNeedTime(package) + self.path.getResourceWorkingTime()
            if self.times + self.path.getReturnTime(package) > define.get_value('time_limit'):
                done = 1
                reward =0
            else:
                done = 0
                self.mask[0, action] = 1
                reward = package.getUrgency()
                self.path.addWorkPackage(package)
                self.path.setResourceWorkingTime(self.times)
                self.path.setResourcePosition(package.getX(), package.getY(), package.getId())

        return torch.FloatTensor([reward]).to(self.device), torch.FloatTensor([done]).to(self.device), self.mask.to(self.device)


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def evaluate(seed_start, num ):
    envs = Envs(seed_start, seed_start + num, num)

    sum_rewards = 0
    envs.reset(0)
    while True:
        masks = envs.masks()
        dist = Categorical(1-masks)
        actions = dist.sample()
        rewards, dones, masks, all_done = envs.step(actions.cpu().detach().numpy())
        sum_rewards += rewards * (1-dones)
        if all_done:
            break

    return sum_rewards.cpu().numpy()

def step_func(args):
    env, action = args
    ret = env.step(action)
    return env, ret

class Envs:
    def __init__(self, seed_start, seed_end, num_env):
        num_seed = (seed_end - seed_start) // num_env
        self.envs = [Env(seed_start + i * num_seed, seed_start + (i+1) * num_seed) for i in range(0, num_env)]
    def reset(self, i):
        ret = []
        for env in self.envs:
            ret.append(env.reset(i))
        # print([env.seed for env in self.envs]
    def masks(self):
        return torch.cat([env.mask for env in self.envs], dim=0)
    def times(self):
        return torch.FloatTensor([env.times for env in self.envs])
    def step(self, actions):
        result = []
        for env, a in zip(self.envs, actions):
            result.append(env.step(a))
        # result = pool.map(step_func,zip(self.envs, actions))
        rewards, dones, mask = list(zip(*result))
        all_stop = torch.sum(1 - torch.cat(dones, dim=0)).cpu().detach().numpy()
        return torch.cat(rewards, dim=0).to(device), torch.cat(dones, dim=0).to(device), torch.cat(mask, dim=0).to(device), all_stop==0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--batch-size", type=int, default=100, help='generate RL sample or IL sample')
    parser.add_argument("--package-num", type=int, help='generate RL sample or IL sample')
    parser.add_argument("--time-limit", type=float, help='generate RL sample or IL sample')
    parser.add_argument("--func-type", type=str, help='generate RL sample or IL sample')
    parser.add_argument("--device", type=str, help='generate RL sample or IL sample')
    args = parser.parse_args()

    define.init()
    define.set_value('package_num', args.package_num)
    define.set_value('time_limit', args.time_limit)
    define.set_value('func_type', args.func_type)
    gen_data.generate_data(100000, args.package_num, args.func_type)

    device  = torch.device(args.device)
    total_baselines = 0
    for i in range(10000 // args.batch_size):
        total_baselines += np.sum(evaluate(i * args.batch_size, args.batch_size))
        print('{} {}'.format(i, total_baselines / (i + 1) / args.batch_size))
