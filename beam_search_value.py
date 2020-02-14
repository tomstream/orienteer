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
import DQN

import schedule_greedy
import argparse
torch.set_default_tensor_type('torch.FloatTensor')
torch.set_num_threads(1)


def evaluate(model, gen_data_func, beam_size, seed):

    packages, resources = gen_data_func(package_num=define.package_num,seed=seed)
    path = path_obj.Path(resources[0], packages, lambda x:0, True, device)
    beam_list = [[0, path, False, 0]]
    result = []
    while len(beam_list) > 0:
        new_beam_list = []
        for logp, path, done, utility in beam_list:
            state = path.to_state()
            value = model(*state)
            value = value.cpu().detach().numpy()[0]
            value = list(enumerate(value))
            value.sort(key=lambda x:x[1], reverse=True)
            value = value[:beam_size]
            tmp_done = False
            for a, v in value:
                if v < -10000:
                    break
                package = packages[a]
                times = path.getResourceNeedTime(package) + path.getResourceWorkingTime()

                reward = package.getUrgency()
                current_path = path.copy()
                if times > define.timeLimit:
                    if tmp_done is True:
                        continue
                    current_done = True
                    current_utility = utility
                    current_path.setResourceWorkingTime(path.getResourceWorkingTime())
                    new_beam_list.append([current_utility, current_path, current_done, current_utility])
                    tmp_done = True
                else:
                    current_done = False
                    current_utility = utility + reward
                    current_path.addWorkPackage(package)
                    current_path.setResourceWorkingTime(times)
                    current_path.setResourcePosition(package.getX(), package.getY(), package.getId())
                    new_beam_list.append([current_utility + v, current_path, current_done, current_utility])

        new_beam_list.sort(key=lambda x:x[0], reverse=True)
        new_beam_list = new_beam_list[:beam_size]
        beam_list.clear()
        for logp, path, done, utility in new_beam_list:
            if done:
                result.append([logp, path, done, utility])
            else:
                beam_list.append([logp, path, done, utility])
    return max(result, key=lambda x:x[3])[1]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99, help="number of sample")
    parser.add_argument("--num-step", type=int, default=16, help="limit of feature")
    parser.add_argument("--max-step", type=int, default=1000000, help='the threshold of random sampling')
    parser.add_argument("--num-env", type=int, default=128, help='the threshold of random sampling')
    parser.add_argument("--lr", type=float, default=1e-5, help='the threshold of random sampling')
    parser.add_argument("--device", type=str, default='cuda:2', help='generate RL sample or IL sample')
    parser.add_argument("--model", type=str, default='cuda:2', help='generate RL sample or IL sample')
    parser.add_argument("--hidden-size", type=int, default=32, help='generate RL sample or IL sample')
    parser.add_argument("--nhead", type=int, default=8, help='generate RL sample or IL sample')
    parser.add_argument("--nlayer", type=int, default=4, help='generate RL sample or IL sample')
    parser.add_argument("--mode", type=str, default='uniform', help='generate RL sample or IL sample')
    parser.add_argument("--pool-num", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=10)
    args = parser.parse_args()

    state_dict = torch.load('model/model_{}.ckpt'.format(args.model), map_location=torch.device('cpu'))
    model = DQN.GraphNet(hidden_size=32, n_head=8, nlayers=4, duel_dqn=False)
    model.load_state_dict(state_dict)
    model.eval()
    print('load successfully')

    device  = torch.device(args.device)
    model = model.to(device)

    if args.mode == 'uniform':
        gen_data_func = gen_data.gen_random_data
    else:
        gen_data_func = gen_data.gen_random_dis

    res = []
    for i in range(100):
        ret = evaluate(model, gen_data_func, args.beam_size, 100000+i)
        tmp = ret.getTotalUrgency()
        print(i, tmp)
        res.append(tmp)
    print(np.mean(res))

