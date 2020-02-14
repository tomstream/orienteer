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
    model_new, model_old = model
    packages, resources = gen_data_func(package_num=define.package_num,seed=seed)
    path = path_obj.Path(resources[0], packages, lambda x:0, True, device)
    beam_list = [[path, False, 0]]
    result = []
    while len(beam_list) > 0:
        new_beam_list = []
        for path, done, utility in beam_list:
            state = path.to_state()
            value_new = model_new(*state)
            value_new = value_new.cpu().detach().numpy()[0]

            value_old = model_old(*state)
            value_old = value_old.cpu().detach().numpy()[0]

            value = list(enumerate(value_new))
            value.sort(key=lambda x:x[1], reverse=True)
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
                    new_beam_list.append([current_utility, current_utility, current_path, current_done, current_utility])
                    tmp_done = True
                else:
                    current_done = False
                    current_utility = utility + reward
                    current_path.addWorkPackage(package)
                    current_path.setResourceWorkingTime(times)
                    current_path.setResourcePosition(package.getX(), package.getY(), package.getId())
                    new_beam_list.append([current_utility + v, current_utility + value_old[a], current_path, current_done, current_utility])

        new_beam_list_enumerate = list(enumerate(new_beam_list))
        beam_new = list(reversed(sorted(new_beam_list_enumerate, key=lambda x:x[1][0])))[:beam_size]
        beam_old = list(reversed(sorted(new_beam_list_enumerate, key=lambda x:x[1][1])))[:beam_size]

        beam_new_idx = {idx for idx, _ in beam_new}
        beam_old_idx = {idx for idx, _ in beam_old}

        merge_idx = beam_new_idx.union(beam_old_idx)

        beam_list.clear()
        for idx in merge_idx:
            new, old, path, done, utility = new_beam_list[idx]
            if done:
                result.append([path, done, utility])
            else:
                beam_list.append([path, done, utility])
    return max(result, key=lambda x:x[2])[0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99, help="number of sample")
    parser.add_argument("--num-step", type=int, default=16, help="limit of feature")
    parser.add_argument("--max-step", type=int, default=1000000, help='the threshold of random sampling')
    parser.add_argument("--num-env", type=int, default=128, help='the threshold of random sampling')
    parser.add_argument("--lr", type=float, default=1e-5, help='the threshold of random sampling')
    parser.add_argument("--device", type=str, default='cuda:2', help='generate RL sample or IL sample')
    parser.add_argument("--model-new", type=str, default='cuda:2', help='generate RL sample or IL sample')
    parser.add_argument("--model-old", type=str, default='cuda:2', help='generate RL sample or IL sample')
    parser.add_argument("--hidden-size", type=int, default=32, help='generate RL sample or IL sample')
    parser.add_argument("--nhead", type=int, default=8, help='generate RL sample or IL sample')
    parser.add_argument("--nlayer", type=int, default=4, help='generate RL sample or IL sample')
    parser.add_argument("--mode", type=str, default='uniform', help='generate RL sample or IL sample')
    parser.add_argument("--pool-num", type=int, default=10)
    parser.add_argument("--beam-size", type=int, default=10)
    args = parser.parse_args()

    state_dict_new = torch.load('model/model_{}.ckpt'.format(args.model_new), map_location=torch.device('cpu'))
    state_dict_old = torch.load('model/model_{}.ckpt'.format(args.model_old), map_location=torch.device('cpu'))
    model_new = DQN.GraphNet(hidden_size=32, n_head=8, nlayers=4, duel_dqn=False)
    model_old = DQN.GraphNet(hidden_size=32, n_head=8, nlayers=4, duel_dqn=False)
    model_new.load_state_dict(state_dict_new)
    model_old.load_state_dict(state_dict_old)
    model_new.eval()
    model_old.eval()
    print('load successfully')

    device  = torch.device(args.device)
    model_new = model_new.to(device)
    model_old = model_old.to(device)

    if args.mode == 'uniform':
        gen_data_func = gen_data.gen_random_data
    else:
        gen_data_func = gen_data.gen_random_dis

    res = []
    for i in range(100):
        ret = evaluate([model_new, model_old], gen_data_func, args.beam_size, 100000+i)
        tmp = ret.getTotalUrgency()
        print(i, tmp)
        res.append(tmp)
    print(np.mean(res))

