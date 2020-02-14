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


def extract_feat(resource, work_packages):
    work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency(), p.getId()] for p in work_packages]
    rx, ry, rid = resource.getPosition()
    gnn_feature = np.zeros((define.get_value('package_num') + 1, define.get_value('package_num') + 1, 4))

    for p_i in work_packages:
        i = p_i[5]
        for p_j in work_packages:
            j = p_j[5]
            gnn_feature[i, j, 0] = p_i[2]
            gnn_feature[i, j, 1] = p_j[2]
            gnn_feature[i, j, 2] = define.dis(p_i[0], p_j[0], p_i[1], p_j[1]) / define.get_value('speed') + p_j[3]
            gnn_feature[i, j, 3] = 1
    for p in work_packages:
        i = p[5]
        gnn_feature[-1, i, 2] = define.dis(p[0], rx, p[1], ry) / define.get_value('speed') + p[3]
        gnn_feature[i, -1, 2] = define.dis(p[0], rx, p[1], ry) / define.get_value('speed')
    return torch.from_numpy(gnn_feature).unsqueeze(0).float()

class GraphNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, n_head=4, nlayers=2,  dropout=0.1):
        super(GraphNet, self).__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.leakyReLu = nn.LeakyReLU()
        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.norm1 = torch.nn.BatchNorm1d(hidden_size)

        self.multilinear1 = MultiLinear(self.hidden_size, 1, self.n_head)

        encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers)

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

    def GAT(self, multilinear, feat):
        feat_view = feat.view(feat.size(0) * feat.size(1) * feat.size(2), feat.size(3))

        attention_weight = multilinear(feat_view) # H, B * N * N, 1
        attention_weight = attention_weight.view(self.n_head, feat.size(0), feat.size(1), feat.size(2), 1)

        weight = torch.exp(self.leakyReLu(attention_weight))

        weight_sum = torch.sum(weight, dim=3).unsqueeze(3)
        weight = weight / weight_sum

        hidden1 = feat * weight
        hidden_sum = torch.sum(hidden1, dim=2)
        hidden_sum = torch.mean(hidden_sum, dim=0)
        hidden_sum = self.norm1(hidden_sum.transpose(1,2)).transpose(1,2)
        return hidden_sum

    def forward(self, feature):
        feat = self.linear1(feature)
        hidden_sum = self.GAT(self.multilinear1, feat)
        hidden_sum = hidden_sum.transpose(1,0)
        output = self.transformer_encoder(hidden_sum).transpose(1,0)
        return output

class GraphNetDecoder(nn.Module):
    def __init__(self, hidden_size=32, dropout=0.1):
        super(GraphNetDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.vf = nn.Parameter(torch.rand(1, hidden_size))
        self.linear1 = nn.Linear(hidden_size * 2 + 1, hidden_size)
        self.self_attn = transformer.MultiheadAttention(hidden_size, 1, dropout=dropout, my_scaling=True)
        self.qlinear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.klinear = nn.Linear(hidden_size, hidden_size, bias=False)
    def forward(self, node_embedding, mask, times, idxc=None):
        node_avg = torch.sum(node_embedding * (1-mask.unsqueeze(-1)), dim=1)/torch.sum(1-mask.unsqueeze(-1),dim=1)
        if idxc is None:
            f = self.vf.expand([node_embedding.size(0),-1])
        else:
            # print(idxc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_size).size(), node_embedding.size())
            f = node_embedding.gather(1, idxc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.hidden_size)).squeeze(1)
        emb = self.linear1(torch.cat([node_avg, f, times.unsqueeze(1)], dim=1))
        emb = emb.unsqueeze(0).transpose(0,1)
        weight = torch.sum(self.qlinear(emb) * self.klinear(node_embedding), dim=-1)/math.sqrt(emb.size(-1))
        weight -= mask * 999999999
        weight = torch.softmax(weight, dim=-1)
        # assert torch.sum(torch.sum(1-mask, dim=1) < 1) < 1, 'zero'
        return weight

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
        self.mask = torch.zeros(1, define.get_value('package_num') + 1)
        self.mask[0, -1] =1
        self.path = path_obj.Path(self.resources[0], self.packages, None, False, self.device)
        self.reward = 0
        self.times = 0
        self.done = 0
        return extract_feat(self.resources[0], self.packages).to(device)
    def step(self, action):
        if self.mask[0, action] == 1:
            done = 1
            reward = 0
            print('seed')
        else:
            package = self.packages[action]
            self.times = self.path.getResourceNeedTime(package) + self.path.getResourceWorkingTime()
            if self.done == 1 or self.times + self.path.getReturnTime(package) > define.get_value('time_limit'):
                done = 1
                self.done = 1
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

def evaluate(encoder, decoder, seed_start, num ):
    envs = Envs(seed_start, seed_start + num, num)
    encoder.eval()
    decoder.eval()

    sum_rewards = 0
    state = envs.reset(0)

    emb = encoder(state)
    first = True
    while True:
        if first:
            first = False
            dis = decoder(emb, envs.masks().to(device),envs.times().to(device), None)
            prob, actions = torch.max(dis, dim=1)
        else:
            dis = decoder(emb, masks, envs.times().to(device), last_actions)
            prob, actions = torch.max(dis, dim=1)

        last_actions = actions
        rewards, dones, masks, all_done = envs.step(actions.cpu().detach().numpy())
        sum_rewards += rewards * (1-dones)
        if all_done:
            break

    return sum_rewards.cpu().numpy()

def beam_search(encoder, decoder, beam_size):
    import copy
    total_baselines = 0
    time_start = time.time()
    for _index in range(0,10000):
        env = Env(_index, _index+1)
        beam_list = [[env, 1, 0, 0]]
        max_rewards = 0
        state = env.reset(0)
        envs = Envs([env for env, prob, action, reward in beam_list])
        emb = encoder(state)
        first = True
        while True:

            if first:
                first = False
                prob = decoder(emb.expand(len(envs.envs), -1, -1), envs.masks().to(device), envs.times().to(device), None)
            else:
                prob = decoder(emb.expand(len(envs.envs), -1, -1), envs.masks().to(device), envs.times().to(device), torch.LongTensor([a for e, p, a, r in beam_list]).to(device))

            prob = prob.cpu().detach().numpy()
            tmp_list = []
            for i in range(len(beam_list)):
                for j in range(define.get_value('package_num')):
                    if prob[i, j] < 1e-10:
                        continue
                    tmp_list.append([beam_list[i][0], beam_list[i][1] * prob[i, j], j, beam_list[i][3]])
            tmp_list.sort(key = lambda x:x[1], reverse=True)
            tmp_list = tmp_list[:beam_size]
            envs = Envs([copy.deepcopy(env) for env, prob, action, r in tmp_list])
            action = [a for env, prob, a, r in tmp_list]
            rewards, dones, masks, all_done = envs.step(action)

            beam_list.clear()
            rewards = rewards.cpu().numpy()
            for i, (env, prob, a, r) in enumerate(tmp_list):
                r = r + rewards[i]
                max_rewards = max(max_rewards, r)
                beam_list.append([copy.deepcopy(envs.envs[i]), prob, a, r])
            if np.sum(1 - dones.cpu().numpy()) == 0:
                break
        total_baselines += max_rewards
        print('{} {} {}'.format(_index, total_baselines/(_index+1), (time.time()-time_start)/(_index+1)))


def step_func(args):
    env, action = args
    ret = env.step(action)
    return env, ret

class Envs:
    def __init__(self, seed_start, seed_end=None, num_env=None):
        if seed_end != None:
            num_seed = (seed_end - seed_start) // num_env
            self.envs = [Env(seed_start + i * num_seed, seed_start + (i+1) * num_seed) for i in range(0, num_env)]
        else:
            self.envs = seed_start
    def reset(self, i):
        ret = []
        for env in self.envs:
            ret.append(env.reset(i))
        # print([env.seed for env in self.envs])
        return torch.cat(ret, dim=0).to(device)
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
    parser.add_argument("--gamma", type=float, default=0.99, help="number of sample")
    parser.add_argument("--num-step", type=int, default=16, help="limit of feature")
    parser.add_argument("--max-step", type=int, default=1000000, help='the threshold of random sampling')
    parser.add_argument("--num-env", type=int, default=8, help='the threshold of random sampling')
    parser.add_argument("--lr", type=float, default=1e-3, help='the threshold of random sampling')
    parser.add_argument("--lr-d", type=float, default=2, help='the threshold of random sampling')
    parser.add_argument("--device", type=str, default='cuda:2', help='generate RL sample or IL sample')
    parser.add_argument("--hidden-size", type=int, default=32, help='generate RL sample or IL sample')
    parser.add_argument("--nhead", type=int, default=8, help='generate RL sample or IL sample')
    parser.add_argument("--nlayer", type=int, default=3, help='generate RL sample or IL sample')
    parser.add_argument("--mode", type=str, default='uniform', help='generate RL sample or IL sample')
    parser.add_argument("--ntest", type=int, default=300, help='generate RL sample or IL sample')
    parser.add_argument("--test", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--package-num", type=int, help='generate RL sample or IL sample')
    parser.add_argument("--batch-size", type=int, default=100, help='generate RL sample or IL sample')
    parser.add_argument("--time-limit", type=float, help='generate RL sample or IL sample')
    parser.add_argument("--duel-dqn", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--double-dqn", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--beam", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--func-type", type=str, default='uniform', help='generate RL sample or IL sample')
    parser.add_argument("--fn", type=str, default='')
    args = parser.parse_args()

    define.init()
    define.set_value('package_num', args.package_num)
    define.set_value('time_limit', args.time_limit)
    define.set_value('func_type', args.func_type)
    gen_data.generate_data(100000, args.package_num, args.func_type)

    device  = torch.device(args.device)
    encoder = GraphNet( hidden_size=args.hidden_size, n_head=args.nhead, nlayers=args.nlayer).to(device)
    decoder = GraphNetDecoder(hidden_size=args.hidden_size).to(device)

    if args.beam:
        encoder.load_state_dict(
            torch.load('model/model_att_encoder_{}_{}.ckpt'.format(define.get_value('package_num'), args.func_type)
                       , map_location=device))
        decoder.load_state_dict(
            torch.load('model/model_att_decoder_{}_{}.ckpt'.format(define.get_value('package_num'), args.func_type)
                       , map_location=device))
        encoder.eval()
        decoder.eval()
        gen_data.generate_data(10000, args.package_num, args.func_type)

        beam_search(encoder, decoder, 100)

    if args.test:
        encoder.load_state_dict(torch.load('model/model_att_encoder_{}_{}.ckpt'.format(define.get_value('package_num'),args.func_type)
                                         ,map_location=device))
        decoder.load_state_dict(torch.load('model/model_att_decoder_{}_{}.ckpt'.format(define.get_value('package_num'),args.func_type)
                                         ,map_location=device))
        total_baselines = 0
        for i in range(10000 // args.batch_size):
            total_baselines += np.sum(evaluate(encoder, decoder, i * args.batch_size, args.batch_size))
            print('{} {}'.format(i, total_baselines / (i + 1) / args.batch_size))
        exit()

    encoder_old = encoder
    decoder_old = decoder

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr * args.lr_d)
    gamma = args.gamma
    num_steps = args.num_step
    max_step = args.max_step
    num_env = args.num_env
    time_last = time.time()

    performance = []
    print(num_env)

    envs = Envs(11000,90000, num_env)

    old_result = evaluate(encoder_old, decoder_old, 10000, args.ntest)


    for _i in range(max_step):
        encoder.train()
        decoder.train()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        sum_log_probs = 0
        values = []
        sum_rewards = 0
        entropy = 0
        state = envs.reset(_i)
        emb = encoder(state)
        first = True
        while True:
            if first:
                first = False
                prob = decoder(emb, envs.masks().to(device), envs.times().to(device), None)
                dist = Categorical(prob)
                actions = dist.sample()
                log_prob = dist.log_prob(actions)
            else:
                prob = decoder(emb, masks, envs.times().to(device), last_actions)
                dist = Categorical(prob)
                actions = dist.sample()
                log_prob = dist.log_prob(actions)

            last_actions = actions
            rewards, dones, masks, all_done = envs.step(actions.cpu().detach().numpy())
            if all_done:
                break

            sum_log_probs += log_prob * (1-dones)
            sum_rewards += rewards * (1-dones)

        baselines = 0
        state = envs.reset(_i)
        emb = encoder_old(state)
        first=True
        while True:
            if first:
                first = False
                prob = decoder_old(emb, envs.masks().to(device), envs.times().to(device),None)
                prob, actions = torch.max(prob, dim=1)
            else:
                prob = decoder_old(emb, masks, envs.times().to(device), last_actions)
                prob, actions = torch.max(prob, dim=1)

            last_actions = actions
            rewards, dones, masks, all_done = envs.step(actions.cpu().detach().numpy())
            if all_done:
                break

            baselines += rewards * (1-dones)

        # print(sum_log_probs, sum_rewards)
        actor_loss  = -(sum_log_probs * (sum_rewards-baselines).detach()).mean()

        # loss = actor_loss  - 0.001 * entropy
        loss = actor_loss

        loss.backward()

        ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
        dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1)

        encoder_optimizer.step()
        decoder_optimizer.step()

        if _i % 50 == 0:
            # encoder_old.load_state_dict(encoder.state_dict())
            # decoder_old.load_state_dict(decoder.state_dict())
            ret_c = evaluate(encoder, decoder, 10000, args.ntest)
            mean_c = np.mean(ret_c)
            mean_o = np.mean(old_result)
            print('att', args.package_num, args.func_type, mean_c, mean_o)
            if mean_c - mean_o < 0.01:
                continue
            t, p = ttest_rel(old_result, ret_c)
            p_val = p / 2  # one-sided
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))

            if p_val < 0.05:
                print('Update baseline')
                encoder_old.load_state_dict(encoder.state_dict())
                decoder_old.load_state_dict(decoder.state_dict())
                torch.save(encoder_old.state_dict(),'model/model_att_encoder_{}_{}.ckpt'.format(define.get_value('package_num'),args.func_type))
                torch.save(decoder_old.state_dict(),'model/model_att_decoder_{}_{}.ckpt'.format(define.get_value('package_num'),args.func_type))
                # encoder.load_state_dict(torch.load(
                #     'model/model_att_encoder_{}_{}.ckpt'.format(define.get_value('package_num'), args.func_type)
                #     , map_location=device))
                # decoder.load_state_dict(torch.load(
                #     'model/model_att_decoder_{}_{}.ckpt'.format(define.get_value('package_num'), args.func_type)
                #     , map_location=device))
                old_result = evaluate(encoder_old, decoder_old, 10000, args.ntest)