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
from memory import ReplayMemory
import DQN
import pickle
torch.set_default_tensor_type('torch.FloatTensor')


class GraphNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=32, n_head=6, n_gat_head=8, nlayers=4, duel_dqn=True, dropout=0.1):
        super(GraphNet, self).__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.n_gat_head = n_gat_head
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.leakyReLu = nn.LeakyReLU()
        self.linear1 = nn.Linear(input_size, self.hidden_size)

        self.multilinear1 = MultiLinear(self.hidden_size, 1, self.n_gat_head)

        encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers)
        self.adv_encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.value_encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.outputlayer = nn.Linear(hidden_size, 1)
        self.advlinear = nn.Linear(hidden_size, 1)
        self.duel_dqn =duel_dqn

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
        attention_weight = attention_weight.view(self.n_gat_head, feat.size(0), feat.size(1), feat.size(2), 1)

        weight = torch.exp(self.leakyReLu(attention_weight))
        weight = weight * valid.unsqueeze(0)  # B * N * N * 1

        weight_sum = torch.sum(weight, dim=3).unsqueeze(3) + 1e-9
        weight = weight / weight_sum

        hidden1 = feat * weight
        hidden_sum = torch.sum(hidden1, dim=2)
        hidden_sum = torch.mean(hidden_sum, dim=0)
        return hidden_sum

    def vanilla_forward(self, feature, package_valid_matrix):
        feat = self.linear1(feature)
        hidden_sum = self.GAT(self.multilinear1, feat, package_valid_matrix)
        hidden_sum = hidden_sum.transpose(0,1)
        mask = torch.diagonal(package_valid_matrix[:, :, :, 0],dim1=1,dim2=2)
        src_key_padding_mask = (1 - mask).bool()
        value_output = self.transformer_encoder(hidden_sum, src_key_padding_mask=src_key_padding_mask).transpose(1,0)
        # value_output = self.value_encoder_layers(output, src_key_padding_mask=src_key_padding_mask).transpose(1,0)
        # value_output = self.value_encoder_layers(output, src_key_padding_mask=src_key_padding_mask).transpose(1,0)

        value_output = self.outputlayer(value_output).squeeze(-1)
        value_output -= 9999999999 * (1-mask)

        # output = torch.sum(value_output * mask.unsqueeze(-1), dim=1)
        # output = self.outputlayer(output)
        return value_output

    def duel_forward(self, feature, package_valid_matrix):
        feat = self.linear1(feature)
        hidden_sum = self.GAT(self.multilinear1, feat, package_valid_matrix)
        hidden_sum = hidden_sum.transpose(0,1)
        mask = torch.diagonal(package_valid_matrix[:, :, :, 0],dim1=1,dim2=2)
        src_key_padding_mask = (1 - mask).bool()
        output = self.transformer_encoder(hidden_sum, src_key_padding_mask=src_key_padding_mask)
        adv_output = self.adv_encoder_layers(output, src_key_padding_mask=src_key_padding_mask).transpose(1,0)
        value_output = self.value_encoder_layers(output, src_key_padding_mask=src_key_padding_mask).transpose(1,0)
        adv = self.advlinear(adv_output).squeeze(-1)

        value = torch.sum(value_output * mask.unsqueeze(-1), dim=1)
        value = self.outputlayer(value) # B * 1

        sum_mask = torch.sum(mask, dim=1)
        value_adv = value - (torch.sum(adv * mask, dim=1)/sum_mask).unsqueeze(1)

        ret = value_adv + adv
        ret -= 9999999999 * (1-mask)
        return ret
    def forward(self, feature, package_valid_matrix):
        if self.duel_dqn:
            return self.duel_forward(feature, package_valid_matrix)
        else:
            return self.vanilla_forward(feature, package_valid_matrix)

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
    def to_state(self):
        return self.path.to_state()
    def get_total_reward(self):
        return self.path.getTotalUrgency()
    def state(self):
        return self.path.to_state()
    def step(self, action, reset=True):
        package = self.packages[action]
        times = self.path.getResourceNeedTime(package) + self.path.getResourceWorkingTime()
        if self.done==1 or times + self.path.getReturnTime(package) > define.get_value('time_limit'):
            self.done = 1
            done = 1
            reward = 0
            if reset:
                self.reset()
        else:
            self.done = 0
            done = 0
            reward = package.getUrgency()
            self.path.addWorkPackage(package)
            self.path.setResourceWorkingTime(times)
            self.path.setResourcePosition(package.getX(), package.getY(), package.getId())
            if self.path.getWorkPackageSize() >= define.get_value('package_num'):
                self.done = 1
                done = 1
                self.reset()
        return self.path.to_state(), torch.FloatTensor([reward]).to(self.device), torch.FloatTensor([done]).to(self.device)


def evaluate(model, seed):
    returns = 0
    packages, resources = gen_data.wrapper(seed)
    path = path_obj.Path(resources[0], packages, lambda x:0, True, device)
    action_all = []
    while True:
        state = path.to_state()
        value = model(*state)
        action = torch.argmax(value, dim=1)
        action = action.cpu().numpy()[0]
        action_all.append(action)
        package = packages[action]
        times = path.getResourceNeedTime(package) + path.getResourceWorkingTime()
        if times + path.getReturnTime(package) > define.get_value('time_limit'):
            return returns,action_all[:-1]
        reward = package.getUrgency()
        path.addWorkPackage(package)
        path.setResourceWorkingTime(times)
        path.setResourcePosition(package.getX(), package.getY(), package.getId())
        returns += reward

    return returns, action_all[-1]

def evaluate_batch(model, start, batch_size):
    envs = Envs(start, start + batch_size, batch_size)
    states = envs.reset()
    baseline = 0
    all_actions = []

    while True:
        values = model(*states)
        actions = torch.argmax(values, dim=1)
        actions = actions.cpu().numpy()
        states, rewards, dones = envs.step(actions, False)
        done_numpy = dones.cpu().numpy()
        baseline += np.sum(rewards.cpu().numpy())
        all_actions.append(actions[0])
        if np.sum(1 - done_numpy) == 0:
            break
    return baseline/batch_size,all_actions

def beam_search(model, beam_size):
    import copy
    total_baselines = 0
    time_start = time.time()
    for _index in range(0,10000):
        env = Env(_index, _index+1)
        beam_list = [[env, 0, 0, 0]]
        max_rewards = 0
        state = env.reset()
        envs = Envs([env for env, prob, action, reward in beam_list])
        while True:
            state = envs.to_state()
            value = model(*state)
            value = value.cpu().detach().numpy()
            tmp_list = []
            for i in range(len(beam_list)):
                for j in range(define.get_value('package_num')):
                    if value[i, j] < 1e-10:
                        continue
                    tmp_list.append([beam_list[i][0], beam_list[i][1] + value[i, j], j, beam_list[i][3]])
            if len(tmp_list) == 0:
                break
            tmp_list.sort(key = lambda x:x[1], reverse=True)
            tmp_list = tmp_list[:beam_size]
            envs = Envs([copy.deepcopy(env) for env, prob, action, r in tmp_list])
            action = [a for env, prob, a, r in tmp_list]
            next_state, reward, done = envs.step(action, False)

            beam_list.clear()
            reward = reward.cpu().numpy()
            for i, (env, prob, a, r) in enumerate(tmp_list):
                r = r + reward[i]
                max_rewards = max(max_rewards, r)
                beam_list.append([copy.deepcopy(envs.envs[i]), envs.envs[i].get_total_reward(), a, r])
            if np.sum(1 - done.cpu().numpy()) == 0:
                break
        total_baselines += max_rewards
        print('{} {} {}'.format(_index, total_baselines/(_index+1), (time.time()-time_start)/(_index+1)))

def learn(memory, eval_net, target_net, learn_step_counter, double_dqn=True):
    if (learn_step_counter + 1) % q_network_iteration ==0:
        target_net.load_state_dict(eval_net.state_dict())
    eval_net.train()
    target_net.eval()

    batch_state, batch_mask, batch_action, batch_reward, batch_next_state, batch_next_mask, batch_done = [torch.cat(l, dim=0).to(device) for l in zip(*memory.sample(batch_size))]
    batch_action = batch_action.unsqueeze(1)
    batch_reward = batch_reward.unsqueeze(1)
    batch_done = batch_done.unsqueeze(1)
    #q_eval
    # print(q_eval.size(), batch_action.size())
    q_eval = eval_net(batch_state, batch_mask).gather(1, batch_action)
    # print(q_eval.size())

    if double_dqn:
        q_next_action = torch.argmax(eval_net(batch_next_state, batch_next_mask), dim=1).detach().unsqueeze(-1)
        q_next = eval_net(batch_next_state, batch_next_mask).gather(1, q_next_action)
    else:
        q_next = target_net(batch_next_state, batch_next_mask).max(1)[0].view(batch_size, 1)

    q_target = batch_reward + gamma * q_next * (1 - batch_done)
    # print(q_eval.size(), q_next_action.size(), batch_action.size(), q_target.size())

    loss = loss_func(q_eval, q_target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

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
    def reset(self):
        ret = []
        for env in self.envs:
            ret.append(env.reset())
        return [torch.cat(x, dim=0).to(device) for x in zip(*ret)]
    def step(self, actions, reset=True):
        result = []

        for env, action in zip(self.envs, actions):
            result.append(env.step(action, reset))
        ret = result
        states, rewards, dones = list(zip(*ret))

        return [torch.cat(x, dim=0).to(device) for x in zip(*states)], torch.cat(rewards, dim=0).to(device), torch.cat(dones, dim=0).to(device)
    def to_state(self):
        ret = []
        for env in self.envs:
            ret.append(env.to_state())
        return [torch.cat(x, dim=0).to(device) for x in zip(*ret)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=1, help="number of sample")
    parser.add_argument("--epsilon", type=float, default=0.90, help="number of sample")
    parser.add_argument("--batch-size", type=int, default=128, help="number of sample")
    parser.add_argument("--max-step", type=int, default=10000000, help='the threshold of random sampling')
    parser.add_argument("--num-env", type=int, default=32, help='the threshold of random sampling')
    parser.add_argument("--lr", type=float, default=5e-3, help='the threshold of random sampling')
    parser.add_argument("--device", type=str, default='cuda:2', help='generate RL sample or IL sample')
    parser.add_argument("--hidden-size", type=int, default=32, help='generate RL sample or IL sample')
    parser.add_argument("--nhead", type=int, default=8, help='generate RL sample or IL sample')
    parser.add_argument("--n-gat-head", type=int, default=8, help='generate RL sample or IL sample')
    parser.add_argument("--nlayer", type=int, default=4, help='generate RL sample or IL sample')
    parser.add_argument("--package-num", type=int, help='generate RL sample or IL sample')
    parser.add_argument("--time-limit", type=float, help='generate RL sample or IL sample')
    parser.add_argument("--duel-dqn", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--double-dqn", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--test", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--sample", action='store_true', help='generate RL sample or IL sample')
    parser.add_argument("--func-type", type=str, default='uniform', help='generate RL sample or IL sample')
    parser.add_argument("--fn", type=str, default='')
    parser.add_argument('--path', action='store_true')
    parser.add_argument('--beam', action='store_true')
    args = parser.parse_args()

    define.init()
    define.set_value('package_num', args.package_num)
    define.set_value('time_limit', args.time_limit)
    define.set_value('func_type', args.func_type)
    if args.device == 'cpu':
        torch.set_num_threads(58)
    else:
        torch.set_num_threads(1)

    if args.beam:
        device  = torch.device(args.device)
        model = GraphNet( hidden_size=args.hidden_size, n_head=args.nhead, nlayers=args.nlayer, duel_dqn=args.duel_dqn)
        model.load_state_dict(torch.load('model/model_dqn{}_{}_{}_{}_{}_{}_{}_{}_{}.ckpt'.format(args.fn, define.get_value('package_num'), args.func_type, args.num_env, args.hidden_size, args.nhead, args.nlayer, 'double' if args.double_dqn else 'vanilla', 'duel' if args.duel_dqn else 'vanilla')
                                         ,map_location=torch.device('cpu')))
        model = model.to(device)

        model.eval()
        gen_data.generate_data(10000, args.package_num, args.func_type)

        beam_search(model, 100)
        exit()

    if args.path:

        device  = torch.device(args.device)
        model = GraphNet( hidden_size=args.hidden_size, n_head=args.nhead, nlayers=args.nlayer, duel_dqn=args.duel_dqn)
        model.load_state_dict(torch.load('model/model_dqn{}_{}_{}_{}_{}_{}_{}_{}_{}.ckpt'.format(args.fn, define.get_value('package_num'), args.func_type, args.num_env, args.hidden_size, args.nhead, args.nlayer, 'double' if args.double_dqn else 'vanilla', 'duel' if args.duel_dqn else 'vanilla')
                                         ,map_location=torch.device('cpu')))
        model = model.to(device)
        total_baselines = 0
        ret = []
        gen_data.generate_data(10000,args.package_num,args.func_type)
        for i in range(50):
            ret.append(evaluate_batch(model, i, 1))
        pickle.dump(ret, open('solution/dqn_greedy.pkl', 'wb'))
        exit()


    if not args.test:
        gen_data.generate_data(100000,args.package_num,args.func_type)

        device  = torch.device(args.device)
        eval_net = GraphNet( hidden_size=args.hidden_size, n_head=args.nhead, nlayers=args.nlayer, duel_dqn=args.duel_dqn, n_gat_head=args.n_gat_head).to(device)
        target_net = GraphNet( hidden_size=args.hidden_size, n_head=args.nhead, nlayers=args.nlayer, duel_dqn=args.duel_dqn, n_gat_head=args.n_gat_head).to(device)
        optimizer = torch.optim.Adam(eval_net.parameters(), lr=args.lr)
        gamma = args.gamma
        epsilon = args.epsilon
        batch_size = args.batch_size
        max_step = args.max_step
        num_env = args.num_env
        time_last = time.time()
        loss_func = nn.MSELoss()
        learn_step_counter = 0
        q_network_iteration = 10

        memory = ReplayMemory(4096)

        performance = []
        envs = Envs(10000,100000, num_env)

        state = envs.reset()

        for _i in range(max_step):
            eval_net.eval()
            values = eval_net(*state)
            # print(values)
            # print(state)
            if random.random() > epsilon:
                prob_uniform = (values > -9999999).float()
                dist = Categorical(prob_uniform)
                action = dist.sample()
            else:
                action = torch.argmax(values, dim=1)
            next_state, reward, done = envs.step(action.cpu().numpy())
            for idx in range(state[0].size(0)):
                memory.append([state[0][[idx]].cpu(),
                               state[1][[idx]].cpu(),
                               action[[idx]].cpu(),
                               reward[[idx]].cpu(),
                               next_state[0][[idx]].cpu(),
                               next_state[1][[idx]].cpu(),
                               done[[idx]].cpu()]
                              )
            state = next_state

            if len(memory) >= memory.capacity:
                learn(memory, eval_net, target_net, learn_step_counter, args.double_dqn)
                learn_step_counter += 1

            if _i % 50 == 0:
                ret = evaluate_batch(eval_net, 0, 100)
                ret = ret[0]
                performance.append(ret)
                time_now = time.time()
                print('average performance on{} {}{} {}: {:.4f}, time: {:.4f}, step: {}'.format(args.fn, args.func_type,args.package_num, _i, performance[-1], time_now - time_last, learn_step_counter//q_network_iteration))

            # if _i % 10 == 0:
            #     result = []
            #     for _k in range(0, 50):
            #         ret = evaluate(eval_net, _k)
            #         result.append(ret)
            #     time_now = time.time()
            #     performance.append(np.mean(result))
            #     print('average performance on {}{} {}: {:.4f}, time: {:.4f}, step: {}'.format(args.func_type,args.package_num, _i, performance[-1], time_now - time_last, learn_step_counter//q_network_iteration))
            #     time_last = time_now
            if (_i) % 50 == 0:
                torch.save(eval_net.state_dict(), 'model/model_dqn{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.ckpt'.format(args.fn, _i+1, define.get_value('package_num'), args.func_type, num_env, args.hidden_size, args.nhead, args.nlayer, 'double' if args.double_dqn else 'vanilla', 'duel' if args.duel_dqn else 'vanilla'))
                torch.save(eval_net.state_dict(), 'model/model_dqn{}_{}_{}_{}_{}_{}_{}_{}_{}.ckpt'.format(args.fn, define.get_value('package_num'), args.func_type, num_env, args.hidden_size, args.nhead, args.nlayer, 'double' if args.double_dqn else 'vanilla', 'duel' if args.duel_dqn else 'vanilla'))
                # np.save('result/performance{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.fn, args.func_type, define.get_value('package_num'), num_env, args.hidden_size, args.nhead, args.nlayer,'double' if args.double_dqn else 'vanilla', 'duel' if args.duel_dqn else 'vanilla'), performance)
                np.save('result/performance{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args.fn, args.func_type, define.get_value('package_num'), num_env, args.hidden_size, args.nhead, args.nlayer,'double' if args.double_dqn else 'vanilla', 'duel' if args.duel_dqn else 'vanilla'), performance)
                print('#####dump######')
    else:
        gen_data.generate_data(100000,args.package_num,args.func_type)
        device  = torch.device(args.device)
        model = GraphNet( hidden_size=args.hidden_size, n_head=args.nhead, nlayers=args.nlayer, duel_dqn=args.duel_dqn)
        model.load_state_dict(torch.load('model/model_dqn{}_{}_{}_{}_{}_{}_{}_{}_{}.ckpt'.format(args.fn, define.get_value('package_num'), args.func_type, args.num_env, args.hidden_size, args.nhead, args.nlayer, 'double' if args.double_dqn else 'vanilla', 'duel' if args.duel_dqn else 'vanilla')
                                         ,map_location=torch.device('cpu')))
        model = model.to(device)
        total_baselines = 0
        for i in range(10000//args.batch_size):
            envs = Envs(i * args.batch_size, (i+1) * args.batch_size, args.batch_size)
            returns = 0
            states = envs.reset()
            baseline = 0
            while True:
                values = model(*states)
                actions = torch.argmax(values, dim=1)
                actions = actions.cpu().numpy()
                states, rewards, dones = envs.step(actions, False)

                baseline += np.sum(rewards.cpu().numpy())
                if np.sum(1 - dones.cpu().numpy()) == 0:
                    break
            total_baselines += baseline
            print('{} {}'.format(i, total_baselines/(i+1)/args.batch_size))


