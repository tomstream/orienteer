import define

from torch import nn
import torch
torch.set_default_tensor_type('torch.FloatTensor')
import pickle
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import math
import sys

class ConvNet(nn.Module):
    def __init__(self, height_width=40, input_channel=7, conv_dim = [16,32], output_dim=1):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(input_channel, conv_dim[0], kernel_size=5, stride=1),
                                    nn.BatchNorm2d(conv_dim[0]),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(nn.Conv2d(conv_dim[0], conv_dim[1], kernel_size=5, stride=1),
                                    nn.BatchNorm2d(conv_dim[1]),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))

        # self.layer3 = nn.Sequential(nn.Conv2d(conv_dim[1], conv_dim[2], kernel_size=3, stride=1),
        #                             nn.BatchNorm2d(conv_dim[2]),
        #                             nn.ReLU(),
        #                             nn.MaxPool2d(kernel_size=2, stride=2))
        size = (height_width - 4)//2
        size = (size - 4)//2
        # size = (size-2)//2

        self.fc = nn.Sequential(nn.Linear(size * size * conv_dim[1], 10),
                                   nn.ReLU(),
                                   nn.Linear(10, 1))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = out * out
        return out

class MultiLinear(nn.Module):
    def __init__(self, n_in, n_out, n_head):
        super(MultiLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_head = n_head
        self.W = nn.Parameter(torch.rand(n_head, n_in, n_out))
        self.b = nn.Parameter(torch.rand(n_head, 1, n_out))
        # if torch.cuda.is_available():
        #     self.W = self.W.cuda()
        #     self.b = self.b.cuda()

    def forward(self, x):
        x = x.unsqueeze(0)
        x_expand = x.expand(self.n_head, -1, -1)
        out = torch.matmul(x_expand, self.W) + self.b
        return out



class GraphNet(nn.Module):
    def __init__(self, input_size, hidden_size, n_head=10, dropout=0.5):
        super(GraphNet, self).__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.leakyReLu = nn.LeakyReLU()

        self.linear1 = nn.Linear(input_size, self.hidden_size)
        self.linear2 = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),self.dropout,nn.LeakyReLU())
        self.linear3 = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),self.dropout,nn.LeakyReLU())
        self.linear4 = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),self.dropout,nn.LeakyReLU())
        self.linear5 = nn.Sequential(nn.Linear(self.hidden_size * 2, self.hidden_size),self.dropout,nn.LeakyReLU())


        self.outputlayer = nn.Sequential(nn.Linear(hidden_size, hidden_size//2),
                                         self.dropout,
                                         nn.LeakyReLU(),
                                         nn.Linear(hidden_size//2, 1),
                                         self.dropout,
                                         nn.LeakyReLU())
        self.multilinear1 = MultiLinear(self.hidden_size, 1, self.n_head)
        self.multilinear2 = MultiLinear(self.hidden_size, 1, self.n_head)
        self.multilinear3 = MultiLinear(self.hidden_size, 1, self.n_head)
        self.multilinear4 = MultiLinear(self.hidden_size, 1, self.n_head)
        self.multilinear5 = MultiLinear(self.hidden_size, 1, self.n_head)
    def repeat(self, feat):
        # print(feat)
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
        hidden_sum = torch.max(hidden_sum, dim=0)[0]
        return hidden_sum

    def forward(self, feature, package_valid_matrix):
        feat = self.linear1(feature)
        hidden_sum1 = self.GAT(self.multilinear1, feat, package_valid_matrix)

        hidden_repeat, hidden_repeat_transpose = self.repeat(hidden_sum1)
        cat = torch.cat([hidden_repeat, hidden_repeat_transpose], dim=3)
        feat = self.linear2(cat)
        hidden_sum2 = self.GAT(self.multilinear2, feat, package_valid_matrix)

        hidden_repeat, hidden_repeat_transpose = self.repeat(hidden_sum2)
        cat = torch.cat([hidden_repeat, hidden_repeat_transpose], dim=3)
        feat = self.linear3(cat)
        hidden_sum3 = self.GAT(self.multilinear3, feat, package_valid_matrix)

        hidden_repeat, hidden_repeat_transpose = self.repeat(hidden_sum3)
        cat = torch.cat([hidden_repeat, hidden_repeat_transpose], dim=3)
        feat = self.linear4(cat)
        hidden_sum4 = self.GAT(self.multilinear4, feat, package_valid_matrix)

        hidden_repeat, hidden_repeat_transpose = self.repeat(hidden_sum4)
        cat = torch.cat([hidden_repeat, hidden_repeat_transpose], dim=3)
        feat = self.linear5(cat)
        hidden_sum5 = self.GAT(self.multilinear5, feat, package_valid_matrix)

        output = self.outputlayer(hidden_sum5) # B * N * 1
        output = output.squeeze(2)
        # print(np.mean(self.timecount1), np.mean(self.timecount2))
        return output

def optimize_model(policy_net, target_net, optimizer, memory):
    if len(memory) < define.batch_size:
        print('the size is less than batch size',len(memory))
        return
    # print(len(memory))
    policy_net = policy_net.train()
    transitions = memory.sample(define.batch_size)
    print('length of transition', len(transitions))

    loss = 0
    for transition in transitions:
        state, action, next_state, reward = transition
        state_action = state[action]
        state_action_reward = policy_net(state_action)[0]
        if next_state is None:
            next_state_values = 0
        else:
            next_state = torch.cat(list(next_state.values()), dim=0)
            next_state_values = target_net(next_state).max(0)[0].detach()
        expected_state_action_values = next_state_values * define.gamma + reward
        # loss += torch.abs(state_action_reward - expected_state_action_values)
        loss += torch.abs(state_action_reward - expected_state_action_values) * (state_action_reward - expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

if __name__ == "__main__":
    import matplotlib as mpl
    mpl.use('Agg')
    mpl.rc('pdf', fonttype=42)
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str,help="mode of training")
    parser.add_argument("--number", type=int, default=4000, help="number of sample")
    parser.add_argument("--threshold", type=float, default=0.2, help='the threshold of random sampling')
    parser.add_argument("--learning-rate", type=float, default=1e-5, help='learning rate')
    parser.add_argument("--batch-size", type=int, default=512, help='learning rate')
    parser.add_argument("--num-epochs", type=int, default=500, help='# epochs')
    parser.add_argument("--device", type=int, default=0, help='cuda device')
    parser.add_argument("--target-update", type=int, default=1)
    parser.add_argument("--input-number", type=int, default=10000)
    args = parser.parse_args()
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')


    def IL_optimize(data_loader, model, optimizer, epoch, train=True, tmp_loss=None, log=True):
        total_loss = 0
        num = 0
        # criterion = torch.nn.MSELoss(reduction='sum')
        criterion = torch.nn.L1Loss(reduction='sum')
        for i, (features, labels) in enumerate(data_loader):
            features = features.to(device)
            labels = labels.to(device)
            num += features.size(0)

            # Forward pass
            outputs = model(features)
            loss = criterion(outputs[:,0], labels)
            total_loss += loss.item()

            # Backward and optimize
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if log and (i+1) % 10 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, args.num_epochs, i+1, len(data_loader), loss.item()))
                if tmp_loss!=None:
                    tmp_loss.append(loss.item())
        if log:
            sys.stdout.write('\n')
        # return np.sqrt(total_loss/num)
        return total_loss/num

    def IL():
        '''load data'''
        # X, y = pickle.load(open('data/data_{}_{}.pkl'.format(args.number, args.threshold), 'rb'))
        X = np.load('data/datax_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        y = np.load('data/datay_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=define.seed)
        train_X = torch.from_numpy(train_X).float().to(device)
        train_y = torch.from_numpy(train_y).float().to(device)
        test_X = torch.from_numpy(test_X).float().to(device)
        test_y = torch.from_numpy(test_y).float().to(device)

        train_loader = DataLoader(dataset=TensorDataset(train_X, train_y), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=TensorDataset(test_X, test_y), batch_size=args.batch_size, shuffle=False)
        print('finish loading')

        model = ConvNet().to(device)

        # Loss and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.MSELoss(reduction='sum')

        tmp_loss = []
        # Train the model
        for epoch in range(args.num_epochs):
            IL_optimize(train_loader, model.train(), optimizer, epoch, train=True, tmp_loss=tmp_loss)
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    ret = IL_optimize(test_loader, model.eval(), optimizer, epoch, train=False, tmp_loss=None, log=False)
                    print("\r@@evalution avg loss: {}".format(ret))

        # Save the model checkpoint
        if not os.path.exists('model'):
            os.mkdir('model')
        plt.plot(range(len(tmp_loss)), tmp_loss)
        plt.savefig('model/curve_{}.pdf'.format(args.mode))
        torch.save(model.state_dict(), 'model/model_{}_{}.ckpt'.format(args.mode, args.number))

    def RL_optimize(data_loader, policy_model, target_model, optimizer, epoch, train=True, tmp_loss=None, log=True):
        total_loss = 0
        num = 0
        criterion = torch.nn.MSELoss(reduction='sum')
        if train:
            policy_model.train()
        else:
            policy_model.eval()

        for i, (_current, _rest, _reward, _end, _mask) in enumerate(data_loader):
            num += _current.size(0)
            _current =_current.to(device)
            _rest = _rest.to(device)
            _reward = _reward.to(device)
            _end = _end.to(device)
            _mask = _mask.to(device)
            # Forward pass
            value = policy_model(_current).view(-1)
            rest_view = _rest.view(_rest.size(0) * _rest.size(1), _rest.size(2), _rest.size(3), _rest.size(4))
            value_rest = target_model(rest_view)
            value_rest = value_rest.view(_rest.size(0), _rest.size(1))

            max_value = torch.max(value_rest + _reward - (1-_mask)*1e10, dim=1)[0]
            # print(_reward)
            # input()
            estimate_value = max_value * (1 - _end[:,0]) + _reward[:,0] * _end[:,0]
            loss = criterion(value, estimate_value)

            total_loss += loss.item()

            # Backward and optimize
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if log and (i+1) % 10 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                       .format(epoch+1, args.num_epochs, i+1, len(data_loader), loss.item()))
                if tmp_loss is not None:
                    tmp_loss.append(loss.item())
        if log:
            sys.stdout.write('\n')

        return np.sqrt(total_loss/num)

    def to_torch(arg_list):
        ret_list = []
        for arg in arg_list:
            ret_list.append(torch.from_numpy(arg).float())
        return ret_list

    def RL():
        '''load data'''
        current = np.load('data/current_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        rest = np.load('data/rest_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        reward = np.load('data/reward_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        end = np.load('data/end_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        mask = np.load('data/mask_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.1, random_state=define.seed)

        train_current, test_current, train_rest, test_rest, train_reward, test_reward, train_end, test_end, train_mask, test_mask \
            = train_test_split(current, rest, reward, end, mask, test_size=0.1, random_state=define.seed)

        train_current, test_current, train_rest, test_rest, train_reward, test_reward, train_end, test_end, train_mask, test_mask \
            = to_torch([train_current, test_current, train_rest, test_rest, train_reward, test_reward, train_end, test_end, train_mask, test_mask])

        train_loader = DataLoader(dataset=TensorDataset(train_current, train_rest, train_reward, train_end, train_mask), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=TensorDataset(test_current, test_rest, test_reward, test_end, test_mask), batch_size=512, shuffle=False)
        print('finish loading')

        policy_model = ConvNet().to(device)
        target_model = ConvNet().to(device)
        target_model.eval()
        policy_model.load_state_dict(torch.load('model/model_IL_{}.ckpt'.format(args.input_number)))

        target_model.load_state_dict(policy_model.state_dict())
        target_model.eval()

        # Loss and optimizer
        optimizer = torch.optim.Adam(policy_model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.MSELoss(reduction='sum')

        tmp_loss = []
        # Train the model

        for epoch in range(args.num_epochs):
            RL_optimize(train_loader, policy_model, target_model, optimizer, epoch, train=True, tmp_loss=tmp_loss)
            if (epoch + 1) % 1 ==0:
                with torch.no_grad():
                    ret = RL_optimize(test_loader, policy_model, target_model, optimizer, epoch, train=False, tmp_loss=None, log=False)
                    print("@@evalution avg loss: {}".format(ret))
            if (epoch+1) % args.target_update == 0:
                target_model.load_state_dict(policy_model.state_dict())


        # for epoch in range(args.num_epochs):
        #     for i, (_current, _rest, _reward, _end, _mask) in enumerate(data_loader):
        #         _current =_current.to(device)
        #         _rest = _rest.to(device)
        #         _reward = _reward.to(device)
        #         _end = _end.to(device)
        #         _mask = _mask.to(device)
        #         # Forward pass
        #         value = model(_current).view(-1)
        #         rest_view = _rest.view(_rest.size(0) * _rest.size(1), _rest.size(2), _rest.size(3), _rest.size(4))
        #         value_rest = model(rest_view)
        #         value_rest = value_rest.view(_rest.size(0), _rest.size(1))
        #         max_value = torch.max(value_rest + _reward - (1-_mask)*1e10, dim=1)[0]
        #         # print(max_value.size(), _end.size(), _reward.size())
        #         estimate_value = max_value * (1 - _end[:,0]) + _reward[:,0] * _end[:,0]
        #         loss = criterion(value, estimate_value)
        #
        #         # Backward and optimize
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #         if (i+1) % 32 == 0:
        #             print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
        #                    .format(epoch+1, args.num_epochs, i+1, len(data_loader), loss.item()))
        #             tmp_loss.append(loss.item())
        # Save the model checkpoint
        if not os.path.exists('model'):
            os.mkdir('model')
        plt.plot(range(len(tmp_loss)), tmp_loss)
        plt.savefig('model/curve_{}.pdf'.format(args.mode))
        torch.save(policy_model.state_dict(), 'model/model_{}_{}.ckpt'.format(args.mode, args.number))

    if args.mode == 'IL':
        IL()
    elif args.mode == 'RL':
        RL()
    else:
        raise Exception('Undefined Mode')