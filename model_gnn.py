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
import time
import sys
import transformer


class MultiLinear(nn.Module):
    def __init__(self, n_in, n_out, n_head):
        super(MultiLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_head = n_head
        self.W = nn.Parameter(torch.rand(n_head, n_in, n_out))
        self.b = nn.Parameter(torch.rand(n_head, 1, n_out))

    def forward(self, x):
        x = x.unsqueeze(0)
        x_expand = x.expand(self.n_head, -1, -1)
        out = torch.matmul(x_expand, self.W) + self.b
        return out

class GAT(nn.Module):
    def __init__(self, n_in, n_head=10):
        super(GAT, self).__init__()
        self.multilinear = MultiLinear(n_in, 1, n_head)
    def forward(self, feat, valid):
        feat_view = feat.view(feat.size(0) * feat.size(1) * feat.size(2), feat.size(3))

        attention_weight = self.multilinear(feat_view) # H, B * N * N, 1
        attention_weight = attention_weight.view(self.n_head, feat.size(0), feat.size(1), feat.size(2), 1)

        weight = torch.exp(self.leakyReLu(attention_weight))
        weight = weight * valid.unsqueeze(0)  # B * N * N * 1

        weight_sum = torch.sum(weight, dim=3).unsqueeze(3) + 1e-9
        weight = weight / weight_sum

        hidden1 = feat * weight
        hidden_sum = torch.sum(hidden1, dim=2)
        hidden_sum = torch.max(hidden_sum, dim=0)[0]
        return hidden_sum

class GraphNet(nn.Module):
    def __init__(self, input_size=8, hidden_size=64, n_head=4, nlayers=3,  dropout=0.1):
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

        # policy_output = self.policylinear(policy_output).squeeze(-1)
        # policy_output -= 9999999999 * (1-mask)
        output = torch.sum(value_output * mask.unsqueeze(-1), dim=1)
        output = self.outputlayer(output)
        return output.squeeze(-1)


class GraphNetXX(nn.Module):
    def __init__(self, input_size=8, hidden_size=20, n_head=4, nlayers=3,  dropout=0.1):
        super(GraphNet, self).__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.leakyReLu = nn.LeakyReLU()
        self.linear1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout())
        self.attn1 = nn.Linear(hidden_size, 1)
        self.attn2 = nn.Linear(hidden_size, 1)
        self.linear2 = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.ReLU(), nn.Dropout())

        encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers)

        self.value_encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        self.outputlayer = nn.Linear(hidden_size, 1)

    def forward(self, feature, package_valid_matrix):
        mask = torch.diagonal(package_valid_matrix[:, :, :, 0],dim1=1,dim2=2)
        src_key_padding_mask = (1 - mask).bool()

        feature = self.linear1(feature)
        weight = self.attn1(feature) - (1 - package_valid_matrix) * 9999999999
        weight = torch.softmax(weight, dim=2)
        hidden_sum = torch.sum(weight * feature, dim=2)
        hidden_sum = hidden_sum.transpose(0,1)
        output = self.encoder_layers(hidden_sum, src_key_padding_mask=src_key_padding_mask).transpose(1,0)

        expand = output.unsqueeze(2).expand(-1,-1,40,-1)

        cat = torch.cat([expand, expand.transpose(2,1), feature], dim=3)
        feature = self.linear2(cat)
        weight = self.attn2(feature) - (1 - package_valid_matrix) * 9999999999
        weight = torch.softmax(weight, dim=2)
        hidden_sum = torch.sum(weight * feature, dim=2)
        hidden_sum = hidden_sum.transpose(0,1)
        value_output = self.transformer_encoder(hidden_sum, src_key_padding_mask=src_key_padding_mask).transpose(1,0)

        output = torch.sum(value_output * mask.unsqueeze(-1), dim=1)
        output = self.outputlayer(output).squeeze(-1)
        return output

class GNN(nn.Module):
    def __init__(self, input_size=8, hidden_size=20, n_head=4, nlayers=3,  dropout=0.1):
        super(GNN, self).__init__()
        self.input_size = input_size
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.leakyReLu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.linear1 = nn.Sequential(nn.Dropout(),nn.ReLU(),nn.Linear(input_size, self.hidden_size))
        self.linear2 = nn.Sequential(nn.Dropout(),nn.ReLU(),nn.Linear(self.hidden_size * 3, self.hidden_size))
        self.layerNorm = nn.LayerNorm(self.hidden_size)

        self.multilinear1 = MultiLinear(self.hidden_size, 1, self.n_head)

        # encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        # self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers)
        # self.policy_encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        # self.value_encoder_layers = transformer.TransformerEncoderLayer(hidden_size, n_head, hidden_size * 2, dropout)
        # self.outputlayer = nn.Linear(hidden_size, 1)
        # self.policylinear = nn.Linear(hidden_size, 1)

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

    def forward(self, feature, package_valid_matrix, matrix=True):
        hidden_sum = self.GAT(self.multilinear1, feature, package_valid_matrix) # B * N * H
        if not matrix:
            return hidden_sum
        nodes_matrix = hidden_sum.unsqueeze(2).repeat(1, 1, define.package_num, 1)
        feature = torch.cat([nodes_matrix, nodes_matrix.transpose(2,1), feature], dim=3)
        feature = self.linear2(feature) * package_valid_matrix
        return feature

class GraphNetM(nn.Module):
    def __init__(self, input_size=8, hidden_size=20, n_head=4, nlayers=3,  dropout=0.1):
        super(GraphNet, self).__init__()
        self.linear1 = nn.Sequential(nn.Dropout(),nn.ReLU(),nn.Linear(input_size, hidden_size))
        self.graph_models = nn.ModuleList([GNN(hidden_size=hidden_size, n_head=n_head, nlayers=nlayers, dropout=dropout) for _ in range(nlayers)])
        self.outputlayer = nn.Linear(hidden_size, 1)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.multilinear1 = MultiLinear(hidden_size, 1, n_head)

    def forward(self, feature, package_valid_matrix):
        mask = torch.diagonal(package_valid_matrix[:, :, :, 0],dim1=1,dim2=2)
        feature = self.linear1(feature)
        for i in range(len(self.graph_models)-1):
            feature = self.graph_models[i](feature, package_valid_matrix)
        feature = self.graph_models[i](feature, package_valid_matrix, False)
        output = torch.sum(feature * mask.unsqueeze(-1), dim=1)
        output = self.linear2(output).squeeze(-1)
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
    parser.add_argument("--name", type=str, default='', help="name")
    parser.add_argument("--threshold", type=str, default=0.2, help='the threshold of random sampling')
    parser.add_argument("--learning-rate", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--batch-size", type=int, default=512, help='learning rate')
    parser.add_argument("--num-epochs", type=int, default=500, help='# epochs')
    parser.add_argument("--device", type=str, default='cuda:2', help='cuda device')
    parser.add_argument("--target-update", type=int, default=1)
    parser.add_argument("--input-number", type=int, default=10000)
    parser.add_argument("--update", action='store_true')
    args = parser.parse_args()
    device = torch.device(args.device)

    define.init()


    def IL_optimize(data_loader, model, optimizer, epoch, train=True, tmp_loss=None, log=True):
        total_loss = 0
        total_ce = 0
        num = 0
        criterion_value = torch.nn.MSELoss(reduction='sum')
        criterion_ce = torch.nn.CrossEntropyLoss(reduction='sum')
        loss_count = []
        for i, (features, masks, labels, pid) in enumerate(data_loader):
            features = features.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            pid = pid.to(device).long()
            valid_pid = pid >= 0
            features = features[valid_pid]
            masks = masks[valid_pid]
            labels = labels[valid_pid]
            pid = pid[valid_pid]

            num += features.size(0)

            # Forward pass
            values = model(features, masks)
            value_loss = criterion_value(values, labels)
            loss = value_loss/features.size(0)
            total_loss += (criterion_value(values, labels)).item()

            # Backward and optimize
            if train:
                optimizer.zero_grad()
                loss.backward()
                # for param in model.parameters():
                #     param.grad.data.clamp_(-1, 1)
                optimizer.step()

            if log and (i+1) % 10 == 0:
                sys.stdout.flush()
                sys.stdout.write('\rEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}, entropy: {:.4f}'
                       .format(epoch+1, args.num_epochs, i+1, len(data_loader), value_loss.item(), 0))
                if tmp_loss!=None:
                    tmp_loss.append(value_loss.item())
            # del values,masks,labels,features
            # torch.cuda.empty_cache()
        if log:
            sys.stdout.write('\n')
        return np.sqrt(total_loss/num), total_ce/num

    def IL():
        '''load data'''
        feature = np.load('data/gnn_feature_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        mask = np.load('data/gnn_mask_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        urgency = np.load('data/gnn_urgency_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        pid = np.load('data/gnn_id_{}_{}_{}.npy'.format(args.mode, args.number, args.threshold))
        model = GraphNet().to(device)
        model.train()

        if args.update:
            update = pickle.load(open('sample/sample_40000_0.1_764_update_new.pkl', 'rb'))
            print(len(update))
            for k, v in update.items():
                urgency[k] = v
            state_dict = torch.load('model/model_IL_{}.ckpt'.format(args.number), map_location=device)
            model.load_state_dict(state_dict)
            args.name = args.name + '_update'
        print('load successfully from disk')

        train_feature, test_feature, train_mask, test_mask, train_urgency, test_urgency , train_pid, test_pid = train_test_split(feature, mask, urgency, pid, test_size=0.1, random_state=define.seed)
        train_feature, test_feature, train_mask, test_mask, train_urgency, test_urgency, train_pid, test_pid = to_torch([train_feature, test_feature, train_mask, test_mask, train_urgency, test_urgency, train_pid, test_pid])


        train_loader = DataLoader(dataset=TensorDataset(train_feature, train_mask, train_urgency,train_pid), batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=TensorDataset(test_feature, test_mask, test_urgency, test_pid), batch_size=args.batch_size, shuffle=False)
        print('finish create dataset')



        # Loss and optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        tmp_loss = []
        # Train the model
        for epoch in range(args.num_epochs):
            start_time = time.time()
            ret = IL_optimize(train_loader, model.train(), optimizer, epoch, train=True, tmp_loss=tmp_loss)
            end_time = time.time()
            print("\rtime for a epoch: {:.2f}s loss: {:.4f} {:.4f}".format(end_time - start_time, ret[0], ret[1]))
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    ret = IL_optimize(test_loader, model.eval(), optimizer, epoch, train=False, tmp_loss=None, log=False)
                    print("@@evalution avg loss: {}, {}".format(*ret))
                    if not os.path.exists('model'):
                        os.mkdir('model')
                    torch.save(model.state_dict(), 'model/model_{}_{}{}.ckpt'.format(args.mode, args.number, args.name))

        # Save the model checkpoint

        plt.plot(range(len(tmp_loss)), tmp_loss)
        plt.savefig('model/curve_{}.pdf'.format(args.mode))


    def RL_optimize(data_loader, policy_model, target_model, optimizer, epoch, train=True, tmp_loss=None, log=True):
        total_loss = 0
        num = 0
        criterion = torch.nn.MSELoss(size_average=False)
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