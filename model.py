import memory
import define

from torch import nn
import torch
torch.set_default_tensor_type('torch.FloatTensor')
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, input_size, dropout_rate):
        super(DQN, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        linear1 = nn.Linear(input_size, 10)
        linear2 = nn.Linear(10, 10)
        linear3 = nn.Linear(10, 4)
        linear4 = nn.Linear(4, 1)

        self.net = nn.Sequential(linear1, self.dropout, nn.ReLU(),
                                 linear2, self.dropout, nn.ReLU(),
                                 linear3, self.dropout, nn.ReLU(),
                                 linear4)

    def forward(self, state_action):
        result = self.net(state_action)
        return result * result

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