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
from schedule_policy import Schedule as SchedulePolicy
import os
import math
import time
import sys
import random
import feature_extractor_gnn
from multiprocessing import Pool, Lock, Value
import resource_obj
import model_gnn
import random_test
import gen_data

torch.set_num_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help='start')
parser.add_argument("--number", type=int,default=5000,
                    help="number of sample")
parser.add_argument("--threshold", type=float, default=0.4, help='the threshold of random sampling')
parser.add_argument("--device", type=str, default='cpu', help='cpu')
parser.add_argument("--pool-num", type=int, default=30, help='the threshold of random sampling')
parser.add_argument("--learning-rate", type=float, default=1e-4, help='learning rate')
parser.add_argument("--limit", type=int, default=30, help='plan limit')
parser.add_argument("--debug", action='store_true', help='debug mode')
parser.add_argument("--update", action='store_true', help='debug mode')
parser.add_argument("--size", type=int, default=1280, help='debug mode')
parser.add_argument("--batch-num", type=int, default=1280, help='batch number')
parser.add_argument("--save-count", type=int, default=5)
parser.add_argument("--name", type=str, default='')
parser.add_argument("--eval-count", type=int, default=20, help='debug mode')
args = parser.parse_args()

lock = Lock()
counter = Value('i', 0)
device = torch.device(args.device)
samples = pickle.load(open('sample/sample_{}_{}.pkl'.format(args.number, args.threshold), 'rb'))
print('load {} samples successfully'.format(len(samples)))




def f(sample):
    idx, resources, packages, origin_urgency, total_time, state_dict = sample
    model = model_gnn.GraphNet()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    schedule = SchedulePolicy(resources, packages, None, model, is_dqn=True, time_limit=total_time, plan_limit=30, time_interval=total_time/20, device=device)
    schedule.greedySchedule()
    urgency = schedule.get_urgency()
    del model
    del schedule
    global lock, counter
    with lock:
        counter.value += 1
        if counter.value % 1 == 0:
            sys.stdout.write('\r{}/{}'.format(counter.value, args.size))
            sys.stdout.flush()
    return idx, urgency, origin_urgency

print('#pool: {}'.format(args.pool_num))
# result = []
# for line in data:
#     result.append(f(line))

pool = Pool(args.pool_num)
model_train = model_gnn.GraphNet()
state_dict = torch.load('model/model_IL_{}.ckpt'.format(args.number), map_location=torch.device('cpu'))
if args.update:
    update = pickle.load(open('sample/sample_40000_0.1_764_update_new.pkl', 'rb'))
    print(len(update))
    args.name = 'update'
    for k, v in update.items():
        samples[k][2] = v
    state_dict = torch.load('model/model_IL_{}_update.ckpt'.format(args.number), map_location=torch.device('cpu'))

model_train.load_state_dict(state_dict)
model_train.train()
print('init model')
update_dict = {}

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_train.parameters(), lr=args.learning_rate)
for i in range(len(samples)//args.size):
    print('batch {}'.format(i))
    data = []
    randint = np.arange(i * args.size, (i+1) * args.size)
    for idx in randint:
        resource_pos, packages, urgency, total_time, pid = samples[idx]
        resources = [resource_obj.Resource("resource", resource_pos[0], resource_pos[1], timeLimit=total_time, speed=define.speed)]
        data.append([idx, resources, packages, urgency, total_time, state_dict])

    time0 = time.time()

    result = pool.map(f, data)
    print('\rtime: {:.4f}s'.format(time.time()-time0))

    count = 0
    for idx, u, o_u in result:
        if u > samples[idx][2]:
            # print(u, samples[idx][2], o_u)
            samples[idx][2] = u
            count += 1
            update_dict[idx] = u
    print('count up: {}/{}/{}'.format(count, len(update_dict), args.size * (i + 1)))

    # data2 = [samples[idx] for idx in randint]
    # feature_extractor_gnn.feature_extractor2()
    # result = pool.map(feature_extractor_gnn.feature_extractor_IL, data2)

    # cat_feature = [r[0] for r in result]
    # cat_mask = [r[1] for r in result]
    # cat_urgency = [r[2] for r in result]
    #
    # cat_feature = np.concatenate(cat_feature, axis=0)
    # cat_mask = np.concatenate(cat_mask, axis=0)
    # cat_urgency = np.concatenate(cat_urgency, axis=0)
    #
    # feature_batch = torch.from_numpy(cat_feature).float()
    # mask_batch = torch.from_numpy(cat_mask).float()
    # label_batch = torch.from_numpy(cat_urgency).float()
    #
    # values = model_train(feature_batch, mask_batch)
    # loss = criterion(values, label_batch)
    # print('\rloss: {}'.format(loss.item()))
    #
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    #
    counter.value = 0

    if (i+1) % args.save_count == 0:
        pickle.dump(update_dict, open('sample/sample_{}_{}_{}{}_update_new.pkl'.format(args.number, args.threshold, i, args.name), 'wb'))
        # torch.save(model_train.state_dict(), 'model/model_IL_{}_{}.ckpt'.format(args.number, i))
        # state_dict = model_train.state_dict()
        print('####dump####')
    # if i % args.eval_count == 0:
    #     result = pool.map(random_test.f2,[[i, state_dict, device, gen_data.gen_random_data] for i in range(100000, 100000+50)])
    #     print("\revaluation: {}".format(np.mean(result)))






