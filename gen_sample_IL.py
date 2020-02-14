from work_package import WorkPackage
from resource_obj import Resource
from schedule import Schedule
from path_obj import Path
import copy
import define
from memory import ReplayMemory
from schedule_policy import Schedule as SchedulePolicy
import random
import pickle
import gen_data
import argparse
import os
import numpy as np
import sys
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from multiprocessing import Pool, Lock, Value
torch.set_num_threads(1)
import model_gnn
import schedule_policy
import time

print(torch.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help='start')
parser.add_argument("--model", type=str, default='IL_40000', help="start of seed")
parser.add_argument("--name", type=str, default='', help="name")
parser.add_argument("--device", type=str, default='cpu', help="name")
parser.add_argument("--number", type=int,default=5000,help="number of sample")
parser.add_argument("--threshold", type=float, default=0.4, help='the threshold of random sampling')
parser.add_argument("--pool-num", type=int, default=10, help='the threshold of random sampling')
parser.add_argument("--limit", type=int, default=20, help='plan limit')
parser.add_argument("--debug", action='store_true', help='debug mode')

args = parser.parse_args()


lock = Lock()
counter = Value('i', 0)
spent_time = 0

def f(seed):
    time0 = time.time()
    data = []

    model = model_gnn.GraphNet()
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    torch.no_grad()

    workpackages, resources = gen_data.gen_random_data(package_num=define.package_num, seed=seed)
    schedule = SchedulePolicy(resources, workpackages, None, model, is_dqn=True, time_limit=define.timeLimit, plan_limit=10, time_interval=define.timeInterval, device=device)
    schedule.greedySchedule()
    path = schedule.getPath()[0]
    path_packages = path.getWorkPackages()
    id2workpackages = {p.getId():p for p in workpackages}
    resource = path.getResource()
    resource_pos = [resource. getInitialX(), resource.getInitialY()]
    total_urgency = path.getTotalUrgency()
    total_time = define.timeLimit

    if random.random() < args.threshold:
        data.append([resource_pos, list(id2workpackages.values()), total_urgency, total_time, path_packages[0].getId()])
    for i in range(len(path_packages)):
        wk = path_packages[i]
        id2workpackages.pop(wk.getId(), None)
        reward = wk.getUrgency()
        total_time -= wk.getWorkingTime() + define.dis(resource_pos[0], wk.getX(), resource_pos[1], wk.getY())/define.speed
        resource_pos[0] = wk.getX()
        resource_pos[1] = wk.getY()
        total_urgency -= reward
        if random.random() < args.threshold:
            resources = []
            random.seed(seed)
            tmp = Resource("resource", resource_pos[0], resource_pos[1], timeLimit=define.timeLimit, speed=define.speed)
            resources.append(tmp)
            packages = list(id2workpackages.values())
            schedule = SchedulePolicy(resources, workpackages, None, model, is_dqn=True, time_limit=define.timeLimit, plan_limit=args.limit, time_interval=define.timeInterval, device=device)
            schedule.greedySchedule()
            tmp_path = schedule.getPath()[0].getWorkPackages()
            if len(tmp_path) > 0:
                pid = tmp_path[0].getId()
            else:
                pid = -1
            urgency = schedule.get_urgency()
            # if args.debug:
            #     print(total_time, urgency)
            data.append([resource_pos, packages, urgency, total_time, pid])
    # if args.debug:
    #     input()
    global lock, counter, spent_time
    with lock:
        spent_time += time.time() - time0
        counter.value += 1
        if counter.value % 1 == 0:
            sys.stdout.write('\r{}/{}, {}'.format(counter.value, args.number, spent_time/counter.value))
            sys.stdout.flush()
    return data

if __name__ == '__main__':
    device = torch.device(args.device)
    state_dict = torch.load('model/model_{}.ckpt'.format(args.model), map_location=torch.device('cpu'))

    data_in = np.random.randint(0,100000-1,args.number)
    # print(data_in)
    if args.debug:
        result = []
        print('start')
        for i in data_in:
            result.append(f(i))
    else:
        pool = Pool(args.pool_num)
        result = pool.map(f, data_in)
        pool.close()
        pool.join()

    # print(result)

    data = [r_i for r in result for r_i in r]
    print(len(data))
    if not os.path.exists('sample'):
        os.mkdir('sample')
    pickle.dump(data, open('sample/sample_{}_{}_IL{}.pkl'.format(args.number, args.threshold, args.name), 'wb'))

