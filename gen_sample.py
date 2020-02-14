from work_package import WorkPackage
from resource_obj import Resource
from schedule import Schedule
from path_obj import Path
import copy
import define
from memory import ReplayMemory
import random
import time
import pickle
import model
import torch.optim as optim
import argparse
import os
import sys
import gen_data
from torch import nn
import torch
torch.set_default_tensor_type('torch.FloatTensor')
from multiprocessing import Pool, Lock, Value

print(torch.__version__)

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help='start')
parser.add_argument("--number", type=int,default=5000,
                    help="number of sample")
parser.add_argument("--threshold", type=float, default=0.4, help='the threshold of random sampling')
parser.add_argument("--pool-num", type=int, default=30, help='the threshold of random sampling')
parser.add_argument("--limit", type=int, default=30, help='plan limit')
parser.add_argument("--debug", action='store_true', help='debug mode')
parser.add_argument('--mode', type=str, default='uniform')

args = parser.parse_args()






# for _ in range(args.number):
#     if _ % 10 == 0:
#         sys.stdout.write("{}/{}\r".format(_, args.number))
#         sys.stdout.flush()
#
#     workpackages, resources = gen_random_data(package_num=define.package_num)
#     schedule = Schedule(resources, workpackages, replay_memory=None, net=None, is_dqn=False, time_limit=define.timeLimit, plan_limit=40, time_interval=0.05)
#     schedule.greedySchedule()
#     path = schedule.getPath()[0]
#     path_packages = path.getWorkPackages()
#     id2workpackages = {p.getId():p for p in workpackages}
#     resource = path.getResource()
#     resource_pos = [resource.getInitialX(), resource.getInitialY()]
#     resource_time = 0
#     total_urgency = path.getTotalUrgency()
#     total_time = define.timeLimit
#
#     if random.random() < args.threshold:
#         data.append([resource_pos, list(id2workpackages.values()), total_urgency])
#     for i in range(len(path_packages)):
#         wk = path_packages[i]
#         id2workpackages.pop(wk.getId(), None)
#         reward = wk.getUrgency()
#         resource_pos[0] = wk.getX()
#         resource_pos[1] = wk.getY()
#         total_time -= wk.getWorkingTime()
#         total_urgency -= reward
#         if random.random() < args.threshold:
#             data.append([resource_pos, list(id2workpackages.values()), total_urgency, total_time])
lock = Lock()
counter = Value('i', 0)

def f(seed):
    data = []
    if args.mode == 'uniform':
        workpackages, resources = gen_data.gen_random_data(package_num=define.package_num, seed=seed)
    else:
        workpackages, resources = gen_data.gen_random_dis(package_num=define.package_num, seed=seed)
    schedule = Schedule(resources, workpackages, replay_memory=None, net=None, is_dqn=False, time_limit=define.timeLimit, plan_limit=int(args.limit), time_interval=0.05)
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
            tmp = Resource("resource", resource_pos[0], resource_pos[1], timeLimit=total_time, speed=define.speed)
            resources.append(tmp)
            packages = list(id2workpackages.values())
            schedule = Schedule(resources, packages, replay_memory=None, net=None, is_dqn=False, time_limit=total_time, plan_limit=args.limit, time_interval=total_time/20)
            schedule.greedySchedule()
            tmp_path = schedule.getPath()[0].getWorkPackages()
            if len(tmp_path) > 0:
                pid = tmp_path[0].getId()
            else:
                pid = -1
                continue
            urgency = schedule.get_urgency()
            if args.debug:
                print(total_time, urgency)
            data.append([resource_pos, packages, urgency, total_time, pid])
    if args.debug:
        input()
    global lock, counter
    with lock:
        counter.value += 1
        if counter.value % 5 == 0:
            sys.stdout.write('\r{}/{}, {}'.format(counter.value, args.number, len(data)))
            sys.stdout.flush()
    return data




if __name__ == '__main__':

    if args.debug:
        result = []
        print('start')
        for i in range(args.start, args.start + args.number):
            result.append(f(i))
    else:
        pool = Pool(args.pool_num)
        result = pool.map(f, range(args.start, args.start + args.number))
        pool.close()
        pool.join()

    # print(result)

    data = [r_i for r in result for r_i in r]
    print(len(data))
    if not os.path.exists('sample'):
        os.mkdir('sample')
    pickle.dump(data, open('sample/sample_{}_{}_{}.pkl'.format(args.number, args.threshold, args.mode), 'wb'))

