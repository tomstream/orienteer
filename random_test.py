import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool, Lock, Value

import numpy as np
import torch

import DQN
from baselines import A2C
from data import gen_data
from schedule_prize import Schedule
from schedule_learn import Schedule as SchedulePolicy
from utils import model_gnn, define


def load():
    return pickle.load(open('/Users/liuzongtao/data/rl-data/synthesis_data.pkl', 'rb'))

lock = Lock()
counter = Value('i', 0)
timer = Value('f', 0)
values = Value('f', 0)



def no_dqn_schedule( data, plan_limit=None, time_limit=None, time_interval=None):
    plan_limit = define.get_value('plan_limit') if plan_limit is None else plan_limit
    time_limit = define.get_value('time_limit') if time_limit is None else time_limit
    time_interval = define.get_value('time_interval') if time_interval is None else time_interval
    workpackages, resources = data
    schedule = Schedule(resources, workpackages, replay_memory=None, net=None, is_dqn=False, time_limit=time_limit, plan_limit=plan_limit, time_interval=time_interval)
    schedule.greedySchedule()
    # schedule.vis('x_{}_{}'.format(seed,plan_limit))
    # print('time spent : {0:.3f}s'.format(time.time() - time0))
    return schedule.get_urgency(), schedule.path_idx

def dqn_schedule(model, data, device, plan_limit=None, time_limit=None, time_interval=None):
    plan_limit = define.get_value('plan_limit') if plan_limit is None else plan_limit
    time_limit = define.get_value('time_limit') if time_limit is None else time_limit
    time_interval = define.get_value('time_interval') if time_interval is None else time_interval
    data0, data1 = data
    workpackages, resources = data0
    time0 = time.time()
    schedule = SchedulePolicy(resources, workpackages, None, model, is_dqn=True, time_limit=time_limit, plan_limit = plan_limit, time_interval=time_interval, device=device, batch_size=args.batch_size, data=data1)
    schedule.greedySchedule()
    ret = schedule.get_urgency()
    global lock, counter, timer, values
    with lock:
        timer.value += time.time() - time0
        values.value += ret
        counter.value += 1
        if counter.value % 1 == 0:
            sys.stdout.write('\r{} {:.4f} {:.4f}'.format(counter.value, values.value/counter.value, timer.value/counter.value))
            sys.stdout.flush()

    return ret, schedule.path_idx


def f(data):
    result = []
    # result.append(no_dqn_schedule(data, 10))
    # time_start = time.time()
    result.append(no_dqn_schedule(data, 20))
    # time_spent = time.time() - time_start
    # result.append(no_dqn_schedule(data, 40))
    # result.append(no_dqn_schedule())
    # result.append(no_dqn_schedule(data, 100))
    # result.append(no_dqn_schedule(data, 400))
    # result.append(no_dqn_schedule(seed, 320))
    # result.append(no_dqn_schedule(seed, 320))
    # result.append(no_dqn_schedule(data, 640))
    # result.append(no_dqn_schedule(seed, 640))
    # result.append(no_dqn_schedule(data, 1280))

    global lock, counter, timer, values
    # timer.value += time_spent
    with lock:
        counter.value += 1
        if counter.value % 1 == 0:
            sys.stdout.write('\r{} {}'.format(counter.value, timer.value/(counter.value + 1)))
            sys.stdout.flush()
    return result

def f2(argv):
    state_dict, device, datas, data2s = argv
    if 'dqn' in args.model:
        if 'duel' in args.model:
            model = DQN.GraphNet(hidden_size=args.hidden_size, n_head=8, nlayers=4, duel_dqn=True)
        else:
            model = DQN.GraphNet(hidden_size=args.hidden_size, n_head=8, nlayers=4, duel_dqn=False)
    elif 'IL' in args.model:
        model = model_gnn.GraphNet()
    elif 'RL' in args.model:
        model = A2C.GraphNet(n_head=4, nlayers=2)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    torch.no_grad()
    ret = []
    for data, data2 in zip(datas, data2s):
        ret.append(dqn_schedule(model, [data, data2], device, plan_limit= args.planlimit))
    return ret

def f3(argv):
    state_dict, device, datas, data2s = argv
    if 'dqn' in args.model:
        if 'duel' in args.model:
            model = DQN.GraphNet(hidden_size=args.hidden_size, n_head=8, nlayers=4, duel_dqn=True)
        else:
            model = DQN.GraphNet(hidden_size=args.hidden_size, n_head=8, nlayers=4, duel_dqn=False)
    elif 'IL' in args.model:
        model = model_gnn.GraphNet()
    elif 'RL' in args.model:
        model = A2C.GraphNet(n_head=4, nlayers=2)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    torch.no_grad()
    ret = []
    for data, data2 in zip(datas, data2s):
        ret.append(dqn_schedule(model, [data, data2], device, plan_limit= args.planlimit))
    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000, help="start of seed")
    parser.add_argument("--model", type=str, default='IL_10000', help="start of seed")
    parser.add_argument("--device", type=str, default='cpu', help="device")
    parser.add_argument("--planlimit", type=int, default=20, help="planlimit")
    parser.add_argument("--threads", type=int, default=1, help="threads")
    parser.add_argument("--span", type=int, default=50, help="span")
    parser.add_argument("--mode", type=str, default='dqn')
    parser.add_argument("--sample", type=str, default='uniform')
    parser.add_argument("--pool-num", type=int, default=10)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--package-num", type=int, help='generate RL sample or IL sample')
    parser.add_argument("--time-limit", type=float, help='generate RL sample or IL sample')
    parser.add_argument("--time-interval", type=float, help='generate RL sample or IL sample')
    parser.add_argument("--func-type", type=str, default='uniform', help='generate RL sample or IL sample')
    parser.add_argument("--fn", type=str, default='')
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == 'cpu':
        torch.set_num_threads(58)
    else:
        torch.set_num_threads(1)
    print(args.device)

    if not os.path.exists('output'):
        os.mkdir('output')

    # model = model_gnn.GraphNet()
    #
    # model.load_state_dict()
    # model = model.to(device)
    # model.eval()
    # torch.no_grad()

    define.init()
    define.set_value('package_num', args.package_num)
    define.set_value('time_limit', args.time_limit)
    define.set_value('time_interval', args.time_interval)
    define.set_value('func_type', args.func_type)
    gen_data.generate_data(10000, args.package_num, args.func_type)

    if args.mode == 'no':
        pool = Pool(args.pool_num)
        result = pool.map(f, [gen_data.wrapper(i) for i in range(0, args.span)])
        print(result)
        p = [r[0][0] for r in result]
        s = [r[0][1] for r in result]
        print(np.mean(p, axis=0))
        np.save('output/output_uniform_{}_{}'.format(args.seed, args.span), p)
        pickle.dump([r[0] for r in result], open('solution/no_dqn.pkl', 'wb'))
    elif args.mode == 'dqn':
        state_dict = torch.load('model/model_{}.ckpt'.format(args.model), map_location=torch.device('cpu'))
        datas = [gen_data.wrapper(i) for i in range(0, args.span)]
        data2s = [gen_data.get_data(i) for i in range(0, args.span)]
        print(len(datas), args.span)
        if args.debug:
            result = []
            for line in [[state_dict, device, datas[i:i+args.chunk_size], data2s[i:i+args.chunk_size]] for i in range(0, args.span, args.chunk_size)]:
                result += f2(line)
        else:
            pool = Pool(args.pool_num)
            ret = pool.map(f2,[[state_dict, device, datas[i:i+args.chunk_size], data2s[i:i+args.chunk_size]] for i in range(0, args.span, args.chunk_size)])
            result = []
            for line in ret:
                result += line
        print(np.mean([r[0] for r in result],axis=0))
        print(timer.value/args.span)
        pickle.dump(result, open('solution/dqn.pkl', 'wb'))


    # result = []
    # for seed in range(args.seed, args.seed + args.span):
    #     print('\r{}'.format(seed))
    #     result.append(f(seed))



    # time0 = time.time()
    # workpackages, resources = gen_data.gen_random_data(package_num=define.package_num,seed=seed)
    # schedule = SchedulePolicy(resources, workpackages, None, model, is_dqn=True, time_limit=define.timeLimit, plan_limit=40, time_interval=define.timeInterval, device=device)
    # schedule.greedySchedule()
    # result[-1].append(schedule.get_urgency())
    # schedule.print()
    # print('time spent : {0:.3f}'.format(time.time() - time0))




    # time0 = time.time()
    # workpackages, resources = gen_data.gen_random_data(package_num=define.package_num,seed=seed)
    # schedule = SchedulePolicy(resources, workpackages, None, model, is_dqn=True, time_limit=define.timeLimit, plan_limit=10, time_interval=define.timeInterval, device=device)
    # schedule.greedySchedule()
    # result[-1].append(schedule.get_urgency())
    # schedule.print()
    # print('time spent : {0:.3f}'.format(time.time() - time0))

    # time0 = time.time()
    # workpackages, resources = gen_random_data(package_num=define.package_num,seed=seed)
    # schedule = SchedulePolicy(resources, workpackages, None, model, is_dqn=True, time_limit=define.timeLimit, plan_limit=20, time_interval=define.timeInterval, device=device)
    # schedule.greedySchedule()
    # result[-1].append(schedule.get_urgency())
    # schedule.print()
    # print('time spent : {0:.3f}'.format(time.time() - time0))
    # schedule.vis2('x_{}_dqn'.format(seed))



    #
    #
    # target_net.load_state_dict(policy_net.state_dict())
    # target_net.eval()
    #
    # optimizer = optim.Adam(policy_net.parameters(), lr=define.learning_rate,weight_decay=define.weight_decay)
    #
    # replay_memory = ReplayMemory(define.memory_capacity)
    # workpackages, resources = load()
    # runTime = time.time()
    # schedule = Schedule(resources, workpackages, replay_memory, target_net, is_dqn=True, time_limit=define.timeLimit, plan_limit=define.planLimit, time_interval=define.timeInterval)
    # schedule.greedySchedule()
    # schedule.print()
    # schedule.vis('0')
    #
    # print(schedule.json())
    # for _ in range(2):
    #     print(_)
    #     for i in range(1):
    #         model.optimize_model(policy_net, target_net, optimizer, replay_memory)
    #     target_net.load_state_dict(policy_net.state_dict())
    #
    #     schedule = Schedule(resources, workpackages, replay_memory, target_net, is_dqn=True, time_limit=1, plan_limit=10, time_interval=0.01)
    #     schedule.greedySchedule()
    #     schedule.print()
    #     schedule.vis2('x_{}'.format(_))
    #
    #
    # print("___1")
    # workpackages, resources = gen_random_data(package_num=40,seed=12)
    # schedule = Schedule(resources, workpackages, replay_memory, target_net, is_dqn=False, time_limit=define.timeLimit, plan_limit=define.planLimit, time_interval=define.timeInterval)
    # schedule.greedySchedule()
    # schedule.print()
    #
    # print(schedule.json())
    # for _ in range(2):
    #     print(_)
    #     for i in range(1):
    #         model.optimize_model(policy_net, target_net, optimizer, replay_memory)
    #     target_net.load_state_dict(policy_net.state_dict())
    #
    #     schedule = Schedule(resources, workpackages, replay_memory, target_net, is_dqn=True, time_limit=1, plan_limit=10, time_interval=0.01)
    #     schedule.greedySchedule()
    #     schedule.print()
    # # print(runTime)