import resource_obj
import work_package
import torch
import os
import sys
import define
import pickle
import argparse
import numpy as np
from multiprocessing import Pool, Lock, Value
torch.set_default_tensor_type('torch.FloatTensor')

def feature_extractor_transformer_input(sample):
    return [torch.from_numpy(feature_extractor_transformer(sample_i)).float() for sample_i in sample]

def feature_extractor_transformer(sample):
    resource, work_packages, total_time = sample
    work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency()/p.getWorkingTime()] for p in work_packages]
    work_packages.sort(key=lambda x:x[4])
    work_packages.reverse()

    feature = np.zeros(define.package_num, 2)
    mask = np.zeros((define.package_num))

    for i, p_i in enumerate(work_packages):


    return np.expand_dims(gnn_feature,axis=0), np.expand_dims(mask, axis=0)

lock = Lock()
counter = Value('i', 0)

def print_counter(num=100):
    global lock, counter
    with lock:
        counter.value += 1
        if counter.value % num == 0:
            sys.stdout.write('\r{}/{}'.format(counter.value, args.number))
            sys.stdout.flush()


def feature_extractor_gnn_IL(sample):
    resource, work_packages, total_urgency, total_time = sample
    cnn_feature = feature_extractor_transformer([resource, work_packages, total_time])
    print_counter(100)
    return cnn_feature + (np.array([total_urgency]),)


def feature_extractor_cnn_RL(sample):
    resource, work_packages, total_urgency, total_time = sample
    current_feature = feature_extractor_transformer_input([resource, work_packages, total_time])[0]
    rest_package_features = np.zeros((define.package_num, 7, define.package_num, define.package_num))
    reward_list = np.zeros(define.package_num)
    mask = np.zeros(define.package_num)
    end_label = np.zeros((1,1))

    for i in range(len(work_packages)):
        current_package = work_packages[i]
        mask[i] = 1
        reward_list[i] = current_package.getUrgency()
        tmp_resource = [current_package.getX(), current_package.getY()]
        package_time = current_package.getWorkingTime()
        rest_packages = work_packages[:i] + work_packages[i+1:]
        rest_package_features[i] = feature_extractor_transformer_input([tmp_resource,
                                                               rest_packages,
                                                               max(total_time - package_time, 0)])
    if len(work_packages) == 1:
        end_label[0,0] = 1

    print_counter(10)

    return np.expand_dims(current_feature, axis=0), \
           np.expand_dims(rest_package_features, axis=0), np.expand_dims(reward_list, axis=0), end_label, np.expand_dims(mask, axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--number", type=int,help="number of sample")
    parser.add_argument("--threshold", type=float,help='the threshold of random sampling')
    parser.add_argument("--pool-num", type=int, default=30, help='the threshold of random sampling')
    parser.add_argument("--mode", type=str, default='IL', help='generate RL sample or IL sample')
    args = parser.parse_args()
    data = pickle.load(open('sample/sample_{}_{}.pkl'.format(args.number, args.threshold), 'rb'))
    interval = 1000

    if args.mode == 'IL':
        result = []
        # for i in range(len(data)//interval):
        pool = Pool(args.pool_num)
        result = pool.map(feature_extractor_gnn_IL, data)
        pool.close()
        pool.join()

        # result = []
        # for line in data:
        #     r=feature_extractor_cnn_IL(line)
        #     result.append(r)

        cat_x = [r[0] for r in result]
        cat_y = [r[1] for r in result]

        cat_x = np.concatenate(cat_x, axis=0)
        cat_y = np.concatenate(cat_y, axis=0)

        if not os.path.exists('data'):
            os.mkdir('data')
        np.save('data/gnn_x_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_x)
        np.save('data/gnn_y_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_y)
    elif args.mode == 'RL':
        result = []
        for i in range(len(data)//interval+1):
            pool = Pool(args.pool_num)
            result += pool.map(feature_extractor_gnn_RL, data[interval*i: interval*(i+1)])
            pool.close()
            pool.join()

            print('\r{}\r'.format(i))
        print(counter.value)
        # result = []
        # for line in data:
        #     r=feature_extractor_cnn_RL(line)
        #     result.append(r)
        cat_current = [r[0] for r in result]
        cat_rest = [r[1] for r in result]
        cat_reward = [r[2] for r in result]
        cat_end = [r[3] for r in result]
        cat_mask = [r[4] for r in result]

        cat_current = np.concatenate(cat_current, axis=0)
        cat_rest = np.concatenate(cat_rest, axis=0)
        cat_reward = np.concatenate(cat_reward, axis=0)
        cat_end = np.concatenate(cat_end, axis=0)
        cat_mask = np.concatenate(cat_mask, axis=0)

        if not os.path.exists('data'):
            os.mkdir('data')
        np.save('data/gnn_current_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_current)
        np.save('data/gnn_rest_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_rest)
        np.save('data/gnn_reward_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_reward)
        np.save('data/gnn_end_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_end)
        np.save('data/gnn_mask_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_mask)
    # pickle.dump([cat_x, cat_y], open('data/data_{}_{}.pkl'.format(args.number, args.threshold), 'wb'))


