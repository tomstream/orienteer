import argparse
import os
import pickle
import sys
from multiprocessing import Pool, Lock, Value

import numpy as np
import torch

from utils import define

torch.set_default_tensor_type('torch.FloatTensor')
torch.set_num_threads(1)

def feature_extractor_input(sample, device='cuda:0'):
    features = feature_extractor2(sample)
    ret = [torch.from_numpy(s).to(device).float() for s in features]
    return ret

def paths_to_states(paths):
    return pool.map(path_to_state, [[p.getResource(), p.getWorkPackages(), p.getPackages(), p.getResourceWorkingTime()] for p in paths])

def path_to_state(sample):
    resource, path_packages, packages, working_time = sample
    resource_pos = [resource.getCurrentX(), resource.getCurrentY()]
    id2workpackages = {p.getId():p for p in packages}
    for p in path_packages:
        id2workpackages.pop(p.getId(), None)
    total_time = define.timeLimit - working_time
    return feature_extractor2([resource_pos, list(id2workpackages.values()), total_time])

# def feature_extractor_input(sample, device='cuda:0'):
#     return feature_extractor([sample],device)
#
# def paths_to_states(paths, device, feature_base):
#     result = []
#     for data in [[p.getResource(), p.getWorkPackages(), p.getPackages(), p.getResourceWorkingTime()] for p in paths]:
#         result.append(path_to_state(data))
#     # return pool.map(path_to_state, [[p.getResource(), p.getWorkPackages(), p.getPackages(), p.getResourceWorkingTime()] for p in paths])
#     return feature_extractor(result, device, feature_base)
#
# def path_to_state(sample):
#     resource, path_packages, packages, working_time = sample
#     resource_pos = resource.getCurrentPos()
#     mask = torch.ones(define.package_num,1)
#     for p in packages:
#         p_i = p.getId()
#         mask[p_i] = 0
#
#     total_time = define.timeLimit - working_time
#     return [resource_pos, mask, total_time]


def feature_func(args):
    gnn_feature, mask, i, p_i, total_time, work_packages, resource = args
    for j, p_j in enumerate(work_packages):
        gnn_feature[i, j, 0] = p_i[2]
        gnn_feature[i, j, 1] = p_j[2]
        gnn_feature[i, j, 2] = p_i[3] / max(total_time, 1e-3)
        gnn_feature[i, j, 3] = p_j[3] / max(total_time, 1e-3)
        gnn_feature[i, j, 4] = define.dis(p_i[0], p_j[0], p_i[1], p_j[1]) / define.speed / max(total_time, 1e-3)
        gnn_feature[i, j, 5] = define.dis(resource[0], p_i[0], resource[1], p_i[1]) / define.speed / max(total_time, 1e-3)
        gnn_feature[i, j, 6] = define.dis(resource[0], p_j[0], resource[1], p_j[1]) / define.speed / max(total_time, 1e-3)
        for k in range(2,7):
            if gnn_feature[i, j, k] > 1:
                tmp = gnn_feature[i, j, k] - 1
                gnn_feature[i, j, k] = 2 - np.exp(-tmp)
        mask[i,j] = 1

def feature_extractor_base(sample, device):
    resource, work_packages, total_time = sample
    work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency()/p.getWorkingTime(), p.getId()] for p in work_packages]
    work_packages.sort(key=lambda x:x[5])
    gnn_feature = np.zeros((define.package_num, define.package_num, 5))
    resource_dis = np.zeros(define.package_num)

    for p_i in work_packages:
        i = p_i[5]
        gnn_feature[i, :, 0] = p_i[2]
        gnn_feature[:, i, 1] = p_i[2]
        gnn_feature[i, :, 2] = p_i[3]
        gnn_feature[:, i, 3] = p_i[3]
        resource_dis[i] = define.dis(p_i[0], resource[1], p_i[1], resource[1])
        for p_j in work_packages:
            j = p_j[5]
            gnn_feature[i, j, 4] = define.dis(p_i[0], p_j[0], p_i[1], p_j[1])
    return torch.from_numpy(gnn_feature).to(device), torch.from_numpy(resource_dis).to(device)

def feature_extractor(sample):
    resource, work_packages, total_time = sample
    work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency()/p.getWorkingTime()] for p in work_packages]
    work_packages.sort(key=lambda x:x[4])
    work_packages.reverse()

    gnn_feature = np.zeros((define.package_num, define.package_num, 8))
    mask = np.zeros((define.package_num, define.package_num, 1))

    for i, p_i in enumerate(work_packages):
        for j, p_j in enumerate(work_packages):
            gnn_feature[i, j, 0] = p_i[2]
            gnn_feature[i, j, 1] = p_j[2]
            gnn_feature[i, j, 2] = p_i[3]
            gnn_feature[i, j, 3] = p_j[3]
            gnn_feature[i, j, 4] = define.dis(p_i[0], p_j[0], p_i[1], p_j[1]) / define.speed
            gnn_feature[i, j, 5] = define.dis(resource[0], p_i[0], resource[1], p_i[1]) / define.speed
            gnn_feature[i, j, 6] = define.dis(resource[0], p_j[0], resource[1], p_j[1]) / define.speed
            gnn_feature[i, j, 7] = total_time
            mask[i,j] = 1
    # for i in range(7):
    #     print(np.max(gnn_feature[:,:,i]))
    # input()
    return np.expand_dims(gnn_feature,axis=0), np.expand_dims(mask, axis=0)


def feature_extractortt(sample):
    resource, work_packages, total_time = sample
    work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency()/p.getWorkingTime()] for p in work_packages]
    work_packages.sort(key=lambda x:x[4])
    work_packages.reverse()

    gnn_feature = np.zeros((define.package_num, define.package_num, 7))
    mask = np.zeros((define.package_num, define.package_num, 1))

    for i, p_i in enumerate(work_packages):
        for j, p_j in enumerate(work_packages):
            gnn_feature[i, j, 0] = p_i[2]
            gnn_feature[i, j, 1] = p_j[2]
            gnn_feature[i, j, 2] = p_i[3] / max(total_time, 1e-3)
            gnn_feature[i, j, 3] = p_j[3] / max(total_time, 1e-3)
            gnn_feature[i, j, 4] = define.dis(p_i[0], p_j[0], p_i[1], p_j[1]) / define.speed / max(total_time, 1e-3)
            gnn_feature[i, j, 5] = define.dis(resource[0], p_i[0], resource[1], p_i[1]) / define.speed / max(total_time, 1e-3)
            gnn_feature[i, j, 6] = define.dis(resource[0], p_j[0], resource[1], p_j[1]) / define.speed / max(total_time, 1e-3)
            for k in range(2,7):
                if gnn_feature[i, j, k] > 1:
                    tmp = gnn_feature[i, j, k] - 1
                    gnn_feature[i, j, k] = 2 - np.exp(-tmp)
            mask[i,j] = 1
    # for i in range(7):
    #     print(np.max(gnn_feature[:,:,i]))
    # input()
    return np.expand_dims(gnn_feature,axis=0), np.expand_dims(mask, axis=0)

def feature_extractorxx(samples, device, feature_base):
    gnn_features = torch.zeros(len(samples), define.package_num, define.package_num, 7).to(device)
    gnn_features[:,:, :, :5] = feature_base[0].unsqueeze(0)
    resource_diss = []
    mask2ds = []

    for index, sample in enumerate(samples):
        resource, mask2d, total_time = sample

        gnn_feature = gnn_features[index]
        if resource[2] == -1:
            resource_dis = feature_base[1]
        else:
            resource_dis = feature_base[0][resource[2],:,1]
        resource_diss.append(resource_dis.unsqueeze(0))
        mask2d = 1 - mask2d.to(device)
        div = max(total_time, 1e-3) * define.speed
        gnn_feature[:,:,2:] /= div
        mask2ds.append(mask2d.unsqueeze(0))
        # mask3d = (1 - (mask2d + mask2d.transpose(1,0)).bool().float()).unsqueeze(0)
        # masks.append(mask3d)
    mask2ds = torch.cat(mask2ds, dim=0).unsqueeze(-1)
    masks = (1-(mask2ds + mask2ds.transpose(2,1)).bool().float())
    resource_diss = torch.cat(resource_diss, dim=0)
    gnn_features[:,:,:,5] = resource_diss.unsqueeze(-1)
    # masks = torch.cat(masks,dim=0).unsqueeze(-1)
    gnn_features[:,:,:,6] = gnn_features[:,:,:,5].transpose(2,1)
    tmp = gnn_features[:, :, :, 2:]
    tmp_mask = (tmp > 1).float()
    gnn_features[:, :, :, 2:] = (1 - tmp_mask) * tmp + tmp_mask * (2 - torch.exp(1 - tmp))
    # print(torch.sum(gnn_features),'f@@K')
    # print(torch.sum(masks))
    # input()
    return gnn_features, masks


def feature_extractor_torch(samples, device, feature_base):
    gnn_features = torch.zeros(len(samples), define.package_num, define.package_num, 7).to(device)
    masks = torch.ones(len(samples), define.package_num, define.package_num, 1).to(device)
    for index, sample in enumerate(samples):
        resource, work_packages, total_time = sample
        work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency()/p.getWorkingTime(), p.getId()] for p in work_packages]

        gnn_feature = gnn_features[index]
        mask = masks[index]
        mask2d = torch.ones(define.package_num, 1)
        dis_vec = torch.zeros(define.package_num, 1, 2)
        resource_pos = torch.zeros(define.package_num, define.package_num, 2).to(device)
        resource_pos[:,:,0] = resource[0]
        resource_pos[:,:,1] = resource[1]

        for p_i in work_packages:
            i = p_i[5]
            gnn_feature[i, :, 0] = p_i[2]
            gnn_feature[i, :, 2] = p_i[3]
            mask2d[i] = 0
            dis_vec[i, 0, 0] = p_i[0]
            dis_vec[i, 0, 1] = p_i[1]

        mask2d = mask2d.to(device)
        dis_vec = dis_vec.to(device)

        dis_matrix = dis_vec.repeat(1, define.package_num, 1)
        dis_matrix_transpose = dis_matrix.permute(1,0,2)
        div = max(total_time, 1e-3) * define.speed
        tmp = dis_matrix - dis_matrix_transpose
        gnn_feature[:,:,4] = torch.sqrt(torch.sum(tmp*tmp, dim=2))/div
        tmp = dis_matrix - resource_pos
        gnn_feature[:,:,6] = torch.sqrt(torch.sum(tmp*tmp, dim=2))/div

        mask3d = 1 - (mask2d + mask2d.transpose(1,0)).bool().float()
        mask[:,:,0] = mask3d

    gnn_features[:,:,:,1] = gnn_features[:,:,:,0].permute(0,2,1)
    gnn_features[:,:,:,3] = gnn_features[:,:,:,2].permute(0,2,1)
    gnn_features[:,:,:,5] = gnn_features[:,:,:,4].permute(0,2,1)
    tmp = gnn_features[:,:,:,2:]
    tmp_mask = (tmp > 1).float()
    gnn_features[:,:,:,2:] = (1 - tmp_mask) * tmp + tmp_mask * (2 - torch.exp(1 - tmp))

def feature_extractor_numpy(samples):
    gnn_features = np.zeros((len(samples), define.package_num, define.package_num, 7))
    masks = np.ones((len(samples), define.package_num, define.package_num, 1))
    for index, sample in enumerate(samples):
        resource, work_packages, total_time = sample
        work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency()/p.getWorkingTime(), p.getId()] for p in work_packages]

        gnn_feature = gnn_features[index]
        mask = masks[index]
        mask2d = np.ones((define.package_num, 1))
        dis_vec = np.zeros((define.package_num, 1, 2))
        resource_pos = np.zeros((define.package_num, define.package_num, 2))
        resource_pos[:,:,0] = resource[0]
        resource_pos[:,:,1] = resource[1]

        for p_i in work_packages:
            i = p_i[5]
            gnn_feature[i, :, 0] = p_i[2]
            gnn_feature[i, :, 2] = p_i[3]
            mask2d[i] = 0
            dis_vec[i, 0, 0] = p_i[0]
            dis_vec[i, 0, 1] = p_i[1]

        dis_matrix = dis_vec.repeat(define.package_num, axis=1)
        dis_matrix_transpose = dis_matrix.transpose((1,0,2))
        gnn_feature[:,:,4] = np.sqrt(np.sum(np.square(dis_matrix - dis_matrix_transpose), axis=2))
        gnn_feature[:,:,5] = np.sqrt(np.sum(np.square(dis_matrix - resource_pos), axis=2))

        mask3d = 1 - (mask2d + mask2d.transpose())>0
        mask[:,:,0] = mask3d

        gnn_feature[:,:,4:] /= define.speed
        gnn_feature[:,:,1] = gnn_feature[:,:,0].transpose((1,0))
        gnn_feature[:,:,3] = gnn_feature[:,:,2].transpose((1,0))
        gnn_feature[:,:,6] = gnn_feature[:,:,5].transpose((1,0))
        tmp = gnn_feature[:,:,2:] / max(total_time, 1e-3)
        tmp_mask = tmp > 1
        gnn_feature[:,:,2:] = (1 - tmp_mask) * tmp + tmp_mask * (2 - np.exp(1 - tmp))


    # for i in range(7):
    #     print(np.max(gnn_feature[:,:,i]))
    # input()
    return gnn_features, masks

def feature_extractor2(sample):
    resource, work_packages, total_time, end_axis = sample
    work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency(), p.getId()] for p in work_packages]
    gnn_feature = np.zeros((define.get_value('package_num'), define.get_value('package_num'), 8))
    mask = np.zeros((define.get_value('package_num'), define.get_value('package_num'), 1))

    for p_i in work_packages:
        i = p_i[5]
        for p_j in work_packages:
            j = p_j[5]
            gnn_feature[i, j, 0] = p_i[2]
            gnn_feature[i, j, 1] = p_j[2]
            # gnn_feature[i, j, 2] = p_i[3]
            # gnn_feature[i, j, 3] = p_j[3]
            gnn_feature[i, j, 2] = define.dis(end_axis[0], p_i[0], end_axis[1], p_i[1]) / define.get_value('speed')
            gnn_feature[i, j, 3] = define.dis(end_axis[0], p_j[0], end_axis[1], p_j[1]) / define.get_value('speed')
            gnn_feature[i, j, 4] = define.dis(p_i[0], p_j[0], p_i[1], p_j[1]) / define.get_value('speed') + p_j[3]
            gnn_feature[i, j, 5] = define.dis(resource[0], p_i[0], resource[1], p_i[1]) / define.get_value('speed') + p_i[3]
            gnn_feature[i, j, 6] = define.dis(resource[0], p_j[0], resource[1], p_j[1]) / define.get_value('speed') + p_j[3]
            gnn_feature[i, j, 7] = total_time
            mask[i,j] = 1
    # for i in range(7):
    #     print(np.max(gnn_feature[:,:,i]))
    # input()
    return np.expand_dims(gnn_feature,axis=0), np.expand_dims(mask, axis=0)

def feature_extractor2_numpy(sample):
    resource, work_packages, total_time = sample
    work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency()/p.getWorkingTime(), p.getId()] for p in work_packages]
    gnn_feature = np.zeros((define.package_num, define.package_num, 3))
    mask = np.zeros((define.package_num, define.package_num, 1))
    node_feature = np.zeros((define.package_num, 3))


    for p_i in work_packages:
        i = p_i[5]
        node_feature[i, 0] = p_i[2]
        node_feature[i, 1] = p_i[3]
        node_feature[i, 2] = total_time
        for p_j in work_packages:
            j = p_j[5]
            gnn_feature[i, j, 4] = define.dis(p_i[0], p_j[0], p_i[1], p_j[1]) / define.speed
            gnn_feature[i, j, 5] = define.dis(resource[0], p_i[0], resource[1], p_i[1]) / define.speed + p_i[3]
            gnn_feature[i, j, 6] = define.dis(resource[0], p_j[0], resource[1], p_j[1]) / define.speed + p_j[3]
            mask[i,j] = 1
    # for i in range(7):
    #     print(np.max(gnn_feature[:,:,i]))
    # input()
    return np.expand_dims(gnn_feature,axis=0), np.expand_dims(mask, axis=0), np.expand_dims(node_feature, axis=0)

def feature_extractor_parallel(sample):
    resource, work_packages, total_time = sample
    work_packages = [[p.getX(), p.getY(), p.getUrgency(),p.getWorkingTime(), p.getUrgency()/p.getWorkingTime()] for p in work_packages]
    work_packages.sort(key=lambda x:x[4])
    work_packages.reverse()

    gnn_feature = np.zeros((define.package_num, define.package_num, 7))
    mask = np.zeros((define.package_num, define.package_num, 1))
    data_in = [[gnn_feature, mask, i, p_i, total_time, work_packages, resource] for i, p_i in enumerate(work_packages)]
    pool.map(feature_func,data_in)
    return np.expand_dims(gnn_feature,axis=0), np.expand_dims(mask, axis=0)

lock = Lock()
counter = Value('i', 0)

def print_counter(num=100):
    global lock, counter
    with lock:
        counter.value += 1
        if counter.value % num == 0:
            sys.stdout.write('\r{}'.format(counter.value))
            sys.stdout.flush()


def feature_extractor_IL(sample):
    resource, work_packages, total_urgency, total_time, pid = sample
    cnn_feature = feature_extractor2([resource, work_packages, total_time])
    print_counter(100)
    return cnn_feature[0], cnn_feature[1], np.array([total_urgency]), np.array([pid])


def feature_extractor_RL(sample):
    resource, work_packages, total_urgency, total_time = sample
    current_feature = feature_extractor_input([resource, work_packages, total_time])[0]
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
        rest_package_features[i] = feature_extractor_input([tmp_resource,
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
    parser.add_argument("--limit", type=int, default=-1, help="limit of feature")
    parser.add_argument("--threshold", type=str,help='the threshold of random sampling')
    parser.add_argument("--pool-num", type=int, default=30, help='the threshold of random sampling')
    parser.add_argument("--mode", type=str, default='IL', help='generate RL sample or IL sample')
    args = parser.parse_args()
    data = pickle.load(open('sample/sample_{}_{}.pkl'.format(args.number, args.threshold), 'rb'))
    if args.limit > 0:
        data = data[:args.limit]
    interval = 1000

    if args.mode == 'IL':
        pool = Pool(args.pool_num)
        result = pool.map(feature_extractor_IL, data)
        pool.close()
        pool.join()

        # result = []
        # for line in data:
        #     r=feature_extractor_IL(line)
        #     result.append(r)

        cat_feature = [r[0] for r in result]
        cat_mask = [r[1] for r in result]
        cat_urgency = [r[2] for r in result]
        cat_pid = [r[3] for r in result]

        cat_feature = np.concatenate(cat_feature, axis=0)
        cat_mask = np.concatenate(cat_mask, axis=0)
        cat_urgency = np.concatenate(cat_urgency, axis=0)
        cat_pid = np.concatenate(cat_pid, axis=0)

        if not os.path.exists('data'):
            os.mkdir('data')
        np.save('data/gnn_feature_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_feature)
        np.save('data/gnn_mask_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_mask)
        np.save('data/gnn_urgency_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_urgency)
        np.save('data/gnn_id_{}_{}_{}'.format(args.mode, args.number, args.threshold), cat_pid)

    elif args.mode == 'RL':
        result = []
        for i in range(len(data)//interval+1):
            pool = Pool(args.pool_num)
            result += pool.map(feature_extractor_RL, data[interval*i: interval*(i+1)])
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
else:
    pool = Pool(20)

