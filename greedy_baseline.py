import math
import copy
import queue
import time
import memory
import define
from path_obj import Path
import json
import torch
import feature_extractor_gnn as feature_extractor
import model_gnn
import os
import gen_data
import numpy as np
import sys


class Schedule(object):
    def __init__(self, resources, workpackages, replay_memory, net=None, time_limit=480, device='cuda:0'):
        self.__totalUrgency = 0
        self.__paths = []
        self.__resources = []
        self.__workPackages = []
        self.__unsolvedWorkPackages = []
        self.__resources = copy.deepcopy(resources)
        self.__workPackages = copy.deepcopy(workpackages)
        self.__timeLimit = time_limit
        self.__memory = replay_memory
        self.__net = net
        self.__device = torch.device(device)

    def updateResourcesWorkPackages(self, resourceId, existWorkPackages):
        if resourceId != None:
            for i in range(len(self.__resources)):
                if self.__resources[i].getId() == resourceId:
                    del self.__resources[i]
                    break
        newWorkPackages = []
        for i in range(len(self.__workPackages)):
            if self.__workPackages[i].getId() in existWorkPackages:
                continue
            newWorkPackages.append(self.__workPackages[i])
        self.__workPackages = newWorkPackages

    def greedySchedule(self):
        currentResource = self.__resources[0].copy()
        nowPath = Path(currentResource, self.__workPackages, net=None, device=self.__device)
        # nowPath.cal_q_score()
        # print(nowPath.get_q())
        while True:
            path_list = []
            idx = -1
            value = -1
            for i in range(define.package_num):
                currentWorkPackage = self.__workPackages[i].copy()
                if nowPath.exist(currentWorkPackage.getId()):
                    continue
                times = nowPath.getResourceNeedTime(currentWorkPackage) + nowPath.getResourceWorkingTime()
                urgency = currentWorkPackage.getUrgency()
                if times <= self.__timeLimit:
                    ratio = urgency/times
                    if ratio > value:
                        idx = i
                        value = ratio
            if idx!=-1:
                currentWorkPackage = self.__workPackages[idx].copy()
                nowPath.addWorkPackage(currentWorkPackage)
                times = nowPath.getResourceNeedTime(currentWorkPackage) + nowPath.getResourceWorkingTime()
                nowPath.setResourceWorkingTime(times)
                nowPath.setResourcePosition(currentWorkPackage.getX(), currentWorkPackage.getY(), currentWorkPackage.getId())
            else:
                break
            # data = [path.to_state() for path in path_list]
            # features = torch.cat([d[0] for d in data], axis=0)
            # masks = torch.cat([d[1] for d in data], axis=0)

            # values, scores = self.__net(features, masks)

        self.__paths.append(nowPath)
        self.__totalUrgency = nowPath.getTotalUrgency()

    def get_urgency(self):
        return self.__totalUrgency

    def print(self):
        tmpUrgency = 0
        tmpLen = 0
        for i in range(len(self.__paths)):
            tmpUrgency += self.__paths[i].getTotalUrgency()
            tmpLen += self.__paths[i].getWorkPackageNum()
        print("totalUrgency {0:.3f}, len {1:.3f}".format(self.__totalUrgency, tmpLen))

    def getPath(self):
        return self.__paths
    def json(self):
        ret = []
        for i in range(len(self.__paths)):
            path = self.__paths[i]
            ret.append({"resourceId":path.getResourceId(), "workPackageIds":list(path.getExistWorkPackages().keys())})
        return json.dumps(ret)
    def vis(self, name):
        path = self.getPath()[0]
        path.vis(name)
    def vis2(self, name):
        path = self.getPath()[0]
        path.vis2(name)
    def urgency(self):
        return self.__totalUrgency



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000, help="start of seed")
    parser.add_argument("--model", type=str, default='IL_100000', help="start of seed")
    parser.add_argument("--device", type=str, default='cuda:0', help="device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(args.device)

    if not os.path.exists('output'):
        os.mkdir('output')
    result = []
    for i in range(100):
        workpackages, resources = gen_data.gen_random_data(package_num=define.package_num,seed=i)
        schedule = Schedule(resources, workpackages, None, net=None, time_limit=define.timeLimit, device='cpu')
        time0 = time.time()
        schedule.greedySchedule()
        result.append(schedule.urgency())
    print(np.mean(result))


