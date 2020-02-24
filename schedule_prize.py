import copy
import json
import math
import queue
import time

import torch

import utils.feature_extractor_gnn as feature_extractor
from utils import define
from utils.path_obj import Path


class Schedule(object):
    def __init__(self, resources, workpackages, replay_memory, net=None, is_dqn=False, time_limit=480, plan_limit=100, time_interval=60, global_plan_limit=define.globalPlanLimit, device='cuda:0'):
        self.__totalUrgency = 0
        self.__paths = []
        self.__resources = []
        self.__workPackages = []
        self.__unsolvedWorkPackages = []
        self.__resources = copy.deepcopy(resources)
        self.__workPackages = copy.deepcopy(workpackages)
        self.__globalPlanLimit = global_plan_limit
        self.__timeLimit = time_limit
        self.__planLimit = plan_limit
        self.__timeInterval = time_interval
        self.__memory = replay_memory
        self.__net = net
        self.__is_dqn = is_dqn
        self.__device = torch.device(device)
        self.__feature_base = None

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
        while len(self.__resources) > 0:
            currentResource = self.__resources[0].copy()
            if self.__is_dqn:
                self.__feature_base = feature_extractor.feature_extractor_base([currentResource.getCurrentPos(), self.__workPackages, define.timeLimit], self.__device)
            currentPath = Path(currentResource, self.__workPackages, self.__net, self.__is_dqn, self.__device,self.__feature_base)
            series = []
            for t in range(2 + math.floor(self.__timeLimit/self.__timeInterval)):
                series.append(queue.PriorityQueue())
            series[0].put(Path(currentResource, self.__workPackages, self.__net, self.__is_dqn, self.__device,self.__feature_base))
            globalPriorityQueue = queue.PriorityQueue()
            for t in range(2 + math.floor(self.__timeLimit/self.__timeInterval)):
                while series[t].qsize() != 0:
                    time_with_path = []
                    nowPath = series[t].get()

                    if nowPath.getTotalUrgency() > currentPath.getTotalUrgency():
                        currentPath = nowPath
                    globalPriorityQueue.put(nowPath)
                    while globalPriorityQueue.qsize() > self.__globalPlanLimit:
                        globalPriorityQueue.get()

                    for i in range(len(self.__workPackages)):
                        currentWorkPackage = self.__workPackages[i].copy()
                        if nowPath.exist(currentWorkPackage.getId()):
                            continue
                        times = nowPath.getResourceNeedTime(currentWorkPackage) + nowPath.getResourceWorkingTime()
                        if times + nowPath.getReturnTime(currentWorkPackage) <= self.__timeLimit:
                            newTime = math.ceil(times/self.__timeInterval)
                            newPath = nowPath.copy()
                            newPath.addWorkPackage(currentWorkPackage.copy())
                            newPath.setResourceWorkingTime(times)
                            newPath.setResourcePosition(currentWorkPackage.getX(), currentWorkPackage.getY(), currentWorkPackage.getId())
                            time_with_path.append([newTime, newPath])
                    if self.__is_dqn and len(time_with_path)>0:
                        timex = time.time()
                        features, masks = feature_extractor.paths_to_states([newPath for newTime, newPath in time_with_path], self.__device, self.__feature_base)
                        timey = time.time()
                        values = self.__net(features, masks).cpu().detach().numpy()[:,0]
                        timez = time.time()
                        # print(timey-timex, timez-timey)
                        # input()
                        # print("{:.3f}".format(time.time()-time0))
                        for i in range(len(values)):
                            time_with_path[i][1].set_q(values[i])

                    for newTime, newPath in time_with_path:
                        series[newTime].put(newPath)
                        while series[newTime].qsize() > self.__planLimit:
                            series[newTime].get()
            self.__paths.append(currentPath)
            self.updateResourcesWorkPackages(currentPath.getResourceId(), currentPath.getExistWorkPackages())
            self.__totalUrgency += currentPath.getTotalUrgency()
        for i in range(len(self.__workPackages)):
            self.__unsolvedWorkPackages.append(self.__workPackages[i].copy())
        while not globalPriorityQueue.empty() and self.__memory is not None:
            path = globalPriorityQueue.get()
            memory = path.pathToState()
            for m in memory:
                self.__memory.append(m)
        curr_wp = currentPath.getWorkPackage()
        self.path_idx = [_w.getId() for _w in curr_wp]

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
        # totalPath = 0
        # totalWorkPackage = 0
        # for i in range(len(self.__paths)):
        #     if self.__paths[i].getWorkPackageSize() != 0:
        #         totalPath += 1
        #         totalWorkPackage += self.__paths[i].getWorkPackageSize()
        # print("totalPath " + str(totalPath))
        # print("totalWorkPackage " + str(totalWorkPackage))
        #
        # for i in range(len(self.__paths)):
        #     print("path" + str(i))
        #     self.__paths[i].print()
        #
        # print(str(len(self.__unsolvedWorkPackages)) + " unsolve workpackages")
        # for i in range(len(self.__unsolvedWorkPackages)):
        #     self.__unsolvedWorkPackages[i].print()