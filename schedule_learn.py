import copy
import json
import math
import queue

import torch

from utils import define
from utils.path_obj import Path


class Schedule(object):
    def __init__(self, resources, workpackages, replay_memory, net=None, is_dqn=False, time_limit=480, plan_limit=100, time_interval=60, global_plan_limit=define.globalPlanLimit, device='cuda:0',
                 batch_size = 10, data=None):
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
        self.__batch_size = batch_size
        self.__data = data


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
            currentPath = Path(currentResource, self.__workPackages, self.__net, self.__is_dqn, self.__device)
            series = []
            for t in range(2 + math.floor(self.__timeLimit/self.__timeInterval)):
                series.append(queue.PriorityQueue())
            series[0].put(Path(currentResource, self.__workPackages, self.__net, self.__is_dqn, self.__device))
            for t in range(2 + math.floor(self.__timeLimit/self.__timeInterval)):
                while series[t].qsize() != 0:
                    paths = []
                    for _path_idx in range(self.__batch_size):
                        if series[t].qsize() != 0:
                            paths.append(series[t].get())
                    features, masks = [torch.cat(x,dim=0) for x in zip(*[path.to_state() for path in paths])]
                    values = self.__net(features, masks).cpu().detach().numpy()
                    del features, masks
                    time_with_path = []
                    for _path_idx, nowPath in enumerate(paths):
                        if nowPath.getTotalUrgency() > currentPath.getTotalUrgency():
                            currentPath = nowPath
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
                                newPath.set_q(values[_path_idx][i]-currentWorkPackage.getUrgency())
                                time_with_path.append([newTime, newPath])

                    if len(time_with_path)==0:
                        continue
                    # features, masks = [torch.cat(x,dim=0) for x in zip(*[path.to_state() for t, path in time_with_path])]
                    # values = np.max(self.__net(features, masks).cpu().detach().numpy(), axis=1)
                    # del features, masks
                    for l, (newTime, newPath) in enumerate(time_with_path):
                        # newPath.set_q(values[l])
                        series[newTime].put(newPath)
                        while series[newTime].qsize() > self.__planLimit:
                            tmp = series[newTime].get()
                            del tmp
            curr_wp = currentPath.getWorkPackage()
            self.path_idx = [_w.getId() for _w in curr_wp]
            # print(self.path_idx)
            # else_idx = set(list(range(define.get_value('package_num')))) - set(idx)
            # init = idx + list(else_idx)
            # _, result = local_search.SA_solver(self.__data[0], self.__data[1], self.__data[2], init)
            #
            # print('')
            # print(result-currentPath.getTotalUrgency())
            # print('')

            self.__paths.append(currentPath)
            self.updateResourcesWorkPackages(currentPath.getResourceId(), currentPath.getExistWorkPackages())
            self.__totalUrgency += currentPath.getTotalUrgency()

        for i in range(len(self.__workPackages)):
            self.__unsolvedWorkPackages.append(self.__workPackages[i].copy())
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