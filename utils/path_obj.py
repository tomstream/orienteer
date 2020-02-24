import copy

import matplotlib as mpl
import torch

from utils import define

mpl.use('Agg')
mpl.rc('pdf', fonttype=42)
import matplotlib.pyplot as plt
import numpy as np
import os
import utils.feature_extractor_gnn as feature_extractor


class Path(object):
    def __init__(self, resource, packages, net=None, is_dqn=None, device='cpu', base=None, time_limit = None,
                 dis_matrix=None, feature=None):
        self.__resource = copy.deepcopy(resource)
        self.__workPackages = []
        self.__totalUrgency = 0
        self.__existWorkPackages = {}
        self.__is_dqn = is_dqn
        self.__net = net
        self.__packages = packages
        self.device = device
        self.dis_matrix = None
        self.q = None
        self.score = None
        self.base = base
        self.dis_matrix = dis_matrix
        self.q = None
        self.score = None
        self.base = base
        self.feature = feature
        self.all_urgency = np.array([wk.getUrgency() for wk in packages])
        self.package_num = define.get_value('package_num')
        self.time_limit = define.get_value('time_limit')
        self.speed = define.get_value('speed')
        if self.__net is not None and dis_matrix is None:
            tmp_X = np.zeros((self.package_num, self.package_num, 2))
            tmp_Y = np.zeros((self.package_num, self.package_num, 2))
            tmp_time = np.zeros((self.package_num, self.package_num))

            for wk in packages:
                tmp_X[wk.getId(), :, 0] = wk.getX()
                tmp_X[wk.getId(), :, 1] = wk.getY()
                tmp_Y[:, wk.getId(), 0] = wk.getX()
                tmp_Y[:, wk.getId(), 1] = wk.getY()
                tmp_time[:, wk.getId()] = wk.getWorkingTime()
            self.dis_matrix = np.sqrt(np.sum(np.square(tmp_X -tmp_Y),axis=2))/self.speed + tmp_time
            self.feature = feature_extractor.feature_extractor2([[resource.getInitialX(), resource.getInitialY()],
                                                                 packages,
                                                                 self.time_limit, [resource.getInitialX(), resource.getInitialY()]])

    def set_q(self, q):
        self.q = np.max(q)

    def get_q(self):
        return self.q

    def set_score(self, score):
        self.score = [[i,s] for i,s in enumerate(score)]
        self.score.sort(key=lambda x:x[1], reverse=True)

    def get_score(self):
        return self.score

    def set_dqn(self, is_dqn):
        self.__is_dqn = is_dqn

    def setResource(self, resource):
        self.__resource = resource.copy()
    
    def setWorkPackages(self, WorkPackages):
        self.__workPackages = []
        for item in WorkPackages:
            self.__workPackages.append(item.copy())
    
    def setTotalUrgency(self, totalUrgency):
        self.__totalUrgency = totalUrgency
    
    def setExistWorkPackages(self, existWorkPackages):
        self.__existWorkPackages = {}
        for key in existWorkPackages:
            self.__existWorkPackages[key] = 1
    
    def copy(self):
        tmp = Path(self.__resource, self.__packages, self.__net, self.__is_dqn, self.device,self.base, dis_matrix=self.dis_matrix, feature=copy.deepcopy(self.feature))
        tmp.setResource(self.__resource)
        tmp.setWorkPackages(self.__workPackages)
        tmp.setTotalUrgency(self.__totalUrgency)
        tmp.setExistWorkPackages(self.__existWorkPackages)
        tmp.all_urgency = self.all_urgency.copy()
        return tmp

    def setResourceWorkingTime(self, workingTime):
        self.__resource.setWorkingTime(workingTime)
        if self.__net is not None:
            feature = self.feature[0][0,:,:,7]
            feature[feature!=0] = self.time_limit - workingTime

    def setResourcePosition(self, x, y, pid):
        self.__resource.setPosition(x, y, pid)

    def cal_q_score(self):
        if self.q is None:
            self_state = self.to_state()
            ret = self.__net(*self_state)
            if len(ret) != 2:
                self.set_q(ret.cpu().detach().numpy()[0])
                self.set_score([0]*40)
            else:
                self.set_q(ret[1][0])
                self.set_score(ret[0].cpu().detach().numpy()[0])

    def __lt__(self, other):
        if not self.__is_dqn:
            return self.__totalUrgency < other.__totalUrgency

        self.cal_q_score()
        other.cal_q_score()

        # if isinstance(self.q, np.ndarray):
        #     return self.__totalUrgency + np.max(self.q + self.all_urgency)  <  other.__totalUrgency + np.max(other.q + self.all_urgency)
        # else:
        return self.__totalUrgency + self.q  <  other.__totalUrgency + other.q

    def getReturnTime(self, p):
        end_x, end_y = self.__resource.getInitialPos()
        p_x, p_y = p.getPosition()
        t = define.dis(end_x, p_x, end_y, p_y) / self.speed
        return t

    def getResourceId(self):
        return self.__resource.getId()

    def getResource(self):
        return self.__resource.copy()

    def getWorkPackages(self):
        return copy.deepcopy(self.__workPackages)

    def getWorkPackageId(self):
        return [_.getId() for _ in self.__workPackages]

    def getPackages(self):
        return self.__packages

    def getWorkPackageNum(self):
        return len(self.__workPackages)

    def getTotalUrgency(self, recompute=False):
        if not recompute:
            return self.__totalUrgency
        else:
            ret = 0
            for p in self.__workPackages:
                ret += p.getUrgency()
            return ret

    def getResourceSolveable(self, workPackage):
        return self.__resource.solveable(workPackage)

    def getResourceNeedTime(self, workPackage):
        return self.__resource.needTime(workPackage)
    
    def getResourceWorkingTime(self):
        return self.__resource.getWorkingTime()

    def addWorkPackage(self, workPackage):
        self.__totalUrgency += workPackage.getUrgency()
        self.__workPackages.append(workPackage)
        self.__existWorkPackages[workPackage.getId()] = 1
        if self.__net is not None:
            feature = self.feature[0][0]
            mask = self.feature[1][0]
            wid = workPackage.getId()
            feature[:,:,5] = np.expand_dims(self.dis_matrix[wid],axis=1)
            feature[:,:,6] = feature[:,:,5].transpose(1,0)
            mask[wid, :] = 0
            mask[:, wid] = 0
            feature[wid,:] = 0
            feature[:, wid] = 0

    
    def exist(self, id):
        return id in self.__existWorkPackages
    
    def getExistWorkPackages(self):
        return self.__existWorkPackages
    
    def getWorkPackageSize(self):
        return len(self.__workPackages)

    def getWorkPackage(self):
        return self.__workPackages

    def to_state1(self):
        resource = self.__resource
        resource_pos = [resource.getCurrentX(), resource.getCurrentY()]
        path_packages = self.getWorkPackages()
        id2workpackages = {p.getId():p for p in self.__packages}
        for p in path_packages:
            id2workpackages.pop(p.getId(), None)
        total_time = self.timeLimit - self.getResourceWorkingTime()
        return feature_extractor.feature_extractor_input([resource_pos, list(id2workpackages.values()), total_time], device=self.device)

    def to_state(self, device=None):
        device = self.device if device is None else device
        return [torch.from_numpy(self.feature[0]).to(device).float(), torch.from_numpy(self.feature[1]).to(device).float()]



    def pathToState(self):
        memory = []
        lastPackages = []
        resource = self.__resource
        resource_pos = [resource.getInitialX(), resource.getInitialY()]
        resource_time = 0
        for i in range(len(self.__workPackages)):
            wk = self.__workPackages[i]
            reward = wk.getUrgency()
            current_state = self.featureExtractor(resource_pos, resource_time, lastPackages, self.__packages)
            lastPackages.append(wk)
            resource_pos[0] = wk.getX()
            resource_pos[1] = wk.getY()
            resource_time += wk.getWorkingTime()
            next_state = self.featureExtractor(resource_pos, resource_time, lastPackages, self.__packages)
            memory.append([current_state, wk.getId(), next_state, reward])
        return memory

    def vis2(self, name):
        plt.xlim([0,1])
        plt.ylim([0,1])
        x = []
        y = []
        s = []
        for package in self.__packages:
            x.append(package.getX())
            y.append(package.getY())
            s.append((package.getUrgency()-0.5)*20)
        plt.scatter(x,y,s,color='k',)

        for i in range(len(self.__workPackages)):
            if i == 0:
                plt.plot([self.__resource.getInitialX(),self.__workPackages[0].getX()],[self.__resource.getInitialY(), self.__workPackages[0].getY()],linewidth=1,color='b')
            else:
                plt.plot([self.__workPackages[i-1].getX(),self.__workPackages[i].getX()],[self.__workPackages[i-1].getY(),self.__workPackages[i].getY()],linewidth=1,color='b')

        lastPackages = []
        resource = self.__resource
        resource_pos = [resource.getInitialX(), resource.getInitialY()]
        resource_time = 0
        current_urgency = 0
        id2workpackages = {p.getId():p for p in self.__packages}
        total_time = self.timeLimit

        for i in range(len(self.__workPackages)):
            wk = self.__workPackages[i]
            reward = wk.getUrgency()
            current_urgency += reward
            current_state = feature_extractor.feature_extractor_input([resource_pos, list(id2workpackages.values()), total_time], self.device)
            current_reward = self.__net(current_state[0], current_state[1])
            plt.annotate('%.2f %.2f %.2f'%(current_urgency, current_reward, current_reward + current_urgency), resource_pos, fontsize=4)
            id2workpackages.pop(wk.getId(),0)
            total_time -= wk.getWorkingTime()
            resource_pos[0] = wk.getX()
            resource_pos[1] = wk.getY()
        if not os.path.exists('pic'):
            os.mkdir('pic')
        plt.savefig('pic/result_{}.pdf'.format(name))
        plt.close('all')

    def print(self):
        print("resource ")
        if self.__resource != None:
            self.__resource.print()
        print(str(len(self.__workPackages)) + "workpackages")
        sum = 0
        for i in range(len(self.__workPackages)):
            if i == 0:
                sum += self.__workPackages[0].dist(self.__resource.getInitialX(), self.__resource.getInitialY()) / self.__resource.getSpeed() + self.__workPackages[0].getWorkingTime()
            else:
                sum += self.__workPackages[i].dist(self.__workPackages[i - 1].getX(), self.__workPackages[i - 1].getY()) / self.__resource.getSpeed() + self.__workPackages[i].getWorkingTime()
            print("workpackage" + str(i))
            self.__workPackages[i].print()
        print("workingtime {0:.3f}".format(self.__resource.getWorkingTime()) + "     check workingtime " + str(sum))
    def vis(self, name):
        plt.xlim([0,1])
        plt.ylim([0,1])
        x = []
        y = []
        s = []
        for package in self.__packages:
            x.append(package.getX())
            y.append(package.getY())
            s.append((package.getUrgency()-0.5)*20)
        plt.scatter(x,y,s,color='k',)

        for i in range(len(self.__workPackages)):
            if i == 0:
                plt.plot([self.__resource.getInitialX(),self.__workPackages[0].getX()],[self.__resource.getInitialY(), self.__workPackages[0].getY()],linewidth=1,color='b')
            else:
                plt.plot([self.__workPackages[i-1].getX(),self.__workPackages[i].getX()],[self.__workPackages[i-1].getY(),self.__workPackages[i].getY()],linewidth=1,color='b')

        plt.savefig('pic/result_{}.pdf'.format(name))
        plt.close('all')
