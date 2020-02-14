import random
import define
from resource_obj import Resource
from work_package import WorkPackage
import math
import numpy as np


def generate_data(dataset_size, op_size, prize_type, seed=1234):
    np.random.seed(seed)
    depot = np.random.uniform(size=(dataset_size, 2))
    loc = np.random.uniform(size=(dataset_size, op_size, 2))

    # Methods taken from Fischetti et al. 1998
    if prize_type == 'constant':
        prize = np.ones((dataset_size, op_size))
    elif prize_type == 'uniform':
        prize = (1 + np.random.randint(0, 100, size=(dataset_size, op_size))) / 100.
    else:  # Based on distance to depot
        assert prize_type == 'distance'
        prize_ = np.linalg.norm(depot[:, None, :] - loc, axis=-1)
        prize = (1 + (prize_ / prize_.max(axis=-1, keepdims=True) * 99).astype(int)) / 100.
    global data
    data = []
    data = list(zip(depot.tolist(),loc.tolist(),prize.tolist(),))

def wrapper(idx):
    r_pos, locs, ps = data[idx]
    resource = [Resource("resource", r_pos[0], r_pos[1], timeLimit=define.get_value('time_limit'), speed=define.get_value('speed'))]
    packages = []
    for i, (loc, p) in enumerate(zip(locs, ps)):
        packages.append(WorkPackage(i, loc[0], loc[1], p, 0))
    return packages, resource

def get_data(idx):
    return data[idx]


def gen_random_data(package_num, seed):
    workpackages = []
    resources = []
    random.seed(seed)
    tmp = Resource("resource", random.random(), random.random(), timeLimit=define.get_value('time_limit'), speed=define.get_value('speed'))
    resources.append(tmp)
    for i in range(package_num):
        x = random.random()
        y = random.random()
        urgency = random.randint(1,100)/100
        working_time = random.random() * 0.1
        tmp = WorkPackage(i, x, y, urgency, working_time)
        workpackages.append(tmp)
    return workpackages, resources

def gen_random_data_constant(package_num, seed):
    random.seed(seed)
    resource = [random.random(), random.random()]
    packages = []
    for i in range(package_num):
        x = random.random()
        y = random.random()
        urgency = 1
        working_time = 0
        packages.append([x, y, urgency, working_time])
    return resource, packages

def gen_random_data_uniform(package_num, seed):
    random.seed(seed)
    resource = [random.random(), random.random()]
    packages = []
    for i in range(package_num):
        x = random.random()
        y = random.random()
        urgency = random.randint(1,100)/100
        working_time = 0
        packages.append([x, y, urgency, working_time])
    return resource, packages


def gen_random_dis(package_num, seed):
    random.seed(seed)
    x0 = random.random()
    y0 = random.random()
    resource = [random.random(), random.random()]
    packages = []
    max_urgency = 0
    for i in range(package_num):
        x = random.random()
        y = random.random()
        urgency = math.sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))
        max_urgency = max(urgency, max_urgency)
        working_time = 0
        packages.append([x, y, urgency, working_time])
    for p in packages:
        p[2] = (1 + 99 * p[2] / max_urgency) / 100
    return resource, packages

func_dict = {'uniform': gen_random_data_uniform,
             'constant': gen_random_data_constant,
             'dis': gen_random_dis}

def data_wrapper(package_num, seed):
    resource, packages = func_dict[define.get_value('func_type')](package_num, seed)
    ret_resource = [Resource("resource", resource[0], resource[1], timeLimit=define.get_value('time_limit'), speed=define.get_value('speed'))]
    ret_packages = []
    for i, p in enumerate(packages):
        ret_packages.append(WorkPackage(i, p[0], p[1], p[2], p[3]))
    return ret_packages, ret_resource
