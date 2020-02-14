from __future__ import print_function
import define
import random
import numpy as np
import argparse
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sko.GA import GA_TSP
from sko.SA import SA_TSP
from sko.ACA import ACA_TSP
from sko.IA import IA_TSP
from multiprocessing import Pool, Lock, Value


num_points = 10

points_coordinate = np.random.rand(num_points, 2)  # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')





def create_data_model(seed=100000,capacity=100):
    """Stores the data for the problem."""
    random.seed(seed)
    resource = [random.random(), random.random()]
    packages = []
    for i in range(define.package_num):
        x = random.random()
        y = random.random()
        urgency = 0.1 * random.randint(1,10)
        working_time = random.random() * 0.1
        packages.append([x, y, urgency, working_time])
    time_vector = np.zeros(define.package_num, dtype=float)
    time_matrix = np.zeros((define.package_num,define.package_num),dtype=float)
    urgency_list = np.zeros(define.package_num,dtype=float)
    for i in range(len(packages)):
        time_vector[i] = (define.dis(resource[0], packages[i][0], resource[1], packages[i][1])/define.speed + packages[i][3])
        urgency_list[i] = packages[i][2]

    for i in range(len(packages)):
        for j in range(len(packages)):
            if i == j:
                continue
            time_matrix[i,j] = (define.dis(packages[i][0], packages[j][0], packages[i][1], packages[j][1])/define.speed + packages[j][3])

    return time_vector, time_matrix, urgency_list



def cal_total_distance_func(time_vector, time_matrix, urgency_list):
    def cal_total_distance(routine):
        total_urgency = 0
        total_time = 0
        if time_vector[routine[0]] <= define.timeLimit:
            total_urgency += urgency_list[routine[0]]
            total_time += time_vector[routine[0]]
        for i in range(1, len(routine)):
            tmp_time = time_matrix[routine[i-1],routine[i]]
            tmp_urgency = urgency_list[routine[i]]
            if total_time + tmp_time > define.timeLimit:
                break
            total_urgency += tmp_urgency
            total_time += tmp_time
        return -total_urgency
    return cal_total_distance

def GA_solver(time_vector, time_matrix, urgency_list):
    cal_total_distance = cal_total_distance_func(time_vector,time_matrix,urgency_list)
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=define.package_num, size_pop=500, max_iter=1000, prob_mut=0.1)
    best_points, best_result= ga_tsp.run()
    return best_points,-best_result

def SA_solver(time_vector, time_matrix, urgency_list):
    cal_total_distance = cal_total_distance_func(time_vector,time_matrix,urgency_list)
    sa_tsp = SA_TSP(func=cal_total_distance, x0=range(define.package_num), T_max=500, T_min=1, L=10 * define.package_num)
    best_points, best_result = sa_tsp.run()
    return best_points,-best_result

def IA_solver(time_vector, time_matrix, urgency_list):
    cal_total_distance = cal_total_distance_func(time_vector,time_matrix,urgency_list)
    ia_tsp = IA_TSP(func=cal_total_distance, n_dim=define.package_num, size_pop=1000, max_iter=2000, prob_mut=0.2,
                T=0.6, alpha=0.95)
    best_points, best_result = ia_tsp.run()
    return best_points,-best_result

def run(seed):
    data = create_data_model(seed)

    sa = SA_solver(*data)[1]
    ga = GA_solver(*data)[1]
    ia = IA_solver(*data)[1]
    return [sa,ga,ia]


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1000, help="start of seed")
parser.add_argument("--span", type=int, default=50, help="span")
parser.add_argument("--jobs", type=int, default=30, help="jobs")
args = parser.parse_args()

if args.jobs == 1:
    result = []
    for i in range(args.seed, args.seed + args.span):
        result.append(run(i))
else:
    pool = Pool(args.jobs)
    result = pool.map(run, range(args.seed, args.seed + args.span))
    pool.join()
    pool.close()

sa_result = [r[0] for r in result]
ga_result = [r[1] for r in result]
ia_result = [r[2] for r in result]
print('sa:{} ga:{} ia:{}'.format(np.mean(sa_result),np.mean(ga_result),np.mean(ia_result)))

# data = create_data_model(100000)
# time_vector, time_matrix, urgency_list = data
# # print(time_vector[37],urgency_list[37])
# print(time_vector[14],urgency_list[14])
#
# result = IA_solver(*data)
# print(result)