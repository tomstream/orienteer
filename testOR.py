from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import define
import random
import numpy as np
import argparse

def create_data_model(seed=10000,capacity=100):
    """Stores the data for the problem."""
    random.seed(seed)
    resource = [random.random(), random.random()]
    packages = []
    for i in range(define.package_num):
        x = random.random()
        y = random.random()
        urgency = random.randint(1,10)
        working_time = random.random() * 0.2
        packages.append([x, y, urgency, working_time])
    time_matrix = np.zeros((41,41),dtype=int)
    urgency_list = np.zeros(41,dtype=int)
    for i in range(len(packages)):
        time_matrix[0, i+1] = (define.dis(resource[0], packages[i][0], resource[1], packages[i][1])/define.speed + packages[i][3]) * 1000
        urgency_list[i+1] = packages[i][2]

    for i in range(len(packages)):
        for j in range(len(packages)):
            if i == j:
                continue
            time_matrix[i+1,j+1] = (define.dis(packages[i][0], packages[j][0], packages[i][1], packages[j][1])/define.speed + packages[j][3]) * 1000



    data = {}
    data['time_matrix'] = [list(line) for line in time_matrix]
    data['demands'] = list(urgency_list)
    data['vehicle_capacities'] = capacity
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    total_time = 0
    total_load = 0
    total_time2 = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_load = 0
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            index = assignment.Value(routing.NextVar(index))
            next_index = manager.IndexToNode(index)
            total_time2 += data['time_matrix'][node_index][next_index]

        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    assignment.Min(time_var),
                                                    assignment.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            assignment.Min(time_var))
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_time += assignment.Min(time_var)
        total_load += route_load
    print('Total time of all routes: {}min'.format(total_time))
    print('Total Load of all routes: {}'.format(total_load))

def ret_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    time_dimension = routing.GetDimensionOrDie('Time')
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    total_time = 0
    total_load = 0
    total_time2 = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_load = 0
        while not routing.IsEnd(index):
            time_var = time_dimension.CumulVar(index)
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += '{0} Time({1},{2}) -> '.format(
                manager.IndexToNode(index), assignment.Min(time_var),
                assignment.Max(time_var))
            index = assignment.Value(routing.NextVar(index))
            next_index = manager.IndexToNode(index)
            total_time2 += data['time_matrix'][node_index][next_index]

        time_var = time_dimension.CumulVar(index)
        plan_output += '{0} Time({1},{2})\n'.format(manager.IndexToNode(index),
                                                    assignment.Min(time_var),
                                                    assignment.Max(time_var))
        plan_output += 'Time of the route: {}min\n'.format(
            assignment.Min(time_var))
        plan_output += 'Load of the route: {}\n'.format(route_load)
        total_time += assignment.Min(time_var)
        total_load += route_load
    return total_time

def main(seed, capacity):
    """Solve the VRP with time windows."""
    data = create_data_model(seed,capacity)
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                           data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def time_callback(from_index, to_index):
        """Returns the travel time between the two nodes."""
        # Convert from routing variable Index to time matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['time_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    time = 'Time'
    routing.AddDimension(
        transit_callback_index,
        0,  # allow waiting time
        100000,  # maximum time per vehicle
        True,  # Don't force start cumul to zero.
        time)
    # time_dimension = routing.GetDimensionOrDie(time)
    # Add time window constraints for each location except depot.
    # for location_idx, time_window in enumerate(data['time_windows']):
    #     if location_idx == 0:
    #         continue


    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimension(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')
    # capacity_dimension = routing.GetDimensionOrDie('Capacity')
    # capacity_dimension.CumulVar(routing.End(0)).SetRange(data['vehicle_capacities'],10000)

    penalty = 10000
    for node in range(1, len(data['time_matrix'])):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)


    # for i in range(data['num_vehicles']):
    #     routing.AddVariableMaximizedByFinalizer(
    #         time_dimension.CumulVar(routing.End(i)))
    #     routing.AddVariableMaximizedByFinalizer(
    #         capacity_dimension.CumulVar(routing.End(i)))
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.time_limit.seconds = 60
    search_parameters.solution_limit = 2000
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_MOST_CONSTRAINED_ARC)
    assignment = routing.SolveWithParameters(search_parameters)
    # routing.solver().Add(day_0 + 7 == day_1)
    if assignment:
        # print_solution(data, manager, routing, assignment)
        return ret_solution(data, manager, routing, assignment)
    else:
        return None
if __name__ == '__main__':
    result = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000, help="start of seed")
    parser.add_argument("--span", type=int, default=10, help="span")
    args = parser.parse_args()

    all_result = []
    for seed in range(args.seed + 10000, args.seed + 10000 + args.span):
        # print('\r{}'.format(seed))
        left,right = 30,130
        last_result = [0,0]
        capacity = 0
        time = 0
        for i in range(left, right):
            result = main(seed, i)
            if result < define.timeLimit * 1000 and capacity < i:
                time = result
                capacity = i
                # print(capacity, i)

        print(seed,capacity, time)
        # while left+1!=right:
        #     mid = (left + right)//2
        #     result = main(seed, mid)
        #     if result is not None and result<=define.timeLimit * 1000:
        #         left = mid
        #     else:
        #         right = mid
        #     last_result = (mid, result)
        #     print(last_result, left, right)
        all_result.append(capacity)
    print(np.mean(all_result))

