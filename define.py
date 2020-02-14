import math


def init():
    global _global_dict
    _global_dict = {'package_num':50, 'speed':1, 'time_limit':1, 'func_type':'uniform', 'time_interval':0.05, 'plan_limit':10}


def set_value(key,value):
    """ 定义一个全局变量 """
    _global_dict[key] = value

def get_value(key,defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue

def time_limit():
    p = _global_dict['package_num']
    d = {20:2, 50:3, 100:4}
    return d[p]

seed = 42
# capacity of replay memory
memory_capacity = 1000
batch_size = 128

gamma = 0.99
input_size = 9
dropout_rate = 0.9
learning_rate = 1e-5
weight_decay = 0

# timeLimit = 1
planLimit = 10

timeInterval = 0.05
globalPlanLimit = 20
# speed = 5
num_grid = 50
package_num = 40

# timeLimit = 480
# planLimit = 10
# timeInterval = 20
# globalPlanLimit = 20
# speed = 500

def dis(x1, x2, y1, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))

