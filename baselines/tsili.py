import numpy as np
import time
import math
import queue
from multiprocessing import Pool, Value, Lock



def dis(x, y):
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

class item(object):
    def __init__(self, prob, prize, cost, pos, seq):
        self.prob = prob
        self.prize = prize
        self.cost = cost
        self.pos = pos
        self.seq = seq

    def add(self, new):
        return item(prob = (new[3]/dis(self.pos, [new[1],new[2]]))**4,
                    prize = self.prize + new[3],
                    cost = self.cost + dis(self.pos, [new[1],new[2]]),
                    pos = [new[1],new[2]],
                    seq = self.seq+[new[0]])
    def __lt__(self,other):
        return self.prob < other.prob

    def __str__(self):
        return str(self.prob)+str(self.seq)+str(self.prize)

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
    data = list(zip(depot.tolist(),loc.tolist(),prize.tolist()))
    return data

def second(x):
    return x[1]

def simulate(index, root, pos, val, N, T, size):
    mx_prize = 0
    mx_seq = []

    q = queue.PriorityQueue()
    q.put(item(0, 0, 0, [root[0],root[1]], []))

    while(q.empty()==False):
        v = []
        while(q.empty()==False):#dequeue, to obtain the origin state
            now = q.get()
            v.append(now)
            if(now.prize>mx_prize):
                mx_prize = now.prize
                mx_seq = now.seq
        count = 0
        for now in v:#to obtain the next state
            flag = {}
            for i in now.seq:
                flag[i] = 1
            for i in range(N):
                if i not in flag:
                    if(now.cost + dis(now.pos,pos[i])+dis(pos[i], root)<=T):
                        if(count<size):
                            pre = now.add([i, pos[i][0], pos[i][1], val[i]])
                            q.put(pre)
                            count = count+1
                        else:
                            pre = now.add([i, pos[i][0], pos[i][1], val[i]])
                            q.put(pre)
                            q.get()
    return index, mx_prize, mx_seq

def multicore(input, T, size):
    u = time.time()
    query = len(input)

    pool = Pool(58)
    res = []
    for i in range(query):
        root = input[i][0]
        pos = input[i][1]
        N = len(pos)
        val = input[i][2]
        res.append(pool.apply_async(simulate, (i, root, pos, val, N, T, size)))
        '''cost = dis(root, pos[seq[0]])# To test whether this decision works
        num = len(seq)
        for i in range(num-1):
            cost = cost + dis(pos[seq[i]],pos[seq[i+1]])
        cost = cost + dis(pos[seq[-1]], root)
        print(cost)'''

    output= []

    for res in res:
        index, ans, seq= res.get()
        output.append([index,ans,seq])
        #print("The ",index,"th best sequence is:",seq)
        #print("The max score is:",ans)

    v = time.time()
    print("Time: ",v-u, np.mean([_[1] for _ in output]))
    return output

# time limit = 2
if __name__=='__main__':
    multicore(generate_data(dataset_size=10000, op_size=20, prize_type='constant', seed=1234), T=2, size=100)
    multicore(generate_data(dataset_size=10000, op_size=20, prize_type='uniform', seed=1234), T=2, size=100)
    multicore(generate_data(dataset_size=10000, op_size=20, prize_type='distance', seed=1234), T=2, size=100)

    # time limit = 3
    multicore(generate_data(dataset_size=10000, op_size=50, prize_type='constant', seed=1234), T=3, size=100)
    multicore(generate_data(dataset_size=10000, op_size=50, prize_type='uniform', seed=1234), T=3, size=100)
    multicore(generate_data(dataset_size=10000, op_size=50, prize_type='distance', seed=1234), T=3, size=100)

    # time limit = 4
    multicore(generate_data(dataset_size=10000, op_size=100, prize_type='constant', seed=1234), T=4, size=100)
    multicore(generate_data(dataset_size=10000, op_size=100, prize_type='uniform', seed=1234), T=4, size=100)
    multicore(generate_data(dataset_size=10000, op_size=100, prize_type='distance', seed=1234), T=4, size=100)