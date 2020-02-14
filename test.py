import copy
import math
import queue
import time
st = time.time()
ed = time.time()
print(ed - st)
# class Resource(object):
#     __timeLimit = 480
#     __speed = 500

#     def __init__(self, x, y):
#         self.__currentX = x
#         self.__currentY = y
    
#     def dist(self, x, y):
#         return math.sqrt((self.__currentX - x) * (self.__currentX - x) + (self.__currentY - y) * (self.__currentY - y))
    
#     def dist2(self, x, y):
#         return self.dist(x, y) + 1

# re = Resource(1, 1)
# print(re.dist2(2, 2))
# a = "sssa"
# b = "sssa"
# c = "sssb"
# print(a == b, a == c)
# print(math.sqrt(10))
# L = ["a", "b", "c", "d"]
# del L[0]
# print(L)
# dic = {}
# dic[1] = 2
# dic[3] = 4
# print(1 in dic)
# print(2 in dic)
class node(object):
    __timeLimit = 480
    def __init__(self, number = 0):
        self.__number = number
    def print(self):
        print(self.__number, self.__timeLimit)
    def change(self, number):
        self.__number = number
    def __lt__(self, other):
        return self.__number > other.__number
    def copy(self):
        tmp = node()
        tmp.change(self.__number)
        return tmp
        
node1 = node(10)
time1 = time.time()
for i in range(10000):
    node2 = copy.copy(node1)
time1 = time.time() - time1
time2 = time.time()
for i in range(10000):
    node3 = copy.deepcopy(node1)
time2 = time.time() - time2
time3 = time.time()
for i in range(10000):
    node4 = node1
time3 = time.time() - time3
time4 = time.time()
for i in range(10000):
    node5 = node1.copy()
time4 = time.time() - time4
node2.change(1)
node3.change(2)
node4.change(3)
node5.change(4)
node1.print()
node2.print()
node3.print()
node4.print()
node5.print()
print(time1, time2, time3, time4)
# node1 = node(1)
# node2 = node(2)
# node3 = node(3)
# list1 = []
# list1.append(node1)
# list1.append(node2)
# list1.append(node3)
# list2 = copy.deepcopy(list1)
# list2[0].change(4)
# for item in list1:
#     item.print()
# for item in list2:
#     item.print()

# pq = queue.PriorityQueue()
# lis = []
# lis.append(pq)
# lis[0].put(node(10))
# lis[0].put(node(11))
# lis[0].put(node(9))
# while not lis[0].empty():
#     lis[0].get().print()
#     print(lis[0].qsize())
