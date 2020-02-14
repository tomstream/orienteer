from work_package import WorkPackage
from resource_obj import Resource
from schedule import Schedule
from path_obj import Path
import copy
import time
import define
from memory import ReplayMemory
import time
import model
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description='E-Net')
# general settings
parser.add_argument('--input', default='in.json', type=str, help='name of input file')
parser.add_argument('--output', default='out.json', type=str, help='name of output file')
args = parser.parse_args()

def readFile(pathName):
    workpackages = []
    resources = []
    print(pathName)
    with open(pathName, 'r', encoding = 'utf-8') as file:
        flag = 0
        while True:
            line = file.readline()
            if not line:
                break
            line = line[:-1]
            length = len(line)
            if length >= 1 and line[length - 1] == '[':
                flag = 1
            if flag == 1:
                line = file.readline()
                line = line[:-1]
                length = len(line)
                if length >= 2 and line[length - 2] == ']':
                    break
                line = file.readline()
                line = line[:-1]
                length = len(line)
                cnt = 0
                l = 0
                for i in range(0, length):
                    if line[i] == '"':
                        cnt = cnt + 1
                    if cnt == 3:
                        l = i + 1
                        break
                id = line[l : length - 2]


                line = file.readline()
                line = line[:-1]
                length = len(line)
                cnt = 0
                l = 0
                r = 0
                mid = 0
                for i in range(0, length):
                    if line[i] == '(':
                        l = i
                    elif line[i] == ')' or line[i] == ',':
                        r = i
                        break
                    elif line[i] == ' ' and l != 0 and mid == 0:
                        mid = i
                x = float(line[l + 1 : mid])
                y = float(line[mid + 1 : r])

                line = file.readline()
                line = line[:-1]
                length = len(line)
                cnt = 0
                l = 0
                for i in range(0, length):
                    if line[i] == '"':
                        cnt = cnt + 1
                    if cnt == 3:
                        l = i + 1
                        break
                urgent = float(line[l : length - 2])

                line = file.readline()
                line = line[:-1]
                length = len(line)
                cnt = 0
                l = 0
                for i in range(0, length):
                    if line[i] == '"':
                        cnt = cnt + 1
                    if cnt == 3:
                        l = i + 1
                        break
                workingTime = float(line[l : length - 2])

                for i in range(0, 10):
                    line = file.readline()
                    line = line[:-1]
                tmp = WorkPackage(id, x, y, urgent, workingTime)
                workpackages.append(tmp)
        
        flag = 0
        while True:
            line = file.readline()
            if not line:
                break
            line = line[:-1]
            length = len(line)
            if length >= 1 and line[length - 1] == '[':
                flag = 1
            if flag == 1:
                line = file.readline()
                line = line[:-1]
                length = len(line)
                if length >= 2 and line[length - 1] == ']':
                    break
                line = file.readline()
                line = line[:-1]
                length = len(line)
                cnt = 0
                l = 0
                for i in range(0, length):
                    if line[i] == '"':
                        cnt = cnt + 1
                    if cnt == 3:
                        l = i + 1
                        break
                id = line[l : length - 2]
                line = file.readline()
                line = line[:-1]
                line = file.readline()
                line = line[:-1]
                line = file.readline()
                line = line[:-1]

                line = file.readline()
                line = line[:-1]
                length = len(line)
                cnt = 0
                l = 0
                r = 0
                mid = 0
                for i in range(0, length):
                    if line[i] == '(':
                        l = i
                    elif line[i] == ')' or line[i] == ',':
                        r = i
                        break
                    elif line[i] == ' ' and l != 0 and mid == 0:
                        mid = i
                x = float(line[l + 1 : mid])
                y = float(line[mid + 1 : r])

                for i in range(0, 5):
                    line = file.readline()
                    line = line[:-1]
                tmp = Resource(id, x, y)
                # print("success resource")
                resources.append(tmp)
    return resources, workpackages

if __name__ == '__main__':
    resources, workpackages = readFile(args.input)
    # print("initial workpackage " + str(len(workpackages)) + " initial resource " + str(len(resources)))
    policy_net = model.DQN(define.input_size, define.dropout_rate)
    target_net = model.DQN(define.input_size, define.dropout_rate)

    define.timeLimit = 480
    define.planLimit = 10
    define.timeInterval = 20
    define.globalPlanLimit = 20
    define.speed = 500

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=define.learning_rate,weight_decay=define.weight_decay)

    replay_memory = ReplayMemory(define.memory_capacity)
    runTime = time.time()
    schedule = Schedule(resources, workpackages, replay_memory, target_net, is_dqn=False, time_limit=define.timeLimit, plan_limit=define.planLimit, time_interval=define.timeInterval)

    schedule.greedySchedule()
    schedule.print()

    for i in range(30):
        model.optimize_model(policy_net, target_net, optimizer, replay_memory)
    target_net.load_state_dict(policy_net.state_dict())
    target_net = target_net.eval()

    schedule = Schedule(resources, workpackages, replay_memory, target_net, is_dqn=True, time_limit=define.timeLimit, plan_limit=define.planLimit, time_interval=define.timeInterval)
    schedule.greedySchedule()

    with open(args.output, 'w') as f:
        f.write(schedule.json())
