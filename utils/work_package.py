import math
class WorkPackage(object):

    def __init__(self, id = "", x = 1, y = 1, urgency = 1, workingTime = 1):
        self.__id = id
        self.__x = x
        self.__y = y
        self.__urgency = urgency
        self.__workingTime = workingTime

    def setId(self, id):
        self.__id = id

    def setX(self, x):
        self.__x = x

    def setY(self, y):
        self.__y = y

    def setUrgency(self, urgency):
        self.__urgency = urgency

    def setWorkingTime(self, workingTime):
        self.__workingTime = workingTime

    def __str__(self):
        return "id: {} x:{} y:{} urgency:{} workingtime:{}".format(self.__id, self.__x, self.__y, self.__urgency, self.__workingTime)

    def copy(self):
        tmp = WorkPackage()
        tmp.setId(self.__id)
        tmp.setX(self.__x)
        tmp.setY(self.__y)
        tmp.setUrgency(self.__urgency)
        tmp.setWorkingTime(self.__workingTime)
        return tmp

    def getId(self):
        return self.__id

    def dist(self, x, y):
        return math.sqrt((self.__x - x) * (self.__x - x) + (self.__y - y) * (self.__y - y))

    def getWorkingTime(self):
        return self.__workingTime

    def getX(self):
        return self.__x

    def getY(self):
        return self.__y

    def getPosition(self):
        return [self.__x, self.__y]

    def getUrgency(self):
        return self.__urgency

    def print(self):
        # print(self.__id, type(self.__id))
        print(self.__id + " " + str(self.__x) + " " + str(self.__y) + " " + str(self.__urgency) + " " + str(self.__workingTime))