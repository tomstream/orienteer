import math

from utils import define


class Resource(object):
    def __init__(self, id = None, x = None, y = None, timeLimit = None, speed = None):
        self.__id = id
        self.__currentX = x
        self.__currentY = y
        self.__initialX = x
        self.__initialY = y
        self.__workingTime = 0
        self.__timeLimit = define.get_value('time_limit') if timeLimit is None else timeLimit
        self.__speed = define.get_value('speed') if speed is None else speed
        self.__pid = -1
    
    def setTimeLimit(self, timeLimit):
        self.__timeLimit = timeLimit
    
    def setSpeed(self, speed):
        self.__speed = speed

    def setId(self, id):
        self.__id = id
    
    def setCurrentX(self, currentX):
        self.__currentX = currentX
    
    def setCurrentY(self, currentY):
        self.__currentY = currentY

    def getCurrentX(self):
        return self.__currentX

    def getCurrentY(self):
        return self.__currentY
    
    def setInitialX(self, initialX):
        self.__initialX = initialX
    
    def setInitialY(self, initialY):
        self.__initialY = initialY
    
    def copy(self):
        tmp = Resource()
        tmp.setTimeLimit(self.__timeLimit)
        tmp.setSpeed(self.__speed)
        tmp.setId(self.__id)
        tmp.setCurrentX(self.__currentX)
        tmp.setCurrentY(self.__currentY)
        tmp.setInitialX(self.__initialX)
        tmp.setInitialY(self.__initialY)
        return tmp
    
    def setWorkingTime(self, workingTime):
        self.__workingTime = workingTime
        
    def setPosition(self, x, y, pid):
        self.__currentX = x
        self.__currentY = y
        self.__pid = pid

    def getPosition(self):
        return self.__currentX, self.__currentY, self.__pid

    def getSpeed(self):
        return self.__speed
        
    def getInitialX(self):
        return self.__initialX

    def getInitialY(self):
        return self.__initialY

    def getInitialPos(self):
        return self.__initialX, self.__initialY

    def getCurrentPos(self):
        return [self.__currentX, self.__currentY, self.__pid]


    def dist(self, x, y):
        return math.sqrt((self.__currentX - x) * (self.__currentX - x) + (self.__currentY - y) * (self.__currentY - y))
    
    def getId(self):
        return self.__id
        
    def getWorkingTime(self):
        return self.__workingTime
        
    def needTime(self, workPackage):
        # print(workPackage.getWorkingTime(), workPackage.getX(), workPackage.getY(), self.__speed)
        # print(workPackage.getWorkingTime(), self.dist(workPackage.getX(), workPackage.getY()) / self.__speed)
        # input()
        return workPackage.getWorkingTime() + self.dist(workPackage.getX(), workPackage.getY()) / self.__speed

    def solveable(self, workPackage):
        if self.needTime(workPackage) <= (self.__timeLimit - self.__workingTime):
            return True
        else:
            return False
        
    def print(self):
        print(self.__id + " " + str(self.__currentX) + " " + str(self.__currentY) + " " + str(self.__workingTime))