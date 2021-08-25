import numpy as np
class SphereFunction:
    numberOfEvaluations=0
    @staticmethod
    def getOptimalResult():
        return 0
    @staticmethod
    def evaluate(varArray):
        SphereFunction.numberOfEvaluations+=1
        return np.sum(np.square(varArray))
    @staticmethod
    def getAcceptableError():
        return 0.01
    @staticmethod
    def getXMax():
        return 5.12


class RastriginFunction:
    numberOfEvaluations=0
    @staticmethod
    def getOptimalResult():
        return 0
    @staticmethod
    def evaluate(varArray):
        RastriginFunction.numberOfEvaluations+=1
        return np.sum(np.square(varArray)-10*np.cos(2*np.pi*varArray)+10)
    @staticmethod
    def getAcceptableError():
        return 100
    @staticmethod
    def getXMax():
        return 5.12

class Griewank:
    numberOfEvaluations=0
    @staticmethod
    def getOptimalResult():
        return 0
    @staticmethod
    def evaluate(varArray):
        Griewank.numberOfEvaluations+=1
        r1=np.sum(np.square(varArray)/4000)
        r2=varArray
        for i in range(len(varArray)):
            r2[i]=r2[i]/((i+1)**0.5)
        r3=np.prod(np.cos(r2))
        return r1-r3+1
        
    @staticmethod
    def getAcceptableError():
        return 0.05
    @staticmethod
    def getXMax():
        return 600


    
class Zakharov:
    numberOfEvaluations=0
    @staticmethod
    def getOptimalResult():
        return 0
    @staticmethod
    def evaluate(varArray):
        Zakharov.numberOfEvaluations+=1
        sum1=0
        sum2=0
        sum3=0
        for i in varArray:
            sum1+=i**2
        for i,j in enumerate(varArray):
            sum2+=0.5*j*i
            sum3+=0.5*j*i
        sum2**=2
        sum3**=4
        return sum1+sum2+sum3
    @staticmethod
    def getAcceptableError():
        return 10
    @staticmethod
    def getXMax():
        return 10

class Rosenbrock:
    numberOfEvaluations=0
    @staticmethod
    def getOptimalResult():
        return 0
    @staticmethod
    def evaluate(varArray):
        Rosenbrock.numberOfEvaluations+=1
        sum=0
        for i in range(0,len(varArray)-1,1):
            sum+=100*(varArray[i+1]-varArray[i]**2)**2+(varArray[i]-1)**2
        return sum
    @staticmethod
    def getAcceptableError():
        return 50
    @staticmethod
    def getXMax():
        return 10
