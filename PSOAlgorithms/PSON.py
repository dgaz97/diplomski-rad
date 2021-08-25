from copy import deepcopy
import numpy as np
import time
class PSON:
    def __init__(self, function, particleCount, dimensions,minIterationCount,maxIterationCount, topology="gbest"):
        self.function=function
        self.particleCount=particleCount
        self.dimensions=dimensions
        self.xMax=self.function.getXMax()
        self.minIterations=minIterationCount
        self.maxIterations=maxIterationCount

        self.timeAtAcceptableResult=None
        self.iterationAtAcceptableResult=None
        self.functionExecutionAtAcceptableResult=None;

        self.iterationCount=0

        particleCoords=np.random.default_rng().uniform(low=-self.xMax,high=self.xMax, size=(self.particleCount, self.dimensions))

        self.particles=[]

        #Generate particles with gbest topology
        if topology=="gbest":
            indexesOfP=list(range(0,particleCount))
            for p in particleCoords:
                self.particles.append(Particle(p,self.function,indexesOfP))
        #Generate particles with lbest topology
        elif isinstance(topology, int):
            allIndexes=list(range(0,particleCount))
            for i,p in enumerate(particleCoords):
                indexesOfP=[allIndexes[i]]
                for j in range(topology//2):
                    indexesOfP.append(allIndexes[(i+j+1)%self.particleCount])
                    indexesOfP.append(allIndexes[i-j-1])
                self.particles.append(Particle(p,self.function,indexesOfP))
        #Generate particles with von Neumann topology
        elif topology=="vonNeumann":
            f1 = round(np.sqrt(self.particleCount))
            f2 = round(self.particleCount/f1)
            allIndexes=np.reshape(list(range(0,self.particleCount)),(f1,f2))
            for i,p in enumerate(particleCoords):
                (x,y)=np.where(allIndexes==i)
                x=x[0]
                y=y[0]
                indexesOfP=[allIndexes[x][y],allIndexes[x-1][y],allIndexes[(x+1)%f1][y],allIndexes[x][y-1],allIndexes[x][(y+1)%f2]]
                self.particles.append(Particle(p,self.function,indexesOfP))
        
    def runIteration(self):
        for p in self.particles:
            p.move(self.particles)
        result=self.getBestResult()
        if(np.abs(result[1]-self.function.getOptimalResult())<=self.function.getAcceptableError() and self.iterationAtAcceptableResult==None):
            self.iterationAtAcceptableResult=self.iterationCount
            self.functionExecutionAtAcceptableResult=self.function.numberOfEvaluations
            self.timeAtAcceptableResult=time.time()
        if(self.iterationCount==self.minIterations):
            self.resultAtMinIter=result
            self.evaluationsAtMinIter=self.function.numberOfEvaluations
            self.timeAtMinIter=time.time()


    def iterate(self):
        bestResults=np.array([])
        for i in range(self.maxIterations):
            self.iterationCount+=1
            self.runIteration()
            bestResults=np.append(bestResults,self.getBestResult())
            if(bestResults[-1]<self.function.getAcceptableError() and self.iterationCount>=self.minIterations):
                return np.reshape(bestResults,(-1,2))
        return np.reshape(bestResults,(-1,2))

    def getBestResult(self):
        bestKnownIndex=0
        for i,p in enumerate(self.particles):
            if p.personalBest[1]<self.particles[bestKnownIndex].personalBest[1]:
                bestKnownIndex=i
        return self.particles[bestKnownIndex].personalBest


class Particle:
    def __init__(self, coords, function,informantIndexes, chi=0.729844, phiMax=4.1):
        self.function=function
        eval=self.function.evaluate(coords)
        self.current=[coords,eval]#current coordinates
        self.personalBest=[coords,eval]#pbest
        self.inertiaVector=np.zeros(coords.shape)
        self.informantIndexes=informantIndexes#neighborhood particle indexes

        self.chi=chi
        self.phiMax=phiMax
    

    def move(self, allParticles):
        dimensions=dimensions=self.current[0].shape[0]

        upperSum=0
        phi=np.zeros(dimensions)

        phiPerElement=self.phiMax/len(self.informantIndexes)
        for p in self.informantIndexes:
            pb=allParticles[p]
            phiI=np.random.uniform(0,phiPerElement,dimensions)
            phi+=phiI
            upperSum+=np.multiply(phiI,pb.personalBest[0])
            
        
        
        attractorPoint=np.divide(upperSum,phi)

        a1=attractorPoint-self.current[0]
        a2=np.multiply(phi,a1)
        a3=self.inertiaVector+a2

        movementVector = self.chi*a3
        
        newLocation=self.current[0]+movementVector
        newLocationEval=self.function.evaluate(newLocation)
        self.inertiaVector=movementVector
        self.current=[newLocation,newLocationEval]
        
        if(newLocationEval<self.personalBest[1]):
            self.personalBest=[newLocation,newLocationEval]

