from copy import deepcopy
import numpy as np
import time
class PSOSA:
    def __init__(self, function, particleCount, dimensions,minIterationCount,maxIterationCount, c1=2, c2=2, inertiaWeightMax=0.9, inertiaWeightMin=0.4, maxN1=5, maxN2=10, eta=1, topology="gbest"):
        self.function=function
        self.particleCount=particleCount
        self.dimensions=dimensions
        self.xMax=self.function.getXMax()
        self.minIterations=minIterationCount
        self.maxIterations=maxIterationCount
        self.c1=c1
        self.c2=c2

        self.timeAtAcceptableResult=None
        self.iterationAtAcceptableResult=None
        self.functionExecutionAtAcceptableResult=None;

        self.inertiaWeightMax=inertiaWeightMax
        self.inertiaWeightMin=inertiaWeightMin
        self.iterationCount=0

        self.maxN1=maxN1
        self.maxN2=maxN2
        self.eta=eta
        self.interdec=particleCount/1000

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
        
        self.checkNeighborhood()

        fBest=self.getBestStarting()
        fWorst=self.getWorstStarting()
        Pr=0.01#some small constant

        startingTemperature=-(fWorst[1]-fBest[1])/np.log(Pr)
        self.temperature=startingTemperature

        self.tempRatio=1/(self.maxN1*self.maxIterations*self.particleCount)


    def checkNeighborhood(self):
        for p in self.particles:
            p.checkNeighborhood(self.particles)

    def runIteration(self):
        for p in self.particles:#do PSO
            p.move(self.c1, self.c2,self.dimensions,self.xMax,self.calculateLinearInertiaWeight())
        
        self.bestResults=np.append(self.bestResults,self.getBestResult())
        self.bestResults=np.reshape(self.bestResults,(-1,2))

        if (self.bestResults[-2][1]-self.bestResults[-1][1])==0:#no change in result
            for p in self.particles:
                self.temperature=p.simulatedAnnealing(self.maxN1, self.maxN2,self.temperature, self.eta,self.tempRatio)#annealing will update temperature

        self.eta=self.eta*(1-self.interdec)
        
        self.checkNeighborhood()

        result=self.getBestResult()
        if(np.abs(result[1]-self.function.getOptimalResult())<=self.function.getAcceptableError() and self.iterationAtAcceptableResult==None):
            self.iterationAtAcceptableResult=self.iterationCount
            self.functionExecutionAtAcceptableResult=self.function.numberOfEvaluations
            self.timeAtAcceptableResult=time.time()
        if(self.iterationCount==self.minIterations):
            self.resultAtMinIter=result
            self.evaluationsAtMinIter=self.function.numberOfEvaluations
            self.timeAtMinIter=time.time()

    def calculateLinearInertiaWeight(self):
        weight = self.inertiaWeightMin+(self.inertiaWeightMax-self.inertiaWeightMin)*(1-(self.iterationCount/self.maxIterations))
        return weight

    def iterate(self):
        self.bestResults=np.array(self.getBestResult())#record starting result
        self.bestResults=np.reshape(self.bestResults,(-1,2))

        for i in range(self.maxIterations):
            self.iterationCount+=1
            self.runIteration()
            if(self.bestResults[-1][1]<self.function.getAcceptableError() and self.iterationCount>=self.minIterations):
                return self.bestResults
        #print(self.eta)
        return self.bestResults
        


    def getBestResult(self):
        bestKnownIndex=0
        for i,p in enumerate(self.particles):
            if p.bestKnown[1]<self.particles[bestKnownIndex].bestKnown[1]:
                bestKnownIndex=i
        return self.particles[bestKnownIndex].bestKnown

    def getBestStarting(self):
        bestKnownIndex=0
        for i,p in enumerate(self.particles):
            if p.current[1]<self.particles[bestKnownIndex].current[1]:
                bestKnownIndex=i
        return deepcopy(self.particles[bestKnownIndex].current)
    def getWorstStarting(self):
        worstKnownIndex=0
        for i,p in enumerate(self.particles):
            if p.current[1]>self.particles[worstKnownIndex].current[1]:
                worstKnownIndex=i
        return deepcopy(self.particles[worstKnownIndex].current)

class Particle:
    def __init__(self, coords, function,informantIndexes):
        self.function=function
        eval=self.function.evaluate(coords)
        self.current=[coords,eval]#current coordinates
        self.personalBest=[coords,eval]#pbest
        self.bestKnown=[coords,eval]#gbest or lbest, depending on topology
        self.inertiaVector=np.zeros(coords.shape)
        self.informantIndexes=informantIndexes#neighborhood particle indexes
    
    def checkNeighborhood(self, allParticles):
        for i in self.informantIndexes:
            if(allParticles[i].personalBest[1]<self.bestKnown[1]):
                self.bestKnown=deepcopy(allParticles[i].personalBest)


    def move(self, c1, c2, dimensions, velocityClamp, inertiaWeight=1):
        inertiaVectorComponent = self.inertiaVector * inertiaWeight
        
        personalBestVectorFactors = np.random.rand(dimensions,)
        personalBestVectorComponent = (self.personalBest[0]-self.current[0])*personalBestVectorFactors*c1

        bestKnownVectorFactors = np.random.rand(dimensions,)
        bestKnownVectorComponent = (self.bestKnown[0]-self.current[0])*bestKnownVectorFactors*c2
        
        
        movementVector = np.minimum(np.maximum(inertiaVectorComponent+personalBestVectorComponent+bestKnownVectorComponent,-velocityClamp),velocityClamp)
        newLocation=np.minimum(np.maximum(self.current[0]+movementVector,-velocityClamp),velocityClamp)
        newLocationEval=self.function.evaluate(newLocation)
        self.inertiaVector=movementVector
        
        self.current=[newLocation,newLocationEval]
        if(newLocationEval<self.personalBest[1]):
            self.personalBest=[newLocation,newLocationEval]
        if(newLocationEval<self.bestKnown[1]):
            self.bestKnown=[newLocation,newLocationEval]
    def simulatedAnnealing(self,maxN1,maxN2, temperature, eta, tempRatio):
        t=temperature
        for i in range(maxN1):
            newPersonalBest=deepcopy(self.personalBest[0])
            for j in range(maxN2):
                rands=np.random.default_rng().normal(0,1, size=(self.current[0].size))*eta
                delta=newPersonalBest*rands

                newPersonalBest=np.minimum(np.maximum(newPersonalBest+delta,-self.function.getXMax()),self.function.getXMax())
                newPersonalBestValue=self.function.evaluate(newPersonalBest)
                if(newPersonalBestValue<self.personalBest[1]):
                    self.personalBest=[newPersonalBest,newPersonalBestValue]
                    break
                elif min(np.exp(-(self.personalBest[1]-self.bestKnown[1])/t),1)>np.random.rand():
                    self.current=[newPersonalBest,newPersonalBestValue]
                    break
            t=t*(1-tempRatio)
        return t
                
    

