from copy import deepcopy
import numpy as np
import time
class PSOWN:
    def __init__(self, function, particleCount, dimensions, maxIterations, topology="gbest"):
        self.function=function
        self.particleCount=particleCount
        self.dimensions=dimensions
        self.xMax=self.function.getXMax()
        self.maxIterations=maxIterations

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
        
        #self.checkNeighborhood()

    # def checkNeighborhood(self):
    #     for p in self.particles:
    #         p.checkNeighborhood(self.particles)

    def runIteration(self):
        for p in self.particles:
            p.move(self.particles,self.xMax)
        #self.checkNeighborhood()
        self.iterationCount+=1
        result=self.getBestResult()
        if(np.abs(result[1]-self.function.getOptimalResult())<=self.function.getAcceptableError() and self.iterationAtAcceptableResult==None):
            self.iterationAtAcceptableResult=self.iterationCount
            self.functionExecutionAtAcceptableResult=self.function.numberOfEvaluations
            self.timeAtAcceptableResult=time.time()


    def iterate(self,iterationCount):
        bestResults=np.array([])
        for i in range(iterationCount):
            self.runIteration()
            bestResults=np.append(bestResults,self.getBestResult())
        return np.reshape(bestResults,(-1,2))

    def getBestResult(self):
        currentResult=deepcopy(self.particles[0])
        for p in self.particles:
            if p.personalBest[1]<currentResult.personalBest[1]:
                currentResult=deepcopy(p)
        return currentResult.personalBest


class Particle:
    def __init__(self, coords, function,informantIndexes, chi=0.729844, phiMax=4.1):
        self.function=function
        self.current=[coords,self.function.evaluate(coords)]#current coordinates
        self.personalBest=deepcopy(self.current)#pbest
        #self.bestKnown=deepcopy(self.current)#gbest or lbest, depending on topology
        self.inertiaVector=np.zeros(coords.shape)
        self.informantIndexes=informantIndexes#neighborhood particle indexes

        self.chi=chi
        self.phiMax=phiMax
    
    # def checkNeighborhood(self, allParticles):
    #     for i in self.informantIndexes:
    #         if(allParticles[i].personalBest[1]<self.bestKnown[1]):
    #             self.bestKnown=deepcopy(allParticles[i].personalBest)


    def move(self, allParticles, velocityClamp=None):
        inertiaVector = deepcopy(self.inertiaVector)

        upperSum=0
        lowerSum=0
        phi=0

        phiPerElement=self.phiMax/len(self.informantIndexes)
        
        for i in self.informantIndexes:
            pb=deepcopy(allParticles[i])
            phiI=np.random.rand()*phiPerElement
            phi+=phiI
            upperSum+=phiI*pb.personalBest[0]/pb.personalBest[1]
            lowerSum+=phiI/pb.personalBest[1]
        
        
        attractorPoint=upperSum/lowerSum
        movementVector = self.chi*(inertiaVector+phi*(attractorPoint-self.current[0]))
        self.current[0]=self.current[0]+movementVector
            
        self.inertiaVector=movementVector
        
        self.current[1]=self.function.evaluate(self.current[0])

        if(self.current[1]<self.personalBest[1]):
            self.personalBest=deepcopy(self.current)
        # if(self.current[1]<self.bestKnown[1]):
        #     self.bestKnown=deepcopy(self.current)

