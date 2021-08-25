from copy import deepcopy
import numpy as np
import time
class HEA:
    def __init__(self, function, particleCount, dimensions,minIterationCount,maxIterationCount, c1=2, c2=2, inertiaWeightMax=0.9, inertiaWeightMin=0.4, reproductionRate=0.7, crossoverRate=0.25, mutationRate=0.05, topology="gbest"):
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

        self.reproductionRate=reproductionRate
        self.crossoverRate=crossoverRate
        self.mutationRate=mutationRate

        particleCoords=np.random.default_rng().uniform(low=-self.xMax,high=self.xMax, size=(self.particleCount, self.dimensions))

        self.particles=[]
        index=0
        #Generate particles with gbest topology
        if topology=="gbest":
            indexesOfP=list(range(0,particleCount))
            for p in particleCoords:
                self.particles.append(Particle(p,self.function,indexesOfP,index))
                index+=1
        #Generate particles with lbest topology
        elif isinstance(topology, int):
            allIndexes=list(range(0,particleCount))
            for i,p in enumerate(particleCoords):
                indexesOfP=[allIndexes[i]]
                for j in range(topology//2):
                    indexesOfP.append(allIndexes[(i+j+1)%self.particleCount])
                    indexesOfP.append(allIndexes[i-j-1])
                self.particles.append(Particle(p,self.function,indexesOfP,index))
                index+=1
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
                self.particles.append(Particle(p,self.function,indexesOfP,index))
                index+=1
        
        self.checkNeighborhood()

    def checkNeighborhood(self):
        for p in self.particles:
            p.checkNeighborhood(self.particles)

    def runIteration(self):
        for p in self.particles:#do PSO
            p.move(self.c1, self.c2,self.dimensions,self.xMax,self.calculateLinearInertiaWeight())

        self.fitnessSet=[]
        probabilities=[]
        for p in self.particles:
            self.fitnessSet.append(p.current[1])#F(X)

        fitnessSum=sum(self.fitnessSet)
        for f in self.fitnessSet:
            if(fitnessSum==0):
                probabilities.append(1/self.particleCount)
            else:
                probabilities.append(f/fitnessSum)#p_i
        probabilities.insert(0,0)

        newPopulation=[]
        for k in range(0,self.particleCount,2):#generating new population
            #pick two parents
            indexa=0
            indexb=0
            indexa=self.rouletteSelectIndex(probabilities,np.random.rand())
            indexb=self.rouletteSelectIndex(probabilities,np.random.rand())
            while (indexa==indexb):
                randb=np.random.rand()
                indexb=self.rouletteSelectIndex(probabilities,randb)
            xa=self.particles[indexa]
            xb=self.particles[indexb]

            r=np.random.rand()
            if(r<=self.reproductionRate):#reproduction (does nothing)
                newPopulation.append(xa)
                newPopulation.append(xb)
            elif (r<=self.reproductionRate+self.crossoverRate):#crossover
                rArr=np.random.uniform(0,1,self.dimensions)
                childACoords=np.multiply(rArr,xa.current[0]) +np.multiply(np.ones(self.dimensions)-rArr,xb.current[0])
                childBCoords=np.multiply(rArr,xb.current[0]) +np.multiply(np.ones(self.dimensions)-rArr,xa.current[0])
                evalA=self.function.evaluate(childACoords)
                evalB=self.function.evaluate(childBCoords)
                xa.current=[childACoords,evalA]
                xb.current=[childBCoords,evalB]
                if(xa.current[1]<xa.personalBest[1]):
                    xa.personalBest=[childACoords,evalA]
                if(xa.current[1]<xa.bestKnown[1]):
                    xa.bestKnown=[childACoords,evalA]
                if(xb.current[1]<xb.personalBest[1]):
                    xb.personalBest=[childBCoords,evalB]
                if(xb.current[1]<xb.bestKnown[1]):
                    xb.bestKnown=[childBCoords,evalB]
                newPopulation.append(xa)
                newPopulation.append(xb)
            else:#Gaussian mutation
                childACoords=xa.current[0]*(1+np.random.normal(0,np.sqrt(0.1*2*self.function.getXMax())))
                childBCoords=xb.current[0]*(1+np.random.normal(0,np.sqrt(0.1*2*self.function.getXMax())))
                evalA=self.function.evaluate(childACoords)
                evalB=self.function.evaluate(childBCoords)
                xa.current=[childACoords,evalA]
                xb.current=[childBCoords,evalB]
                if(xa.current[1]<xa.personalBest[1]):
                    xa.personalBest=[childACoords,evalA]
                if(xa.current[1]<xa.bestKnown[1]):
                    xa.bestKnown=[childACoords,evalA]
                if(xb.current[1]<xb.personalBest[1]):
                    xb.personalBest=[childBCoords,evalB]
                if(xb.current[1]<xb.bestKnown[1]):
                    xb.bestKnown=[childBCoords,evalB]
                newPopulation.append(xa)
                newPopulation.append(xb)
        self.particles=newPopulation
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

    def rouletteSelectIndex(self, a,p):
        for i in range(1,len(a),1):
            k1=round(sum(a[0:i-1]),5)
            k2=round(sum(a[1:i]),5)
            if p>k1 and p<=k2:
                rval=i-2
                return rval
        return len(a)-2

    def calculateLinearInertiaWeight(self):
        weight = self.inertiaWeightMin+(self.inertiaWeightMax-self.inertiaWeightMin)*(1-(self.iterationCount/self.maxIterations))
        return weight

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
            if p.bestKnown[1]<self.particles[bestKnownIndex].bestKnown[1]:
                bestKnownIndex=i
        return self.particles[bestKnownIndex].bestKnown

class Particle:
    def __init__(self, coords, function,informantIndexes,index):
        self.function=function
        self.index=index
        eval=self.function.evaluate(coords)
        self.current=[coords,eval]#current coordinates
        self.personalBest=[coords,eval]#pbest
        self.bestKnown=[coords,eval]#gbest or lbest, depending on topology
        self.inertiaVector=np.zeros(coords.shape)
        self.informantIndexes=informantIndexes#neighborhood particle indexes
    
    def checkNeighborhood(self, allParticles):
        betterIndex=None
        for i in self.informantIndexes:
            if(allParticles[i].personalBest[1]<self.bestKnown[1]):
                betterIndex=i
        if(betterIndex!=None):
            self.bestKnown=deepcopy(allParticles[betterIndex].personalBest)

    def move(self, c1, c2, dimensions, velocityClamp, inertiaWeight):
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

