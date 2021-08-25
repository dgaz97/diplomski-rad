from copy import deepcopy
import numpy as np
import time
class TRIBES:
    def __init__(self, function,minIterationCount,maxIterationCount, dimensions, pseudogradient):
        self.function=function
        self.dimensions=dimensions
        self.xMax=self.function.getXMax()
        self.minIterations=minIterationCount
        self.maxIterations=maxIterationCount

        self.particleIndexes=0
        self.tribeIndexes=0

        self.L=1
        self.tribes=[0]

        self.timeAtAcceptableResult=None
        self.iterationAtAcceptableResult=None
        self.functionExecutionAtAcceptableResult=None;
        self.pseudogradient=pseudogradient

        self.iterationCount=0


        self.particles=[]
        particleCoords=np.random.default_rng().uniform(low=-self.xMax,high=self.xMax, size=(self.dimensions))
        self.particles.append(Particle(deepcopy(self.particleIndexes),particleCoords,self.function,deepcopy(self.tribeIndexes),[],[],self.pseudogradient))
        self.particleIndexes+=1
        self.tribeIndexes+=1

        
        


    def checkNeighborhood(self):
        for p in self.particles:
            p.checkNeighborhood(self.particles)

    def getConnections(self):
        l=0
        for p in self.particles:
            l+=len(p.internalInformants)
            l+=len(p.externalInformants)
        l/=2
        l+=len(self.particles)
        return l

    def runIteration(self):
        goodTribes=[]
        badTribes=[]

        self.L-=1
        
        if self.L<0:#swarm adaptation
            for i in self.tribes:
                tribeMembers=list(filter(lambda x:(x.tribe==i),self.particles))
                goodMembers=list(filter(lambda x:(x.previousMemory==1),tribeMembers))
                if(len(tribeMembers)==0):
                    self.tribes.remove(i)
                    continue
                p=np.random.randint(len(tribeMembers))
                if(len(goodMembers)>p):
                    goodTribes.append(i)
                else:
                    badTribes.append(i)
            if(len(self.particles)>2):
                for i in goodTribes:#delete particles
                    tribeMembers=list(filter(lambda x:(x.tribe==i),self.particles))#get particles of tribe
                    if(len(tribeMembers)>1):
                        tribeMembers.sort(reverse=True,key=lambda x:(x.bestKnown[1]))#sort them by personal best
                        worstParticle=tribeMembers[0]
                        bestParticle=tribeMembers[-1]
                        bestParticle.externalInformants=list(set(bestParticle.externalInformants+worstParticle.externalInformants))

                        #print(self.iterationCount,i,worstParticle.index,"delete from big tribe")
                        if (len(worstParticle.externalInformants)==0):#if particle doesn't have external connections
                            pass#do nothing
                        else:
                            for link in worstParticle.externalInformants:
                                externalParticle=list(filter(lambda x:(x.index==link),self.particles))[0]

                                externalParticle.externalInformants.append(deepcopy(bestParticle.index))
                                bestParticle.externalInformants.append(deepcopy(externalParticle.index))
                        
                        for p in self.particles:
                            try:
                                p.internalInformants.remove(worstParticle.index)
                            except ValueError:
                                pass
                            try:
                                p.externalInformants.remove(worstParticle.index)
                            except ValueError:
                                pass
                            p.externalInformants=list(set(p.externalInformants))
                        self.particles = [i for i in self.particles if i.index!=worstParticle.index]
                    else:
                        worstParticle=tribeMembers[0]
                        connectedExternalParticles=list(filter(lambda x:(worstParticle.index in x.externalInformants),self.particles))

                        
                        #print(self.iterationCount,i,worstParticle.index,"delete from small tribe")

                        betterInformantExists=0#delete only if a better informant exists
                        for p in connectedExternalParticles:
                            if(p.bestKnown[1]<tribeMembers[0].bestKnown[1]):
                                betterInformantExists=1
                                break
                        if betterInformantExists:
                            for p in connectedExternalParticles:
                                otherConnections=[x for x in connectedExternalParticles if x.index!=p.index]

                                othersOfTribe = list(filter(lambda x:(x.tribe==p.tribe),self.particles))
                                for p2 in otherConnections:
                                    othersOfAnotherTribe = list(filter(lambda x:(x.tribe==p2.tribe),self.particles))
                                    anotherConnectionExists=0
                                    for p3 in othersOfTribe:
                                        for p4 in othersOfAnotherTribe:
                                            if(p3.index in p4.externalInformants):
                                                anotherConnectionExists=1
                                                break
                                            if(anotherConnectionExists):
                                                break
                                        if(anotherConnectionExists):
                                            break
                                    if (not anotherConnectionExists):
                                        p.externalInformants.append(deepcopy(p2.index))
                                        p2.externalInformants.append(deepcopy(p.index))
                                    try:
                                        p2.externalInformants.remove(worstParticle.index)
                                    except ValueError:
                                        pass
                                try:
                                    p.externalInformants.remove(worstParticle.index)
                                except ValueError:
                                    pass

                            self.particles = [j for j in self.particles if j.index!=worstParticle.index]#delete particle
                            self.tribes=[j for j in self.tribes if j!=i]#delete empty tribe
            

            newParticles=[]
            newIndexes=[]
            for i in badTribes:#generate tribes
                #print("create",self.iterationCount)
                tribeMembers=list(filter(lambda x:(x.tribe==i),self.particles))#get particles of tribe
                tribeMembers.sort(reverse=True,key=lambda x:(x.bestKnown[1]))#sort them by personal best
                bestMember=tribeMembers[-1]

                #unbound particle
                newCoords=np.random.default_rng().uniform(low=-self.xMax,high=self.xMax, size=(self.dimensions))
                generationRule=np.random.randint(3)
                if(generationRule==0):#completely randomly
                    pass
                elif(generationRule==1):#on side of search space
                    dimensionToChange=np.random.randint(self.dimensions)
                    prefix=-1 if np.random.randint(2) else 1
                    newCoords[dimensionToChange]=prefix*self.function.getXMax()
                elif(generationRule==2):#on edge of search space
                    dimensionToChange1=np.random.randint(self.dimensions)
                    dimensionToChange2=np.random.randint(self.dimensions)
                    while(dimensionToChange2==dimensionToChange1):
                        dimensionToChange2=np.random.randint(self.dimensions)
                    prefix1=-1 if np.random.randint(2) else 1
                    prefix2=-1 if np.random.randint(2) else 1
                    newCoords[dimensionToChange1]=prefix1*self.function.getXMax()
                    newCoords[dimensionToChange2]=prefix2*self.function.getXMax()
                newParticles.append(Particle(deepcopy(self.particleIndexes),newCoords,self.function,deepcopy(self.tribeIndexes),[],[deepcopy(bestMember.index)],self.pseudogradient))
                newIndexes.append(deepcopy(self.particleIndexes))
                bestMember.externalInformants.append(deepcopy(self.particleIndexes))
                self.particleIndexes+=1

                #bound particle
                informantOfBestParticle=list(filter(lambda x:(bestMember.index in x.internalInformants or bestMember.index in x.externalInformants),self.particles))#get informants of best member
                informantOfBestParticle.sort(reverse=True,key=lambda x:(x.bestKnown[1]))#sort them by personal best
                if(len(informantOfBestParticle)==0):#no informants, so can't generate bound particle
                    pass
                else:
                    bestInformant=deepcopy(informantOfBestParticle[-1])
                    distance=np.sqrt(np.sum(np.square(bestInformant.bestKnown[0]-bestMember.bestKnown[0])))
                    r=np.random.default_rng().uniform(low=-distance,high=distance, size=(self.dimensions))
                    newCoords2=deepcopy(bestInformant.bestKnown[0])+r
                    newParticles.append(Particle(deepcopy(self.particleIndexes),newCoords2,self.function,deepcopy(self.tribeIndexes),[],[],self.pseudogradient))
                    newIndexes.append(deepcopy(self.particleIndexes))
                    self.particleIndexes+=1

            self.tribes.append(self.tribeIndexes)
            self.tribeIndexes+=1
            for p in newParticles:
                p.internalInformants=[i for i in newIndexes if i!=p.index]

            self.particles=np.append(self.particles,newParticles)
            self.L=self.getConnections()//2

        for p in self.particles:
            p.move(self.iterationCount)

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
        self.bestResults=np.array([])
        for i in range(self.maxIterations):
            self.iterationCount+=1
            self.runIteration()
            self.bestResults=np.append(self.bestResults,self.getBestResult())
            if(self.bestResults[-1]<self.function.getAcceptableError() and self.iterationCount>=self.minIterations):
                return np.reshape(self.bestResults,(-1,2))
        return np.reshape(self.bestResults,(-1,2))

    
    def getBestResult(self):
        bestKnownIndex=0
        for i,p in enumerate(self.particles):
            if p.bestKnown[1]<self.particles[bestKnownIndex].bestKnown[1]:
                bestKnownIndex=i
        
        return self.particles[bestKnownIndex].bestKnown

class Particle:
    def __init__(self,index, coords, function, tribe, internalInformants, externalInformants, pseudogradient):
        self.index=index
        self.function=function
        self.previousMemory=0
        self.pastPreviousMemory=0
        eval=self.function.evaluate(coords)
        self.current=[coords,eval]#current coordinates
        self.personalBest=[coords,eval]#pbest
        self.bestKnown=[coords,eval]#gbest or lbest, depending on topology
        self.tribe=tribe
        self.pseudogradient=pseudogradient

        self.inertiaVector=np.zeros(coords.shape)
        self.internalInformants=internalInformants#particle indexes that are in same tribe
        self.externalInformants=externalInformants#particle indexes that are not in same tribe
    

    def defineStrategy(self):
        if (self.pastPreviousMemory,self.previousMemory) in ((-1,-1),(0,-1),(1,-1),(-1,-0),(0,0)):
            return self.pivot
        elif (self.pastPreviousMemory,self.previousMemory) in ((1,0),(-1,1)):
            return self.noisyPivot
        elif (self.pastPreviousMemory,self.previousMemory) in ((0,1),(1,1)):
            return self.localGauss
    
    
    def pivot(self):
        dist=np.sqrt(np.sum(np.square(self.personalBest[0]-self.bestKnown[0])))
        dimensions=self.personalBest[0].size
        
        distance=dist/dimensions
        
        r1=np.random.default_rng().uniform(low=-distance,high=distance, size=(dimensions))
        r2=np.random.default_rng().uniform(low=-distance,high=distance, size=(dimensions))
        point1=np.minimum(np.maximum(self.personalBest[0]+r1,-self.function.getXMax()),self.function.getXMax())
        point2=np.minimum(np.maximum(self.bestKnown[0]+r2,-self.function.getXMax()),self.function.getXMax())

        c1=self.personalBest[1]/(self.personalBest[1]+self.bestKnown[1])
        c2=self.bestKnown[1]/(self.personalBest[1]+self.bestKnown[1])

        x=np.minimum(np.maximum(c1*point1+c2*point2,-self.function.getXMax()),self.function.getXMax())

        return x


    def noisyPivot(self):
        dist=np.sqrt(np.sum(np.square(self.personalBest[0]-self.bestKnown[0])))
        dimensions=self.personalBest[0].size
        
        distance=dist/dimensions

        r1=np.random.default_rng().uniform(low=-distance,high=distance, size=(dimensions))
        r2=np.random.default_rng().uniform(low=-distance,high=distance, size=(dimensions))
        point1=np.minimum(np.maximum(self.personalBest[0]+r1,-self.function.getXMax()),self.function.getXMax())
        point2=np.minimum(np.maximum(self.bestKnown[0]+r2,-self.function.getXMax()),self.function.getXMax())

        c1=self.personalBest[1]/(self.personalBest[1]+self.bestKnown[1])
        c2=self.bestKnown[1]/(self.personalBest[1]+self.bestKnown[1])

        x=np.minimum(np.maximum(c1*point1+c2*point2,-self.function.getXMax()),self.function.getXMax())

        deviation=(self.personalBest[1]-self.bestKnown[1])/(self.personalBest[1]+self.bestKnown[1])
        noise=np.random.default_rng().normal(0,abs(deviation), size=(dimensions))
        x=x+noise

        return x

    def localGauss(self):
        point=self.bestKnown[0]-self.current[0]
        dimensions=self.personalBest[0].size
        dist=np.sqrt(np.sum(np.square(self.bestKnown[0]-self.current[0])))
        
        distance=dist/dimensions
        
        gauss=np.random.normal(point,distance,dimensions)

        x=np.minimum(np.maximum(self.bestKnown[0]+gauss,-self.function.getXMax()),self.function.getXMax())
        
        return x

    def checkNeighborhood(self, allParticles):
        bestIndex=self.findBestInformant(allParticles,self.pseudogradient)
        if(bestIndex!=None):
            p=list(filter(lambda x:(x.index==bestIndex),allParticles))[0]
            self.bestKnown=p.personalBest

    def move(self,iter):
        self.iteration=iter
        
        
        movementStrategy=self.defineStrategy()
        newLocation=movementStrategy()
        newLocationEvaluation=self.function.evaluate(newLocation)

        self.pastPreviousMemory=self.previousMemory
        if(newLocationEvaluation<self.current[1]):
            self.previousMemory=1
        elif(newLocationEvaluation>self.current[1]):
            self.previousMemory=-1
        else:
            self.previousMemory=0
        self.current=[newLocation,newLocationEvaluation]

        if(newLocationEvaluation<self.personalBest[1]):
            self.personalBest=[newLocation,newLocationEvaluation]
        if(newLocationEvaluation<self.bestKnown[1]):
            self.bestKnown=[newLocation,newLocationEvaluation]

    def findBestInformant(self, allParticles, pseudogradient=False):
        indexes=np.append(self.internalInformants,self.externalInformants)
        if(len(indexes)<2):
            return None
        bestInformantIndex=indexes[0]
        currentBest=list(filter(lambda x:(x.index==bestInformantIndex),allParticles))[0]
        if(pseudogradient==True):
            for i in indexes:
                p=list(filter(lambda x:(x.index==i),allParticles))[0]
                p11=np.abs(self.bestKnown[1]-p.bestKnown[1])
                p12=np.sqrt(np.sum(np.square(self.bestKnown[0]-p.bestKnown[0])))
                part1=p11/p12
                p21=np.abs(self.bestKnown[1]-currentBest.bestKnown[1])
                p22=np.sqrt(np.sum(np.square(self.bestKnown[0]-currentBest.bestKnown[0])))
                part2=p21/p22
                if(part1>part2):
                    bestInformantIndex=i
                    currentBest=list(filter(lambda x:(x.index==i),allParticles))[0]
        else:
            for i in indexes:
                p=list(filter(lambda x:(x.index==i),allParticles))[0]
                if(p.bestKnown[1]<self.bestKnown[1]):
                    bestInformantIndex=i
        return bestInformantIndex





        
        
                
    

