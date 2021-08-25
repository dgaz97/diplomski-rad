from Functions import SphereFunction,RastriginFunction,Zakharov,Griewank,Rosenbrock
from PSOAlgorithms.OriginalPSO import OriginalPSO
from PSOAlgorithms.StandardPSO import StandardPSO
from PSOAlgorithms.PSON import PSON
from PSOAlgorithms.PSOWN import PSOWN
from PSOAlgorithms.HEA import HEA
from PSOAlgorithms.SDEA import SDEA
from PSOAlgorithms.PSOSA import PSOSA
from PSOAlgorithms.TRIBES import TRIBES
import numpy as np
import time

functions=[Griewank,Zakharov,Rosenbrock]
#PSO
particleCount=20
minIterationCount=1000
maxIterationCount=10000
dimensions=30
c1=2.05
c2=2.05
inertiaWeightMax=0.9
inertiaWeightMin=0.4
#Genetic algorithm
reproductionRate=0.7
crossoverRate=0.25
mutationRate=0.05
#Differential evolution
differentialEvolutionChance=0.25
crossoverProbability=0.5
mutationRate=0.8
#Simulated annealing
maxN1=5
maxN2=5
eta=1
#TRIBES
usePseudogradients=True

topologies=[3,5,"vonNeumann"]

numberOfRounds=20


startTime=time.time()
f=open(F"./Results.txt","w",buffering=1)
f.write(F"function,topology,algorithm,timeToEnd,evaluationsToEnd,iterationsToEnd,timeToMinIter,evaluationsToMinIter,resultAtMinIter,iterationsAtAcceptableError,evaluationsAtAcceptableError,resultAtEnd\n")

for function in functions:
    print(F"Function: {function.__name__}")

    for t in topologies:
        print(F"Topology: {t}")
        for i in range(numberOfRounds):
            print(F"Original PSO round {i+1} starts")
            function.numberOfEvaluations=0
            algorithm = OriginalPSO(function,particleCount,dimensions,minIterationCount,maxIterationCount,c1,c2,t)
            start=time.time()
            bestResults=algorithm.iterate()
            end=time.time()

            f.write(F"{function.__name__},{t},{OriginalPSO.__name__},{end-start},{function.numberOfEvaluations},{algorithm.iterationCount},{algorithm.timeAtMinIter-start},{algorithm.evaluationsAtMinIter},{algorithm.resultAtMinIter[1]},{algorithm.iterationAtAcceptableResult},{algorithm.functionExecutionAtAcceptableResult},{algorithm.getBestResult()[1]}\n")
            f.flush()

    for t in topologies:
        print(F"Topology:{t}")
        for i in range(numberOfRounds):
            print(F"{StandardPSO.__name__} round {i+1} starts")
            function.numberOfEvaluations=0
            algorithm = StandardPSO(function,particleCount,dimensions,minIterationCount,maxIterationCount,c1,c2,inertiaWeightMax,inertiaWeightMin,t)
            start=time.time()
            bestResults=algorithm.iterate()
            end=time.time()

            f.write(F"{function.__name__},{t},{StandardPSO.__name__},{end-start},{function.numberOfEvaluations},{algorithm.iterationCount},{algorithm.timeAtMinIter-start},{algorithm.evaluationsAtMinIter},{algorithm.resultAtMinIter[1]},{algorithm.iterationAtAcceptableResult},{algorithm.functionExecutionAtAcceptableResult},{algorithm.getBestResult()[1]}\n")
            f.flush()

    for t in topologies:
        print(F"Topology:{t}")
        for i in range(numberOfRounds):
            print(F"{PSON.__name__} round {i+1} starts")
            function.numberOfEvaluations=0
            algorithm = PSON(function,particleCount,dimensions,minIterationCount,maxIterationCount,t)
            start=time.time()
            bestResults=algorithm.iterate()
            end=time.time()

            f.write(F"{function.__name__},{t},{PSON.__name__},{end-start},{function.numberOfEvaluations},{algorithm.iterationCount},{algorithm.timeAtMinIter-start},{algorithm.evaluationsAtMinIter},{algorithm.resultAtMinIter[1]},{algorithm.iterationAtAcceptableResult},{algorithm.functionExecutionAtAcceptableResult},{algorithm.getBestResult()[1]}\n")
            f.flush()

    for t in topologies:
        print(F"Topology:{t}")
        for i in range(numberOfRounds):
            print(F"{HEA.__name__} round {i+1} starts")
            function.numberOfEvaluations=0
            algorithm = HEA(function,particleCount,dimensions,minIterationCount,maxIterationCount,c1,c2,inertiaWeightMax,inertiaWeightMin, reproductionRate, crossoverRate, mutationRate,t)
            start=time.time()
            bestResults=algorithm.iterate()
            end=time.time()

            f.write(F"{function.__name__},{t},{HEA.__name__},{end-start},{function.numberOfEvaluations},{algorithm.iterationCount},{algorithm.timeAtMinIter-start},{algorithm.evaluationsAtMinIter},{algorithm.resultAtMinIter[1]},{algorithm.iterationAtAcceptableResult},{algorithm.functionExecutionAtAcceptableResult},{algorithm.getBestResult()[1]}\n")
            f.flush()

    for t in topologies:
        print(F"Topology:{t}")
        for i in range(numberOfRounds):
            print(F"{SDEA.__name__} round {i+1} starts")
            function.numberOfEvaluations=0
            algorithm = SDEA(function,particleCount,dimensions,minIterationCount,maxIterationCount,c1,c2,inertiaWeightMax,inertiaWeightMin, differentialEvolutionChance,crossoverProbability,mutationRate,t)
            start=time.time()
            bestResults=algorithm.iterate()
            end=time.time()

            f.write(F"{function.__name__},{t},{SDEA.__name__},{end-start},{function.numberOfEvaluations},{algorithm.iterationCount},{algorithm.timeAtMinIter-start},{algorithm.evaluationsAtMinIter},{algorithm.resultAtMinIter[1]},{algorithm.iterationAtAcceptableResult},{algorithm.functionExecutionAtAcceptableResult},{algorithm.getBestResult()[1]}\n")
            f.flush()

    for t in topologies:
        print(F"Topology:{t}")
        for i in range(numberOfRounds):
            print(F"{PSOSA.__name__} round {i+1} starts")
            function.numberOfEvaluations=0
            algorithm = PSOSA(function,particleCount,dimensions,minIterationCount,maxIterationCount,c1,c2,inertiaWeightMax,inertiaWeightMin, maxN1, maxN2, eta, t)
            start=time.time()
            bestResults=algorithm.iterate()
            end=time.time()

            f.write(F"{function.__name__},{t},{PSOSA.__name__},{end-start},{function.numberOfEvaluations},{algorithm.iterationCount},{algorithm.timeAtMinIter-start},{algorithm.evaluationsAtMinIter},{algorithm.resultAtMinIter[1]},{algorithm.iterationAtAcceptableResult},{algorithm.functionExecutionAtAcceptableResult},{algorithm.getBestResult()[1]}\n")
            f.flush()


    for i in range(numberOfRounds):
        print(F"{TRIBES.__name__} round {i+1} starts")
        function.numberOfEvaluations=0
        algorithm = TRIBES(function,minIterationCount,maxIterationCount,dimensions,usePseudogradients)
        start=time.time()
        bestResults=algorithm.iterate()
        end=time.time()

        f.write(F"{function.__name__},None,{TRIBES.__name__},{end-start},{function.numberOfEvaluations},{algorithm.iterationCount},{algorithm.timeAtMinIter-start},{algorithm.evaluationsAtMinIter},{algorithm.resultAtMinIter[1]},{algorithm.iterationAtAcceptableResult},{algorithm.functionExecutionAtAcceptableResult},{algorithm.getBestResult()[1]}\n")
        f.flush()
f.close()

endTime=time.time()
print(F"Done after {endTime-startTime}")