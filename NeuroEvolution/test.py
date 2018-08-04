import gym
from gym import wrappers
import numpy as np
import random
from time import sleep
from statistics import mean

populationSize = 10
envName = "CartPole-v0"
class Population:
    def __init__(self):
        self.weights = np.random.uniform(-1.0, 1.0, 4)
        self.surviveSteps = 0
        self.fitness = 0
        self.lifeFitness = 0
        self.lifeSteps = 0
    
class Evolution:
    
    def __init__(self, Popsize, envName):
        print('object made')
        self.size = Popsize
        self.env = gym.make(envName)
    def mutate(self, pop):
        num1 = random.randrange(0, len(pop))
        num2 = random.randrange(0, len(pop))

        mutateAgent1 = pop[num1]
        mutateAgent2 = pop[num2]

        index1 = random.randrange(len(mutateAgent1.weights))
        index2 = random.randrange(len(mutateAgent2.weights))
        
        mutateAgent1.weights[index1] = random.uniform(-1.0, 1.0)
        mutateAgent2.weights[index2] = random.uniform(-1.0, 1.0)

        
            
    def breed(self, pop):
        weightOption = [1, 2, 3, 4]
        for i in range(self.size - len(pop)):
            newAgent = Population()
            randParent1 = pop[random.randrange(0,len(pop))]
            randParent2 = pop[random.randrange(0, len(pop))]

            par1Weights = randParent1.weights
            par2Weights = randParent2.weights

            numChange = random.randrange(0,len(par1Weights))

            for i in range(numChange):
                randomSpot = random.randrange(0, len(par1Weights))
                randomPar = random.randrange(0,2)
                if randomPar == 0:
                    newAgent.weights[randomSpot] = par1Weights[randomSpot]
                else:
                    newAgent.weights[randomSpot] = par2Weights[randomSpot]

            pop.append(newAgent)

        return pop
            

    def runGeneration(self, pop):
        for i in pop:
            observation = self.env.reset()
            for j in range(self.reqSteps):
                action = 1 if np.dot(observation, i.weights) > 0 else 0
                observation, reward, checkDone, info = self.env.step(action)

                i.surviveSteps += 1
                i.fitness += reward
                
                if checkDone == True:
                    break

    def getFittest(self, pop):
        newPop = []
        scores = []

        #Get fitness scores for populations
        scores, _ = self.getScoresAndSteps(pop)
        
        #Get agents top 2 fitness level weights    
        newPop.append(pop[np.argmax(scores)])
        pop.remove(pop[scores.index(max(scores))])
        scores.remove(max(scores))
        newPop.append(pop[np.argmax(scores)])
        pop.remove(pop[scores.index(max(scores))])
        scores.remove(max(scores))


        #Choose one random agent from remaining population 

        num = random.randrange(0,len(scores))

        newPop.append(pop[num])
        
        return newPop
        

    def initPop(self):
        self.pop = []
        for i in range(self.size):
            self.pop.append(Population())
        return self.pop


    def checkDone(self, pop):
        checkDone = False
        for i in pop:
            if(i.surviveSteps >= self.reqSteps):
                checkDone = True
        return checkDone

    def getWeights(self, pop):
        w = []
        for i in pop:
            w.append(i.weights)
        return w

    def getScoresAndSteps(self,pop):
        scores = []
        steps = []
        for i in pop:
            scores.append(i.fitness)
            steps.append(i.surviveSteps)

        return scores, steps


    def maxScoreAndStep(self, scores, steps):
        bestScore = max(scores)
        bestSteps = scores[scores.index(max(scores))]

        return bestScore, bestSteps

    def resetStepsAndFitness(self, pop):
        for i in pop:
            i.lifeSteps += i.surviveSteps
            i.lifeFitness += i.fitness
            i.surviveSteps = 0
            i.fitness = 0
    
    def runEvolution(self):
        self.pop = self.initPop()

        self.reqSteps = 200
        self.bestSteps = 0
        done = False
        self.numGen = 0

        while not done:
            
            self.runGeneration(self.pop)
            scores, steps = self.getScoresAndSteps(self.pop)
            self.maxScore, self.maxScoreSteps = self.maxScoreAndStep(scores, steps) 
            self.pop = self.getFittest(self.pop)
            done = self.checkDone(self.pop)
            self.pop = self.breed(self.pop)
            self.mutate(self.pop)
            self.resetStepsAndFitness(self.pop)

            self.numGen += 1

            

            print("Generation number {} had the highest score of {} for {} steps".format(self.numGen, self.maxScore, self.maxScoreSteps))
            
        self.weights = self.getWeights(self.pop)
        scores, steps = self.getScoresAndSteps(self.pop)
        self.bestScore, self.bestSteps = self.maxScoreAndStep(scores, steps)
        
        

        return self.weights, self.bestScore, self.bestSteps

        
geneticEvolution = Evolution(populationSize, envName)
bestWeights, bestScore, bestSteps = geneticEvolution.runEvolution()


print(bestWeights, bestScore)



env = gym.make(envName)
#env = wrappers.Monitor(env, "C:/Users/rohit/Desktop/Genetic Algorithm", force=False)


allMeans = []
for i in range(len(bestWeights)):
    weights = bestWeights[i]
    allGameScores = []
    for _ in range(100):
        done = False
        observation = env.reset()
        score = 0
        count = 0
        while not done:
            #env.render()
            action = 1 if np.dot(observation, weights) > 0 else 0
            observation, reward, done, info = env.step(action)

            score += reward
            count += 1
        print("Game lasted for {} moves with a score of {}".format(score, count))
        allGameScores.append(score)

    print(mean(allGameScores))
    allMeans.append(mean(allGameScores))










