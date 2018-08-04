import gym
from gym import wrappers
import numpy as np
import random
from time import sleep
from statistics import mean

populationSize = 10
envName = "CartPole-v0"

#Creates Agents for the population
class Population:
    def __init__(self):
        self.weights = np.random.uniform(-1.0, 1.0, 4) #Creates random weights
        self.surviveSteps = 0 #Number of steps the model survives for
        self.fitness = 0 #Total fitness the model gets in a game
        self.lifeFitness = 0 #Total fitness for model in all generations (this is required as fitness for the model is reset every generation)
        self.lifeSteps = 0 #Total number of steps for the model across all generations

#Class that carries out the rest of neuroevolution
class Evolution:
    
    def __init__(self, Popsize, envName):
        print('object made')
        self.size = Popsize #Get the population size
        self.env = gym.make(envName) #Create the CartPole environment

    #This mutates the incoming generation. A new generation is formed after mutations.
    def mutate(self, pop):
        num1 = random.randrange(0, len(pop)) #Get a random number
        num2 = random.randrange(0, len(pop)) #Get a random number

        mutateAgent1 = pop[num1] #Use the random number above to get a random model from the population. This model will be mutated
        mutateAgent2 = pop[num2] #Use the random number above to get a random model from the population. This model will be mutated

        index1 = random.randrange(len(mutateAgent1.weights)) #Get a random number 
        index2 = random.randrange(len(mutateAgent2.weights))
        
        mutateAgent1.weights[index1] = random.uniform(-1.0, 1.0) #Using the random number above, choose a value of the weights. Re-assign that weight to a random number 
        mutateAgent2.weights[index2] = random.uniform(-1.0, 1.0)

        
    #This fills creates new agents for the population using the fittest agents that survived.       
    def breed(self, pop):
        #This creates new Agents. The loop makes sure that we are only filling in new agents until the max population size is reached
        for i in range(self.size - len(pop)):
            newAgent = Population() #Create a new agent
            randParent1 = pop[random.randrange(0,len(pop))] #Get a random parents from the fittest population
            randParent2 = pop[random.randrange(0, len(pop))] #Get another random parent

            par1Weights = randParent1.weights #Get parents' weights
            par2Weights = randParent2.weights

            numChange = random.randrange(0,len(par1Weights)) #Choose how many weight values you want to change for the new Agent

            #Loop over the number of weights you want to change (this is also the number of weights the new Agent will inherit from it's parents)
            for i in range(numChange):
                #Get a random index for a weight
                randomSpot = random.randrange(0, len(par1Weights))
                #Get a random number to decide which parents' weights to use
                randomPar = random.randrange(0,2)
                #From a random parents, get a random weight value 
                if randomPar == 0:
                    newAgent.weights[randomSpot] = par1Weights[randomSpot]
                else:
                    newAgent.weights[randomSpot] = par2Weights[randomSpot]

            #Add the new Agent to the population 
            pop.append(newAgent)

        #Return the new population. The size of the new poplation will equal to the older generations' size. This is to make sure that every generation has equal number of Agents
        return pop
            
    #This runs the Agents in a population. This is used to get the fitness for each Agent.
    def runGeneration(self, pop):
        #Takes each agent and makes it play the game
        for i in pop:
            #Reset the environment, get initial observation
            observation = self.env.reset()
            #This is the goal you have set for the Agent. The maximum number of steps it can take is 200.
            for j in range(self.reqSteps):
                action = 1 if np.dot(observation, i.weights) > 0 else 0
                observation, reward, checkDone, info = self.env.step(action)

                i.surviveSteps += 1
                i.fitness += reward
                
                if checkDone == True:
                    break

    #This gets the top 2 fittest models and 1 randomly chosen model (this does not include the top 2 models). The models are chosen based on fitness levels.
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

        #Return the fittest Agents
        return newPop
        
    #Initialize population. Makes the first generation. 
    def initPop(self):
        self.pop = []
        for i in range(self.size):
            self.pop.append(Population())
        return self.pop

    #Checks whether an Agent has reached the score requirement. The maximum score an Agent can get is 200.
    def checkDone(self, pop):
        checkDone = False
        for i in pop:
            if(i.surviveSteps >= self.reqSteps):
                checkDone = True
        return checkDone

    #Returns a list of weights of the population
    def getWeights(self, pop):
        w = []
        for i in pop:
            w.append(i.weights)
        return w

    #Returns a list of populations fitness and number of steps they survived for.
    def getScoresAndSteps(self,pop):
        scores = []
        steps = []
        for i in pop:
            scores.append(i.fitness)
            steps.append(i.surviveSteps)

        return scores, steps

    #Returns the best score and step count in a population
    def maxScoreAndStep(self, scores, steps):
        bestScore = max(scores)
        bestSteps = scores[scores.index(max(scores))]

        return bestScore, bestSteps

    #Resets the fitness and steps an Agent has for every new generation
    def resetStepsAndFitness(self, pop):
        for i in pop:
            i.lifeSteps += i.surviveSteps
            i.lifeFitness += i.fitness
            i.surviveSteps = 0
            i.fitness = 0

    #This is the main loop that uses all the functions above to conduct neuroevolution
    def runEvolution(self):
        #Initializes population
        self.pop = self.initPop()

        #Number of steps an Agents in a population must survive for.
        self.reqSteps = 200
        #Checkpoint to see whether neuroevolution has converged
        done = False
        #Checkpoint for the number of generations done
        self.numGen = 0

        #Keeps running until an Agents reaches minimum score requirement
        while not done:
            #Runs a generation for getting fitness of every Agent
            self.runGeneration(self.pop)
            #Gets every agents fitness level
            scores, steps = self.getScoresAndSteps(self.pop)
            #Gets the maximum fitness score. This gets printed later at the end of every generation
            self.maxScore, self.maxScoreSteps = self.maxScoreAndStep(scores, steps)
            #Changes the population to only keep the fittest Agents
            self.pop = self.getFittest(self.pop)
            #Checks whether an Agent has successfully got the minimum score requirement
            done = self.checkDone(self.pop)
            #Breeds a new generation if the variable done above is still False. In other words, it breeds new Agents if we have no Agent that is fit enough.
            self.pop = self.breed(self.pop)
            #Mutates the new generation
            self.mutate(self.pop)
            #Resets the fitness scores for every agent in the new generation
            self.resetStepsAndFitness(self.pop)

            #Increases the generation number
            self.numGen += 1

            

            print("Generation number {} had the highest score of {} for {} steps".format(self.numGen, self.maxScore, self.maxScoreSteps))

        #Gets the surviving populations weights
        self.weights = self.getWeights(self.pop)
        #Gets the surviving populations scores
        scores, steps = self.getScoresAndSteps(self.pop)
        #Gets the best score
        self.bestScore, self.bestSteps = self.maxScoreAndStep(scores, steps)
        
        

        return self.weights, self.bestScore, self.bestSteps

#Makes an object from class Evolution
geneticEvolution = Evolution(populationSize, envName)
#Returns the results from neuroevolution
bestWeights, bestScore, bestSteps = geneticEvolution.runEvolution()


print(bestWeights, bestScore)



env = gym.make(envName)
#env = wrappers.Monitor(env, "C:/Users/rohit/Desktop/Genetic Algorithm", force=False)


#This tests the weights returned from neuroevolution

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










