# CartPole challenge by OpenAI solved using NeuroEvolution
This example of neuroevolution was able to get a maximum score of 200 for all 100 consecutive trails. This method only shows basics of neuroevolution. This example does not use Deep Learning in any sort of way. This is a good way to start to get familiar with the comcepts behind neuroevolution. This project was able to run very well only after 4 generations. If you don't understand what that means, don't worry. This repository will tell you. 

## Getting Started 
To get this running, you need to clone only this part of the repository related to neuroevolution. 
### Prerequisites
You will need the following modules installed for python - 
* Gym (toolkit for developing reinforcement learning algorithms)
* Numpy (library that supports multi-dimensional arrays and matrices)
* Random (pseudo-random number generators for various distributions)
* Statistics (optional - calculates mathematical statistics of numeric data)

### Installing
I have a tutorial for installing gym. This also includes other subsets of gym, such as the atari subset. The tutorial works for Windows too (especially Windows users who want atari games). To install the rest, use the following commands on the terminal - 

    pip install numpy 
    pip install statistics
    
##  Running the program
You can call the program on the command line or in IDLE. The program isn't very dynamic. A lot of the code is messy. After this tutorial, I will leave it up to you for further editing the program.

## How it works
Neuroevolution is now a popular tool in AI that seems to work very well. Compared to most learning algorithms, neuroevolution can produce extremely accurate models in short amounts of time. In essence, neuroevolution is trying to replicate biological evolution in learning algorithms. Deep learning started off as a project to mimic the human brain. It produced artificial neural networks that are **"artificial brains"**. Therefore, articial evolution could become the next breakthrough. 

### Things to know about gym
Gym is made by OpenAI for the development of reinforcement learning. To use gym, you can do the following commands - 

    import gym #Imports the module
    
    env = gym.make("CartPole-v0") #This specifies the game we want to make
    env.reset() #You have to reset the game everytime before starting a new one
    observation = env.reset() #This resets the game and also gives an initial observation. More about this below
    
    done = False
    
    if not done:
        env.render() #This is used to visualize the game
        action = env.action_sample.sample() #This gets a random action that can be made in the game
        observation, rewards, done, info = env.step(action) #This is to perform that action. It returns multiple values. Explained below
        
        
So, to make sense of what is going on, we are making random moves in the game right now. We choose a random action, and we perform it in the game. In doing so, we get a reward. The goal of the game is to make sure that the CartPole is upright, till we get a total reward of 200 in 100 consecutive games. Some information on whats going on is below.
#### CartPole Structure
![GitHub Logo](/NeuroEvolution/Videos/cart.jpg)
#### Observations 
Observations is a list that will be used to optimize our model. We get an observation after every step.

Num|Observation
-------|--------
0|Cart Position X-axis
1|Cart Velocity
2|Pole Angle
3|Pole Velocity at tip

#### Actions
Actions are specified by a number. This number determines what the cart should do in the game.

Num|Action
-------|----------
0|Move left
1|Move right

#### Done 
Returns a boolean stating whether the game has finished.
#### Results required
To pass this challenge, our model must get an average score above 195 for 100 consecutive trials. 200 is the max score you can get in a game. After that, the game automatically closes. 

### Background 
Darwin was the pioneer of evolution. He created most of the theories that state how organisms evolove. It is neccessary to understand a little about how evolution takes place, before applying it to neuroevolution. 

Evolution occurs through multiple factors. The first thing to know is that evolution takes place on populations, not individual organisms. A population is a group of animals that live together in a certain area. So, evolution requires a population. When a populatoin is large, there is competition for resources such as food and living space. As a result, only the organisms within a population, that can easily get access to resources can successfully breed. This is known as the survival of the fittest. Fitness is the ability of an organism to survive within it's habitat and reproduce. Animals with traits that are suited to their environment will have a higher fitness. Therefore, majority of the population will start to consist of the animals with similar traits which help obtain higher fitness levels. Animals with bad traits will die and won't produce any offspring. When there is an extreme change in a populations traits, the population is considered to have evolved. 

In order for evolution to take place, you need biological diversity. You need many different traits in a population to maximize the chance that at least some organisms will breed the next generation. Some of this diversity comes from mutations. Mutations can cause a new trait to form that can help in survival or not. 

For more information about this, visit these 2 sources - \
https://evolution.berkeley.edu/evolibrary/article/evo_toc_01 Goes in depth about evolution \
https://www.yourgenome.org/facts/what-is-evolution Very introductory about evolution 

I won't be able to cover every aspect of evolution and neither will the sources above. But, hopefully you now have a general idea of how evolution works.

### Steps that we will take for neuroevolution
For neuroevolution, we once we have established a model, we can take the following steps - 
* Create a population for models - multiple models with their own characteristics. In terms of DNN's, this would mean hyperparameters. This example's population will have randomly initialized weights.
* Find the fittest - run tests on the model. The models total reward during the test is known as fitness. Fitness is a term coined by Darwin. It is a measure of the ability of an organism to reproduce and create new generations.
* Select the fittest - keep the models that have the highest fitness (the highest rewards).
* Make new generations - take the fittest models. Use their characteristics to create new models or in scientific terms, a new generation.
* Mutations - randomly choose a model. Then randomly change something about it. In DNN's, this could be it's hyperparameters.
* Repeat - repeat the steps above until you converge to a very accurate model. 
If some of this didn't make sense, don't worry. There will be detailed explainations of every step below, along with coded examples. Note, we are not doing any learning of sorts. We are simply getting random weights and seeing whether they work well. Then, we improve on those weights till we converge to the ideal weights. 

#### Creating a population
I will keep a population size of 10. This means that I will have 10 seperate models that I will use. 
**Note - I will sometimes, refer to models as Agents. Agents are models that are part of a population. A population is a collection of many models. Please pardon me if I ever interchange Agents with models.**

    class Population:
        def __init__(self):
            self.weights = np.random.uniform(-1.0, 1.0, 4)
            self.surviveSteps = 0
            self.fitness = 0
            self.lifeFitness = 0
            self.lifeSteps = 0

The population class creates an Agent, which is one of the models in the population. Each model will have a corresponding fitness value. Also, for transparency purposes, I we will collect extra data such as the number of steps the model survived in a game. Once a new generation has been created, the fitness and the number of survived steps will be reset. The variables lifeFitness and lifeSteps data of the Agent across all the generations it survives.
#### Find the fittest
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
This method gets the top 2 fittest Agents in a population. They will be used later in order to breed the next generation. We will also select an extra Agent randomly from the group of Agents that didn't do so well. This is to replicate real life scenarios. In the real world, the organisms that don't do well can still survive, but not breed. Since we are going to apply mutations later on, it could also be worth keeping an extra organism. If it happens to undergo mutations, it could potentially start to out perform the other Agents in the population. 
#### Breeding the new generation
Now that we have a list of the fittest Agents from using the method above, we can breed a new generation. We will use the fittest Agents to create a new generation by using the method below. 
    
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

When breeding, we will only breed the number of Agents that are required to fill in a certain population size. We generally try to keep the population size the same across all generation. Higher population sizes usually help, as there is a higher chance of having a lot of diversity between Agents.

Our objective is to initialize an Agent with random weights. Then replace some of it's weights from it's parents' weights. We choose random parents from the fittest population. Then we choose a random number of weights to replace from the new Agent. You might realize that this isn't very realistic. In the real world, all traits are inherited from one of the parents. In this case, we are only replacing a few of the weights. I did this to create more variation within a population. You can tinker with this if you like. 

#### Mutations
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

This piece of code performs mutations on a population. This happens after the new generation has been bred. This piece of code particularly chooses two random Agents. Then it performs it randomly chooses a weight value and assigns it to a random weight. Mutations help in creating diversity. Also, by including mutations, we can help neuroevolution converge to different weights. Sometimes, these values can outperform weights learnt from standard nerual nets. 

#### Repeat
We keep doing the steps above and keep making new generations until the population can perform very well. In this case, I stopped the algorithm once any one Agent in the algorithm got a fitness score of 200. You can change this. Maybe you can keep running the generations until all the Agents in the population can get a fitness score of 200. 

## Results
Results from making random moves|Results from using neuroevolution
--------------------------------|---------------------------------
![GitHub Logo](/NeuroEvolution/Videos/testCase1.gif)|![GitHub Logo](/NeuroEvolution/Videos/testCase2.gif)


You can see that the one with the random moves died almost immediately. It got a fitness score of about 16. The second one that used neuroevolution got a score of 200. You can tell that the model clearly knows what it's doing. Here is another example that performed extraordinarily well. 

![GitHub Logo](/NeuroEvolution/Videos/testCase3.gif)
## Things to take away
Neuroevolution is a great method for training accurate models in short amounts of time. It can be used in the fields of Deep Learning. 
## Things to work on 
This code only shows the basics of neuroevolution. Also, most of the time, you have to get lucky in order to get a model that generalizes well over any scenario for CartPole. A lot of the times, an Agent gets a good score during the evolutionary process. But, when you test it's results, it doesn't do so well anymore. There are solutions to this. You can increase mutations. Do more randomizations to some parts of breeding and mutations. Run the generations for longer even if the score requirement hsa been reached. I'll leave rest of the tinkering process to you.

**NOTE -** I have more or less explained what neuroevolution is by using the code. The orignal code however has a lot of pre-processing and helper methods. To fully understand this, I recommend following the code in this repository. I have heavily commented the code for make things a little transparent. 
## Built with
* Gym
* Python3
