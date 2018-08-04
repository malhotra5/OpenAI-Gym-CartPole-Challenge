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

### Steps that we will take for neuroevolution
For neuroevolution, we once we have established a model, we can take the following steps - 
* Create a population of model - multiple models with their own characteristics. In terms of DNN's, this would mean hyperparameters. This example's population will have randomly initialized weights.
* Find the fittest - run tests on the model. The models total reward during the test is known as fitness. Fitness is a term coined by Darwin. It is a measure of the ability of an organism to reproduce and create new generations.
* Select the fittest - keep the models that have the highest fitness (the highest rewards).
* Make new generations - take the fittest models. Use their characteristics to create new models or in scientific terms, a new generation.
* Mutations - randomly choose a model. Then randomly change something about it. In DNN's, this could be it's hyperparameters.
* Repeat - repeat the steps above until you converge to a very accurate model. 
If some of this didn't make sense, don't worry. There will be detailed explainations of every step below, along with coded examples. Note, we are not doing any learning of sorts. We are simply getting random weights and seeing whether they work well. Then, we improve on those weights till we converge to the ideal weights. 

#### Creating a population
I will keep a population size of 10. This means that I will have 10 seperate models that I will use. 

    class Population:
        def __init__(self):
            self.weights = np.random.uniform(-1.0, 1.0, 4)
            self.surviveSteps = 0
            self.fitness = 0
            self.lifeFitness = 0
            self.lifeSteps = 0

The population class creates an Agents, which is one of the models in the population. Each model will have a corresponding fitness value. Also, for transparency purposes, I we will collect extra data such as the number of steps the model survived in a game.


