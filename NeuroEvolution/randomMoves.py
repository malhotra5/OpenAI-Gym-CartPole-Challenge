import gym
from gym import wrappers
import random 

env = gym.make("CartPole-v0")
env = wrappers.Monitor(env, "C:/Users/rohit/Desktop/Genetic Algorithm", force=False)

observation = env.reset()
done = False

while not done:
    env.render()
    action = random.randrange(0,2)
    observation, reward, done, info = env.step(action)
