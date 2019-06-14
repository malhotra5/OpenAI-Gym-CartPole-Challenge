import gym
import tensorflow
import time
import keras
import numpy as np


env = gym.make("CartPole-v1")


def testResults(modelPath):
	observation = env.reset()
	trainedModel = model()
	trainedModel.load_weights(modelPath)

	scoreTracker = 0

	for i in range(200):
	  env.render()
	  action = trainedModel.predict(np.array([observation]))
	  action = np.argmax(action)

	  observation, reward, done, info = env.step(action)

	  scoreTracker = scoreTracker + reward
	  if done:
	  	observation = env.reset()
	  	print("Simulation performed with a score of {}".format(scoreTracker))
	  	scoreTracker = 0



testResults("weights.hdf5")