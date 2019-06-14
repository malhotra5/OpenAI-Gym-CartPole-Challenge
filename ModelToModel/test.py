import gym
import tensorflow
import time
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np


env = gym.make("CartPole-v1")


def newModel():
	model = Sequential()
	model.add(Dense(16, input_shape=(4,), activation="relu"))
	model.add(Dense(12, activation="relu"))
	model.add(Dense(4, activation="relu"))
	model.add(Dense(2, activation="sigmoid"))
	return model

def previousModel():
	model = Sequential()
	model.add(Dense(12, input_shape=(4,), activation="relu"))
	model.add(Dense(8, activation="relu"))
	model.add(Dense(2, activation="sigmoid"))

	return model

def testResults(modelPath, modelArch):
	observation = env.reset()
	trainedModel = modelArch
	trainedModel.load_weights(modelPath)

	scoreTracker = 0

	for i in range(1000):
	  env.render()
	  action = trainedModel.predict(np.array([observation]))
	  action = np.argmax(action)

	  observation, reward, done, info = env.step(action)

	  scoreTracker = scoreTracker + reward
	  if done:
	  	observation = env.reset()
	  	print("Simulation performed with a score of {}".format(scoreTracker))
	  	scoreTracker = 0



testResults("newModel.hdf5", newModel())