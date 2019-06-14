import gym
import tensorflow
import time
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np


env = gym.make("CartPole-v1")


def model():
	model = Sequential()
	model.add(Dense(12, input_shape=(4,), activation="relu"))
	model.add(Dense(8, activation="relu"))
	model.add(Dense(2, activation="sigmoid"))

	return model


def streamTrain(model):
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	observation = env.reset()
	for _ in range(30000):
	  # env.render()
	  action = env.action_space.sample() # your agent here (this takes random actions)
	 
	  act = [0,0]
	  act[action] = 1


	  model.fit(np.array([observation]), np.array([act]), epochs=1, batch_size=1, verbose=0)

	  observation, reward, done, info = env.step(action)

	  if done:
	    observation = env.reset()

	env.close()

	model.save("weights.hdf5")


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


streamTrain(model())

testResults("weights.hdf5")