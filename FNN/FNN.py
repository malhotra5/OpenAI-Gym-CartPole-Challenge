import gym
import tensorflow
import time
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import numpy as np

import gym
env = gym.make("CartPole-v1")

minScoreReq = 30


def makeData(initDataX, initDataY, minScoreReq):
	oneRunDataX = []
	oneRunDataY = []
	scoreTracker = 0
	observation = env.reset()


	for _ in range(1000):
	  env.render()
	  action = env.action_space.sample() # your agent here (this takes random actions)
	  oneRunDataX.append(observation)
	  oneRunDataY.append(action)

	  observation, reward, done, info = env.step(action)
	  scoreTracker = scoreTracker + reward

	  if done:
	    if(scoreTracker > minScoreReq):
	    	initDataX.append(oneRunDataX)
	    	initDataY.append(oneRunDataY)

	    oneRunDataX = []
	    oneRunDataY = []
	    observation = env.reset()
	    scoreTracker = 0	

	env.close()

	return initDataX, initDataY

def moreData():
	obs, act = makeData([], [], minScoreReq)

	for i in range(3):
		obs, act = makeData(obs, act, minScoreReq)
	
	np.save("feat.npy", np.array(obs))
	np.save("res.npy", np.array(act))

	return obs, act



def loadData(file1Name, file2Name):
	return np.load(file1Name), np.load(file2Name)

def formatData(X):
	finalData = []
	for i in X:
		for j in i:
			finalData.append(j)
	return np.array(finalData)

	return np.array(finalData)
def model():
	model = Sequential()
	model.add(Dense(12, input_shape=(4,), activation="relu"))
	model.add(Dense(8, activation="relu"))
	model.add(Dense(2, activation="sigmoid"))

	return model

def train(model, X, Y):
	model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	model.fit(X,Y, epochs=100, batch_size=10)
	return model

def main():
	#moreData()
	X, Y = loadData("feat.npy", "res.npy")
	setModel = model()
	X = formatData(X)
	Y = formatData(Y)
	Y = to_categorical(Y)


	print(X)
	print(Y)

	finishedModel = train(setModel, X, Y)

	finishedModel.save("model1.hdf5")



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
	  print(scoreTracker)
	  if done:
	  	observation = env.reset()
	  	print(scoreTracker)
	  	scoreTracker = 0


# main()
testResults("model1.hdf5")










