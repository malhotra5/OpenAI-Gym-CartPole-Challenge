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


def trainFromPreviousModel(prevModel, nowModel):
	nowModel.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
	
	observation = env.reset()
	for _ in range(10000):
	  action = prevModel.predict(np.array([observation]))
	  action = np.argmax(action)

	  act = [0,0]
	  act[action] = 1


	  nowModel.fit(np.array([observation]), np.array([act]), epochs=1, batch_size=1, verbose=0)

	  observation, reward, done, info = env.step(action)

	  if done:
	    observation = env.reset()

	env.close()

	nowModel.save("newModel.hdf5")

prevModel = previousModel()
prevModel.load_weights("previousMode.hdf5")
trainFromPreviousModel(prevModel, newModel())


