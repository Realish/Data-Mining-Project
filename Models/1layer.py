import scipy.io
import matplotlib.pyplot as plt
import sys
import numpy as np
import random
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler

k = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]

def loadfile(path):
	#print('read' + path)
	return scipy.io.loadmat(path)

def process_features(array):
	ret_val = []
	for j in range(20, 108):
		e = array[j]
		mu = np.average(e)
		sig = np.std(e)
		for i in k:
			ret_val.append(mu + i*sig)
	return ret_val
		
    
def main():
	genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
	train_matrix = []
	train_label = []
	test_matrix = []
	test_label = []
	for i in range(10):
		path_src = 'pitch_folder/' + genres[i] + '/' + genres[i]
		for j in range(90): 
			if i == 5 and j == 54:
				continue;
			file_src = path_src + '.000' + str(j).zfill(2) + '_pitch_4410'
			data = loadfile(file_src)['f_pitch']
			temp = process_features(data)
			train_matrix.append(temp)
			train_label.append(i)

	for i in range(10):
		path_src = 'pitch_folder/' + genres[i] + '/' + genres[i]
		for j in range(90, 100):
			file_src = path_src + '.000' + str(j).zfill(2) + '_pitch_4410'
			data = loadfile(file_src)['f_pitch']
			temp = process_features(data)
			test_matrix.append(temp)
			test_label.append(i)

	

	test_matrix = np.array(test_matrix)
	test_label = np.array(test_label)
	train_matrix = np.array(train_matrix)
	train_label = np.array(train_label)
	
	print('start training')

	X_train = train_matrix
	X_test = test_matrix

	n_output = 10
	n_input = 616	

	# Convert the target labels to one-hot encoding
	y_train_one_hot = to_categorical(train_label, num_classes=10)
	y_test_one_hot = to_categorical(test_label, num_classes=10)
	
	n_hidden_single = int(math.sqrt(n_input * n_output))	

	# Define the model
	model = Sequential()

	# Add a dense (fully connected) hidden layer with, for example, 128 units and 'relu' activation function
	model.add(Dense(n_hidden_single, input_shape=(616,), activation='relu'))

	# Add the output layer with 10 units (assuming 10 classes) and 'softmax' activation function
	model.add(Dense(10, activation='softmax'))

	# Compile the model with an appropriate optimizer, loss function, and metrics
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Print the model summary to see the architecture and parameters
	model.summary()

	# Train the model with your training data
	model.fit(X_train, y_train_one_hot, epochs=100, batch_size=9, validation_split=0.2)

	# Evaluate the model on the test set
	test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
	print(f'Test accuracy: {test_acc * 100:.2f}%')

if __name__ == "__main__":
	main()
