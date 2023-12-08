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
	
	n_hidden_layer1 = int(n_output * (math.pow((n_input / n_output), (1/3)))*2)
	n_hidden_layer2 = int(n_output * (math.pow((n_input / n_output), (1/3))))

	# Define the model
	model = Sequential()

	# Add a dense (fully connected) hidden layer with, for example, 128 units and 'relu' activation function
	model.add(Dense(n_hidden_layer1, input_shape=(616,), activation='relu'))

	model.add(Dense(n_hidden_layer2, activation='relu'))	

	# Add the output layer with 10 units (assuming 10 classes) and 'softmax' activation function
	model.add(Dense(10, activation='softmax'))

	# Compile the model with an appropriate optimizer, loss function, and metrics
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

	# Print the model summary to see the architecture and parameters
	model.summary()

	# Train the model with your training data
	model.fit(X_train, y_train_one_hot, epochs=100, batch_size=32, validation_split=0.2)

	# Evaluate the model on the test set
	test_loss, test_acc = model.evaluate(X_test, y_test_one_hot)
	print(f'Test accuracy use evaluate function: {test_acc * 100:.2f}%')

	# Evalute the model based on transformed predict data
	predictions = model.predict(X_test)
	predicted_labels = np.argmax(predictions, axis=1)
	predicted_labels_one_hot = to_categorical(predicted_labels, num_classes=10)
	counter = 0
	total = 0
	for i in range(len(y_test_one_hot)):
		total = total + 1
		p = predicted_labels_one_hot[i] 
		t = y_test_one_hot[i]
		for j in range(len(p)):
			if p[j] == 1 and p[j] == t[j]:
				counter = counter + 1

	print('Test Accuracy use counter: ' + str(counter/total))	



if __name__ == "__main__":
	main()
