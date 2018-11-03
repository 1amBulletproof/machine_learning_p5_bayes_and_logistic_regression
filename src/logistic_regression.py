#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			10/31/2018
#@description	Logistic Regression

import numpy as np
import pandas as pd
import argparse
from base_model3 import BaseModel

#@TODO: redo this class (and others) with numpy or pandas?

#=============================
# LogisticRegression
#
# - Class to encapsulate a Logistic Regression implementation
# - **ASSUMES CLASSES ARE 0-based incrementing numbers, i.e. Class1=0, Class2=1, ...***
#=============================
class LogisticRegression(BaseModel):

	#Data is expected to be a data frame
	def __init__(self, data, bias=1):
		BaseModel.__init__(self, data)
		self.bias = bias

	#=============================
	# train()
	#
	#	- train logistic regression model
	#	- effectively determine class weights
	#	- **ASSUMES CLASSES ARE 0-based incrementing numbers, i.e. Class1=0, Class2=1, ...***
	#
	#@param	learning_factor	determines how quickly the model learns
	#@param	new_data	DataFrame of new data to train-on
	#@return	weight_vectors
	#=============================
	def train(self, train_data, learning_rate=0.5):
		#initialize the weight vectors to empty
		self.weight_vectors = list()
		self.data = train_data

		#Get the actual classes
		num_classes = len(self.data[self.data.columns[-1]].unique())
		num_cols = self.data.shape[1] - 1 #no count last col: it is class value
		num_rows = self.data.shape[0]

		data_as_np = self.data.values

		#Get weight vector map where each class corresponds to a given vector
			# - all initial weights random -0.01 - 0.01 
		self.class_weights = np.random.randint(-1, 2, (num_classes, num_cols)) / 100

		weights_changed = True
		while(weights_changed):
			#For all classes: (for all weights) blank the weight delta vectors
			class_weight_deltas = np.zeros((num_classes, num_cols))
			#For all data points:
			for index in range(0, len(data_as_np)):
				#For all classes: 
				point = data_as_np[index]
				point_without_class = point[:-1]
				class_summations = np.dot(self.class_weights, point_without_class)
				class_summations_exp = np.apply_along_axis(
						lambda x: np.exp(x+self.bias), 0, class_summations)
				class_summations_exp_sum = class_summations_exp.sum()
				class_predictions = np.apply_along_axis(
						lambda x: x/class_summations_exp_sum, 0, class_summations_exp)
				point_class_val = point[-1]
				#Setup a vector where very class the return value is 0 unless its the matching weight vector
				#i.e. point[-1] or class == 4, [0,0,0,1,0...]
				class_vs_all_vector = np.zeros(num_classes)
				class_vs_all_vector[point_class_val] = 1

				class_prediction_error = class_vs_all_vector - class_predictions
				#print('class_prediction_error')
				#print(class_prediction_error)
				class_prediction_error = np.reshape(class_prediction_error, (len(class_prediction_error), -1))
				update_val = class_prediction_error * point_without_class
				class_weight_deltas = class_weight_deltas + update_val
				#print('class_weight_deltas')
				#print(class_weight_deltas)

			#For all classes:
			#If the entire 2d array of delta's are 0, you're done
			weight_updates =  class_weight_deltas * learning_rate
			#print('weight_updates')
			#print(weight_updates)
			self.class_weights = self.class_weights + (learning_rate * class_weight_deltas)
			#print('class_weights')
			#print(self.class_weights)
			if (class_weight_deltas.any()):
				weights_changed = False

			#TODO: exit after a number of iterations? In case no convergence?
		return self.class_weights

	#=============================
	# test() #
	#
	#	- test the internal logistic regression model
	#	- Assumes the model already trained!
	#
	#@param test_data	data to test as dataframe
	#@param print_classifications	boolean to decide whether to display classificaiton
	#@return		classification_accuracy as percentage
	#=============================
	def test(self, test_data, print_classifications=False):
		data_as_np = test_data.values
		total_tests = len(data_as_np)
		#print('total_tests')
		#print(total_tests)
		tests_correct = 0
		tests_incorrect = 0
		for index in range(0, len(data_as_np)):
			point = data_as_np[index]
			actual_class = point[-1]
			point_without_class = point[0:-1]

			predicted_class = self.get_prediction(point_without_class)
			if predicted_class != actual_class:
				tests_incorrect += 1
			else:
				tests_correct += 1

			if print_classifications:
				print('input:', point)
				print('class:', actual_class, '  predicted-class:', predicted_class)

		#print('tests_incorrect:', tests_incorrect)
		#print('tests_correct:', tests_correct)

		return float(100 * tests_correct/total_tests)

	#=============================
	# get_prediction() #
	#
	#	- calculate and find the maximum value and return the index
	#
	#@param test_data	data to test as dataframe
	#@return				ouput_vector
	#=============================
	def get_prediction(self, point):
		class_weight_exp = np.apply_along_axis(
				lambda x: np.exp(np.dot(x, point) + 1), 1, self.class_weights)
		return class_weight_exp.argmax()


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main() - testing the training & predictive ability of Logistic Regression')

	print()
	print('TEST 1: train the model')
	print('input data:')
	test_data = [[0, 0, 0], [0, 1, 0], [ 1, 0, 1], [1, 1, 1]] #Should be 100% correlated to X1
	test_data = pd.DataFrame(test_data)
	print(test_data)

	number_of_classes = 2
	logistic_regression = LogisticRegression(test_data)
	logistic_regression_weights = logistic_regression.train(test_data, 0.5)
	print('trained weights')
	print(logistic_regression_weights)
	logistic_regression_accuracy = logistic_regression.test(test_data, True)
	print('accuracy (%):', logistic_regression_accuracy)


	#compare results for multiple classes 
	print()
	print('TEST 2: test 3+ classes')
	test_data1 = [[1,0,0,0], [0,1,0,1], [0,0,1,2]]
	test_data1 = pd.DataFrame(test_data1)
	print('input data')
	print(test_data1)
	logistic_regression_mclass = LogisticRegression(test_data1)
	logistic_regression_mclass_weights = logistic_regression_mclass.train(test_data1, 0.5)
	print('trained weights')
	print(logistic_regression_mclass_weights)

	print('Test results:')
	logistic_regression_mclass_results = logistic_regression_mclass.test(test_data1, True)
	print('logistic regression mclass results: ', logistic_regression_mclass_results)


if __name__ == '__main__':
	main()
