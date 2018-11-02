#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			8/31/2018
#@description	Naive Bayes Classified implementation

import numpy as np
import argparse

#@TODO: redo this class (and others) with numpy or pandas?

#=============================
# NaiveBayes
#
# - Class to encapsulate a Naive Bayes model
#=============================
class NaiveBayes:

	def __init__(self, number_of_classes, pseudo_probability=0.001, pseudo_samples=1):
		self.number_of_classes = number_of_classes
		self.pseudo_probability = pseudo_probability # will use 1 pseudo example added w/ probability 
		self.pseudo_samples = pseudo_samples
		self.feature_percents = list()
		self.class_percents = list()

	#=============================
	# train()
	#
	#	- train a naive bayes model (probabilities) from given train_vectors
	#	- Warning, this will not be a scalable, efficient solution!
	#
	#@param train_vectors	2D matrix of format [X0,X1...Xn, Class]
	#@param number_of_classes	number of classes to expect
	#@return				percentage table
	#=============================
	def train(self, train_vectors):
		sample_size = self._get_number_of_input_vectors(train_vectors)
		class_totals = list()
		all_feature_totals = list()

		for classification in range(0, self.number_of_classes, 1):
			class_totals.append(0)

		number_of_features = self._get_number_of_inputs(train_vectors[0])
		for feature in range(0, number_of_features, 1):
			tmplist = []
			for classification in range(0, self.number_of_classes, 1):
				#TODO: get the range of values & make a list that large
				tmplist.append([0,0])
			all_feature_totals.append(tmplist)
		#print('all_feature_totals')
		#print(all_feature_totals)

		#Count all relevant data
		for train_vector in train_vectors:
			classification = self._get_expected_result(train_vector)
			class_totals[classification] += 1

			for input_itr in range(0, number_of_features, 1):
				input_feature_val = int(float(train_vector[input_itr]))
				#print('input_itr')
				#print(input_itr)
				#print('classification')
				#print(classification)
				#print('input_feature_val')
				#print(input_feature_val)
				all_feature_totals[input_itr][int(classification)][input_feature_val] += 1

		#print('all_feature_totals')
		#print(all_feature_totals)
		#print('class_totals')
		#print(class_totals)
		
		#Create class percentage table based on counts: for ease of use
		for class_itr in range(0, self.number_of_classes, 1):
			tmp_class_percent = float(class_totals[class_itr]) / float(sample_size)
			self.class_percents.append(tmp_class_percent)
		#print(self.class_percents)

		#Create input percentage table based on counts: for ease of use
		for input_itr in range(0, number_of_features, 1):
			self.feature_percents.append(list())
			for classification in range(0, self.number_of_classes, 1):
				number_iterations_class = class_totals[classification]

				number_iterations_feature_0_for_class = all_feature_totals[input_itr][classification][0]
				probability_of_input_0_for_class = \
					float(number_iterations_feature_0_for_class + \
							self.pseudo_samples * self.pseudo_probability) / \
					float(number_iterations_class + self.pseudo_samples)

				number_iterations_feature_1_for_class = all_feature_totals[input_itr][classification][1]
				probability_of_input_1_for_class = \
					float(number_iterations_feature_1_for_class + \
							self.pseudo_samples * self.pseudo_probability) / \
					float(number_iterations_class + self.pseudo_samples )

				self.feature_percents[input_itr].append(
						(probability_of_input_0_for_class, probability_of_input_1_for_class) )

				#print('input_itr ', input_itr, 'class ', classification)
				#print('number_iterations_feature_0_for_class', number_iterations_feature_0_for_class)
				#print('number_iterations_feature_1_for_class', number_iterations_feature_1_for_class)
				#print('number_iterations_class', number_iterations_class)
				#print('probability of input 0 for class', probability_of_input_0_for_class)
				#print('probability of input 1 for class', probability_of_input_1_for_class)

		#print('probability table:')
		#print(self.feature_percents)

		return self.feature_percents
	
	#=============================
	# test() #
	#
	#	- test the internal Naive Bayes model (percents) for given test_vectors
	#	- Assumes the model has already been trained! Otherwise pointless
	#
	#@param test_vectors	2D matrix of format [X0,X1...Xn, Class]
	#@return				ouput_vector
	#=============================
	def test(self, test_vectors):
		class_attempts = 0
		class_fails = 0
		class_success = 0

		number_of_inputs = self._get_number_of_inputs(test_vectors[0])

		final_product = 1
		for test_vector in test_vectors:
			classification_stats = [-1, -1.0] #[class, calculated probability]
			for classification in range(0, self.number_of_classes, 1):
				current_classification_probability = self.class_percents[classification]
				for input_itr in range(0, number_of_inputs, 1):
					input_val = int(float(test_vector[input_itr]))
					
					#TODO: modify this for multiple value sizes?
					current_classification_probability *= \
						self.feature_percents[input_itr][classification][input_val]

				if current_classification_probability > classification_stats[1]:
					classification_stats = [classification, current_classification_probability]

			class_attempts += 1
			if classification_stats[0] == self._get_expected_result(test_vector):
				class_success += 1
			else:
				class_fails += 1

		#4 - choose the largest value

		return (class_attempts, class_fails, class_success)

	#=============================
	# _GET_NUMBER_OF_INPUTS()
	#	-The expected input format is [X1, X2, ... Xn, Fn]
	#=============================
	def _get_number_of_inputs(self, data_vector):
		return len(data_vector) - 1

	#=============================
	# _GET_NUMBER_OF_INPUT_VECTORS()
	#	-The expected input format is [X1, X2, ... Xn, Fn]
	#=============================
	def _get_number_of_input_vectors(self, data_vectors):
		return len(data_vectors)

	#=============================
	# _GET_DATA_INPUTS()
	#	-The expected input format is [X1, X2, ... Xn, Fn]
	#=============================
	def _get_data_inputs(self, data_vector):
		inputs = data_vector[0:self._get_number_of_inputs(data_vector) ] #grabs start:end-1
		return inputs

	#=============================
	# _GET_EXPECTED_RESULT()
	#	-The expected input format is [X1, X2, ... Xn, Fn]
	#=============================
	def _get_expected_result(self, data_vector):
		expected_result = int(float(data_vector[-1])) #prevent possible string returned
		return expected_result


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main() - testing the training & predictive ability of Naive Bayes')

	print()
	print('TEST 1: train the model')
	print('input data:')
	test_data = [[0, 0, 0], [0, 1, 0], [ 1, 0, 1], [1, 1, 1]] #Should be 100% correlated to X1
	print(test_data)

	number_of_classes = 2
	naive_bayes = NaiveBayes(number_of_classes)
	naive_bayes_trained_percents = naive_bayes.train(test_data)
	print('trained percentages')
	print(naive_bayes_trained_percents)

	print()
	print('TEST 2: test the model')
	print('input data:')
	print(test_data)
	naive_bayes_test_results = naive_bayes.test(test_data) #Should get this right since it's the training data!
	print('classification attempts(', naive_bayes_test_results[0], '), \
#fails(', naive_bayes_test_results[1], '), \
#success(' , naive_bayes_test_results[2], ')')

	#compare results for multiple classes 
	print()
	print('TEST 3: test 3+ classes')
	train_data1 = [[1,0,0,0], [0,1,0,1], [0,0,1,2]]
	print('train data1:')
	print(train_data1)
	number_of_classes = 3
	naive_bayes_multi_class = NaiveBayes(number_of_classes)

	naive_bayes_trained_percents = naive_bayes_multi_class.train(train_data1)
	print('trained percentages as input[ class[ (prob 0, prob 1) ] ]')
	print(naive_bayes_trained_percents)

	test_data = train_data1
	print('test_data:')
	print(test_data)

	print('Test results:')
	naive_bayes_test_results = naive_bayes_multi_class.test(test_data)
	print('#classification attempts(', naive_bayes_test_results[0], '), \
#fails(', naive_bayes_test_results[1], '), \
#success(' , naive_bayes_test_results[2], ')')

	#TODO: try training a model with > 2 value attributes
	'''
	print()
	print('TEST 4: test > 2 value inputs')
	train_data4 = [[1,0,0], [2,0,0], [0,1,1], [0,2,1]]
	print('train data4:')
	print(train_data4)
	number_of_classes = 2
	naive_bayes_multi_class = NaiveBayes(number_of_classes)

	naive_bayes_trained_percents = naive_bayes_multi_class.train(train_data4)
	print('trained percentages as input[ class[ (prob 0, prob 1) ] ]')
	print(naive_bayes_trained_percents)

	test_data = train_data4
	print('test_data:')
	print(test_data)

	print('Test results:')
	naive_bayes_test_results = naive_bayes_multi_class.test(test_data)
	print('#classification attempts(', naive_bayes_test_results[0], '), \
#fails(', naive_bayes_test_results[1], '), \
#success(' , naive_bayes_test_results[2], ')')
'''


if __name__ == '__main__':
	main()
