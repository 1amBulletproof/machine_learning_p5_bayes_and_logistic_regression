#!/usr/local/bin/python3

#@author		Brandon Tarney
#@date			10/30/2018
#@description	run experiment

from naive_bayes import NaiveBayes
#from logistic_regression import LogisticRegression

import argparse
from file_manager import FileManager
#from data_manipulator import DataManipulator
import numpy as np

#=============================
# run_model()
#
#	- read-in 5 groups of input data, train on 4/5,
#		test on 5th, cycle the 4/5 & repeat 5 times
#		Record overall result!
#=============================
def run_models_with_cross_validation(num_classes=2):

	#GET DATA
	#- expect data_0 ... data_4
	data_groups = list()
	data_groups.append(FileManager.get_csv_file_data_array('data_0'))
	data_groups.append(FileManager.get_csv_file_data_array('data_1'))
	data_groups.append(FileManager.get_csv_file_data_array('data_2'))
	data_groups.append(FileManager.get_csv_file_data_array('data_3'))
	data_groups.append(FileManager.get_csv_file_data_array('data_4'))
	

	NUM_GROUPS = len(data_groups)

	#For each data_group, train on all others and test on me
	model1_culminating_result = 0;
	model2_culminating_result = 0;
	model1_final_average_result = 0
	model2_final_average_result = 0

	for test_group_id in range(NUM_GROUPS):
		print()
		#Form training data as 4/5 data
		train_data = list()
		for train_group_id in range(len(data_groups)):
			if (train_group_id != test_group_id):
				#Initialize train_data if necessary
				if (len(train_data) == 0):
					train_data = data_groups[train_group_id]
				else:
					train_data = train_data + data_groups[train_group_id]

		print('train_data group', str(test_group_id), 'length: ', len(train_data))
		#print(train_data)

		test_data = data_groups[test_group_id]

		model1_result = 0
		model2_result = 0
		model1 = NaiveBayes(num_classes)
		#model2 = LogisticRegression(train_data)
		model1.train(train_data)
		#model2.train()
		model1_result = model1.test(test_data) # returns (attempts, fails, success)
		#print('result:', result)
		model1_accuracy = (model1_result[2]/model1_result[0]) * 100
		print('Naive Bayes Accuracy (%):', model1_accuracy)
		#model2_result = model2.test(test_data)
		#print('result:', result)
		#print('Logistic Regression Accuracy (%):', result2)
		model1_culminating_result = model1_culminating_result + model1_accuracy
		#model2_culminating_result = model2_culminating_result + model2_result


	model1_final_average_result = model1_culminating_result / NUM_GROUPS
	#model2_final_average_result = model2_culminating_result / NUM_GROUPS
	#print()
	#print('final average result:')
	#print(final_average_result)
	#print()

	return (model1_final_average_result, model2_final_average_result)


#=============================
# MAIN PROGRAM
#=============================
def main():
	#print('LOG: Main program to pre-process House-Votes-84.data file')
	parser = argparse.ArgumentParser(description='Run the classification tree test')
	parser.add_argument('num_classes', type=int, help='number of classes to expect')
	args = parser.parse_args()
	print(args)
	num_classes = args.num_classes

	final_result = run_models_with_cross_validation(num_classes)
	print('Naive Bayes Average Accuracy (%):') 
	print(final_result[0], '%')
	print('Logistic Regression Average pruned Accuracy (%):')
	print(final_result[1], '%')


if __name__ == '__main__':
	main()
