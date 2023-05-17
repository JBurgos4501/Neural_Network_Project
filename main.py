#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:07:24 2023

@author: mugiwara
"""

# Objective:
    # Apply concepts defined in class on data preprocessing, data handling, and classification.
    # Implement and configure artificial neural networks (ANN) in Python.
    # Analyze and synthesize the results.

# The following neural networks will be used:
    # MLP ("Multi-Layer Perceptron")
    # Hybrid: Can be CNN or a combination of CNN with RNN ("Recursive Neural Network")
    # Assigned: Neural network given by the professor.

# Methodology:

# Part 1 (Preprocessing): Using data from 3 patients for "Train" and 1 patient for "Test," perform the following steps for each neural network:
    # Data scaling: Normalize and standardize the data. Compare the results with "Raw data" and determine if it is beneficial to use either of them.
    # Obtain the best parameters using GridSearch.
    # For data balancing, use the balancing option available in the "fit" function. Example:
        # model.fit(X_train, Y_train, batch_size=batch_size, class_weight=weights)
    # where "weights" is calculated as follows:
        # def generate_class_weights(y_train):
        #     from sklearn.utils.class_weight import compute_class_weight
        #     class_labels = np.unique(y_train)  # classes contained in y_train
        #     class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
        #     return dict(zip(class_labels, class_weights))
# Part 2 (Processing): Using all the data and performing cross-validation with FOLD=3, obtain the following:
    # Confusion matrix for each FOLD (test data).
    # "F1-score" for each class in each FOLD (test data).
    # Average metrics (across the 5 FOLDS): "accuracy" and "F1-score".
    # Average time taken for training ("train").
    # Training analysis: Conduct an analysis of each neural network to observe its behavior during training and display the corresponding graphs.


import glob
import os
import pandas as pd
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix





data_base_path = "Data"
# make raw data path
raw_data_path = os.path.join(data_base_path, "RAW")
joined_data_path = os.path.join(data_base_path, "Joined")

# Control Variables
run_join_files = False
get_mlp = False
get_hybrid = False
get_assigned = False


def main():
    # join files if needed
    if run_join_files:
        make_joined_files()

    if get_mlp:
        model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=100, verbose=0)
    elif get_hybrid:
        #define CNN or a combination of CNN with RNN
        pass
    elif get_assigned:
        pass

    
    # group files in groups of 4 patients
    groups=group_patients()
    
    run_model(groups)
    # STEPS:
    # Iterate through groups using for loop
    #for group in groups:
        # Each value in for loop is a list of 4 files, one should be the test patient

        # This has already been done in the group_patients function, if there is any need to change, change there
        # Make function to determine which are test and which are train (this can be changed to a custom way of splitting later)
        # Get train as a list of 3 files
        # Get test as a list of 1 file
        
        # Make function that runs the model, the model being passed as a parameter'
        # Define model object
        # Read in all data files into dataframe
        # Check using breakpoint()


def make_joined_files(): 

    print("Making joined files")

    # use glob to get SC_{num}.csv files, until there is no result
    i = 0

    #get all the SC files in the raw folder
    sc_pattern = os.path.join(raw_data_path, f"SC_{i}.csv")
    sc_results = glob.glob(sc_pattern)
    
    # use glob to get Y_{num}.csv files, until there is no result
    i = 0
    #get all the Y files in the raw folder
    y_pattern = os.path.join(raw_data_path, f"y_{i}.csv")
    y_results = glob.glob(y_pattern)
    # While either result set doesn't return an empty array, keep looping
    while sc_results != [] or y_results != []:
        print(sc_results, y_results)
        if len(sc_results) == 0 or len(y_results) == 0:
            print("Missing either SC or Y file")
            break
        # Read in data to df
        df_data = pd.read_csv(sc_results[0], header=None)
        # Read in label to df
        df_label = pd.read_csv(y_results[0], header=None)
        # Change column name to label 
        df_label.columns = ['label']
        # Concatenate both
        df_joined = pd.concat([df_data, df_label], axis=1)
        # Save to joined folder as joined_{i}
        df_joined.to_csv(os.path.join(joined_data_path, f"joined_{i}.csv"), index=False)
        # Update results
        i+=1
        sc_pattern = os.path.join(raw_data_path, f"SC_{i}.csv")
        sc_results = glob.glob(sc_pattern)
        y_pattern = os.path.join(raw_data_path, f"y_{i}.csv")
        y_results = glob.glob(y_pattern)

    print("Finished making joined files")

def group_patients():   #Group files in groups of 4 in a list of lists

    # get all the files in the joined folder
    joined_files = glob.glob(os.path.join(joined_data_path, "*.csv"))
    #Sort the files by number
    joined_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    #Initialize groups as an empty list
    groups = []
   
    for i in range(0, len(joined_files), 4): #Iterate through the files in groups of 4
        #Get 4 files
        group = joined_files[i:i+4]
        #Add to groups if there are 4 files
        if len(group) == 4:
                # Initialize variables for tracking the best files
            best_train_files = []
            best_test_file = None
            highest_avg_value = 0.0
            #detect which files are better for training and which for testing
            # Evaluate each file
            for file_path in group:
                # Read the file and calculate the average value in a specific column
                data = pd.read_csv(file_path)
                avg_value = data['label'].mean()
                #print the average value for each file
                print(f"Average value for {file_path}: {avg_value}")
                # Update the best files if a higher average value is found
                if avg_value > highest_avg_value:
                    highest_avg_value = avg_value
                    best_train_files = group[:-1]
                    best_test_file = group[-1]
            #Add the train and test files as a tuple to the groups list
            groups.append((best_train_files, best_test_file))
    breakpoint()

    return groups


def run_model(groups):
    # Iterate through the groups
    for group in groups: # group is a tuple of train and test files
        # Get the train and test files
        train_file_paths = group[0]# train_file_paths is a list of 3 file paths
        test_file_path = group[1] # test_file_path is a string of 1 file path

        train_data = []
        for file_path in train_file_paths:
            train_data.append(pd.read_csv(file_path))# train_data is a list of 3 dataframes
        # Get the train labels as a list of arrays
        train_label = []
        for file in train_data:
            train_label.append(file['label'])

        test_data = pd.read_csv(test_file_path)  
        test_label = test_data['label']

        # Get the train and test data as numpy arrays
        for i in range(len(train_data)):
            train_data[i] = train_data[i].drop('label', axis=1).values
        
        test_data = test_data.drop('label', axis=1).values

        # Concatenate the train data and labels
        for i in range(len(train_data)):
            train_data[i] = np.concatenate((train_data[i], train_label[i].values.reshape(-1, 1)), axis=1)

        # define the neural network model
        model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=100, verbose=0)

        # Define the K-Fold cross-validator (k=3)
        kfold = KFold(n_splits=3, shuffle=True)

        # Initialize the confusion matrix variable
        cm_total = np.zeros((10, 10))
        # Iterate through the folds and fit the model
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):
            print(f"Fold {fold + 1}")
            # Get the training and validation data
            for i in range(len(train_data)):
                X_train = train_data[i][train_idx]
                X_val = train_data[i][val_idx]
                y_train = train_label[i][train_idx]
                y_val = train_label[i][val_idx]

            # Fit the model
            for i in range(len(train_data)):
                model.fit(X_train, y_train)
            breakpoint()
            # Predict the labels for the test data
            y_pred = model.predict(test_data)

            # Generate the confusion matrix
            cm = confusion_matrix(test_label, y_pred)
            print(cm)

            # Add the current fold's confusion matrix to the total confusion matrix
            cm_total += cm

            # Evaluate the model on the test data and print the accuracy
            accuracy = model.score(test_data, test_label)
            print(f"Test Accuracy: {accuracy}\n")

        # Print the total confusion matrix after all folds
        print(f"Total Confusion Matrix:\n{cm_total}\n")

        # Print results
        print("Results:\n")
        print("Accuracy: ", np.trace(cm_total) / np.sum(cm_total))
        print("Precision: ", np.trace(cm_total) / np.sum(cm_total, axis=0))
        print("Recall: ", np.trace(cm_total) / np.sum(cm_total, axis=1))
        print("F1: ", 2 * np.trace(cm_total) / (np.sum(cm_total, axis=0) + np.sum(cm_total, axis=1)))
        print("\n")



main()