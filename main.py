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
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout
from keras.optimizers import SGD
#from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder







data_base_path = "Data"
# make raw data path
raw_data_path = os.path.join(data_base_path, "RAW")
joined_data_path = os.path.join(data_base_path, "Joined")

# Control Variables
run_join_files = False
get_mlp = True
get_hybrid = False
get_assigned = False


def main():
    # join files if needed
    if run_join_files:
        make_joined_files()
    if get_mlp:
        model = create_mlp()
    elif get_hybrid:
        #define CNN or a combination of CNN with RNN
        pass
    elif get_assigned:
        pass
        
    # group files in groups of 4 patients
    groups=group_patients()
    
    train_test_list= get_train_test(groups)
    
    
    results_df = run_model(train_test_list, model, folds=10)
    
    print (results_df)

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

def group_patients(make_combinations = False, make_random = False):
    # Get all the files in the joined folder
    joined_files = glob.glob(os.path.join(joined_data_path, "*.csv"))
    # Sort the files by number
    joined_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
   
    # Initialize groups as an empty list
    groups = []

    for i in range(0, len(joined_files), 5):  # Iterate through the files in groups of 5
        # Get 4 files
        group = joined_files[i:i+5]
        # Add to groups if there are 4 files
        if len(group) == 5:
            if make_random:
                # Select a random file to be the test file
                test_file = random.choice(group)
                # Remove the selected test file from the group
                group.remove(test_file)
                # Add the train and test files as a tuple to the groups list
                groups.append((group, test_file))
            else:
                # select the last file to be the test file
                test_file = group[-1]
                # Remove the selected test file from the group
                group.remove(test_file)
                # Add the train and test files as a tuple to the groups list
                groups.append((group, test_file))
               
    return groups

def get_train_test(groups):
    # Example of train_test_list:
    # [["patient_0", "patient_1", "patient_2", "patient_4", "patient_5", "patient_6"], ["patient_3", patient_7"]]
    train_test_list =[[],[]]
    for group in groups:
        # Type of vraiable expcted for group[0], list of strings that represent file paths
        # type of variable expected for group[1], string that represents file path
        for file in group[0]:
            train_test_list[0].append(file)
        train_test_list[1].append(group[1])

    return train_test_list

# def run_preprocessing(normalize, standardize, grid_search):
# # Part 1 (Preprocessing): Using data from 3 patients for "Train" and 1 patient for "Test," perform the following steps for each neural network:
#     groups=group_patients()
#     train_test_list= get_train_test(groups)

#     # Data scaling: Normalize and standardize the data. Compare the results with "Raw data" and determine if it is beneficial to use either of them.
#     if normalize:
#         # Normalize data
#         pass
#     elif standardize:
#         # Standardize data
#         pass
#     elif grid_search:
#         # Get the best parameters using GridSearch
#         pass

#     # Obtain the best parameters using GridSearch.
#     # For data balancing, use the balancing option available in the "fit" function. Example:
#         # model.fit(X_train, Y_train, batch_size=batch_size, class_weight=weights)
#     # where "weights" is calculated as follows:
#         # def generate_class_weights(y_train):
#         #     from sklearn.utils.class_weight import compute_class_weight
#         #     class_labels = np.unique(y_train)  # classes contained in y_train
#         #     class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
#         #     return dict(zip(class_labels, class_weights))

#     return normalize

def run_model(train_test_list, model, folds=3):
    label_encoder = LabelEncoder()

    train_data = [] # List of dataframes    
    # Load data first
    for train_patient_path in train_test_list[0]: 

        # Load patient data
        patient_train_data = pd.read_csv(train_patient_path)
        train_data.append(patient_train_data)

    # Concatenate the train data
    train_data = pd.concat(train_data, axis=0)

    # Convert the train data to a numpy array
    train_data = train_data.to_numpy()

    kfold = KFold(n_splits=folds, shuffle=True)
    # make dataframe with columns for results per fold
    results_df = pd.DataFrame(columns=["Fold", "Accuracy","Kappa", "F1-Weighted_Score", "Confusion_Matrix", "Train_Time"])
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data)):

        results_dict = {}
        
        print(f"Fold {fold + 1}")
        # Get the training and validation data
        model_train_data = train_data[train_idx]
        model_val_data = train_data[val_idx]

        # Make X and y for training and validation
        X_train = model_train_data[:, :-1]
        y_train = model_train_data[:, -1]
        X_val = model_val_data[:, :-1]
        y_val = model_val_data[:, -1]

       
        #convert labels to integers
        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.fit_transform(y_val)

        #convert to one hot encoding
        y_train = keras.utils.to_categorical(y_train, num_classes=5)
        


        #run model
        start_time = time.time()
           # For data balancing, use the balancing option available in the "fit" function. Example:
        # model.fit(X_train, Y_train, batch_size=batch_size, class_weight=weights)
        model.fit(X_train, y_train, batch_size=10, epochs=5)
        end_time = time.time()
        train_time = end_time - start_time

        # Predict the labels for the validation data
        y_pred = model.predict(X_val)

        # Get max value for each row
        y_pred = np.argmax(y_pred, axis=1)


        # Get scores
        accuracy = accuracy_score(y_val, y_pred)
        results_dict["Accuracy"] = accuracy

        # Get kappa
        kappa = cohen_kappa_score(y_val, y_pred)
        results_dict["Kappa"] = kappa
        
        # Get F1 score
        f1 = f1_score(y_val, y_pred, average="weighted")
        results_dict["F1-Weighted_Score"] = f1

        # Get confusion matrix
        cm = confusion_matrix(y_val, y_pred)
        results_dict["Confusion_Matrix"] = cm

        # Add train time
        results_dict["Train_Time"] = train_time

        # Add fold number
        results_dict["Fold"] = fold + 1

        # Append results to results_df
        # Turn dictionary into dataframe
        results_dict_df = pd.DataFrame.from_dict(results_dict, orient="index").T

        # Use pd.concat to add the results_df to a results_df
        results_df = pd.concat([results_df, results_dict_df], axis=0)


    print("Test")
    test_data = [] # List of dataframes    
    # Load data first
    for test_patient_path in train_test_list[1]: 

        # Load patient data
        patient_test_data = pd.read_csv(test_patient_path)
        test_data.append(patient_test_data)

    # Concatenate the train data
    test_data = pd.concat(test_data, axis=0)

    # Convert the train data to a numpy array
    test_data = test_data.to_numpy()

    # Split the test data into X and y
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1]

    #convert labels to integers
    y_test = label_encoder.fit_transform(y_test)

    # Predict the labels for the test data
    y_pred = model.predict(X_test)

    # Get max value for each row
    y_pred = np.argmax(y_pred, axis=1)

    results_dict = {}
    # Get scores
    accuracy = accuracy_score(y_test, y_pred)
    results_dict["Accuracy"] = accuracy

    # Get kappa
    kappa = cohen_kappa_score(y_test, y_pred)
    results_dict["Kappa"] = kappa
    
    # Get F1 score
    f1 = f1_score(y_test, y_pred, average="weighted")
    results_dict["F1-Weighted_Score"] = f1

    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    results_dict["Confusion_Matrix"] = cm

    # Fold is -1 for test
    results_dict["Fold"] = -1

    # train time is -1 for test
    results_dict["Train_Time"] = -1

    # Append results to results_df
    # Turn dictionary into dataframe
    results_dict_df = pd.DataFrame.from_dict(results_dict, orient="index").T

    # Use pd.concat to add the results_df to a results_df
    results_df = pd.concat([results_df, results_dict_df], axis=0)

    return results_df


def create_mlp():
    model = Sequential()
   
    # Add the input layer
    model.add(Dense(units=128, activation='relu', input_dim=3000))
    
    # Add one or more hidden layers
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    
    # Add the output layer
    model.add(Dense(units=5, activation='softmax'))

    #define optimizer
    sgd = SGD(learning_rate=0.01, momentum=0.2)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics= None)
    
    return model

main()

