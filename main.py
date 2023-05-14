#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:07:24 2023

@author: mugiwara
"""
import glob
import os
import pandas as pd
import random
from sklearn.neural_network import MLPClassifier
import joblib




data_base_path = "Data"
# make raw data path
raw_data_path = os.path.join(data_base_path, "RAW")
joined_data_path = os.path.join(data_base_path, "Joined")

# Control Variables
run_join_files = False


def main():
    # join files if needed
    if run_join_files:
        make_joined_files()
    
    # group files in groups of 4 patients
    groups=group_patients()
    
    # preprocced_data= preprocess_data(groups)
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
                #Shuffle the group to randomly select a test file
                #Assign the last file as the test file
                test_file = group[-1]
                #Assign the other files as the train files
                train_files = group[:-1]
                #Add the train and test files as a tuple to the groups list
                groups.append((train_files, test_file))

    return groups


# Make function that runs the model, the model being passed as a parameter'


def run_model(groups):
    # Iterate through the groups
    for group in groups: # group is a tuple of train and test files
        # Get the train and test files
        train_file_paths = group[0]# train_file_paths is a list of 3 file paths
        test_file_path = group[1] # test_file_path is a string of 1 file path

        train_data = []
        for file_path in train_file_paths:
            train_data.append(pd.read_csv(file_path))
            
        breakpoint()


        # Get the train and test labels
        train_label = train_data['label']
        test_label = test_data['label']

        # Get the train and test data as numpy arrays
        train_data = train_data.drop('label', axis=1).values
        test_data = test_data.drop('label', axis=1).values


        # Run the neural network model
        model = MLPClassifier(hidden_layer_sizes=(64, 64), activation='relu', solver='adam', max_iter=100, verbose=0)

        # Fit the model on the training data
        model.fit(train_data, train_label)

        # Evaluate the model on the test data and print the accuracy
        accuracy = model.score(test_data, test_label)
        print(f"Test Accuracy: {accuracy}")





    

main()