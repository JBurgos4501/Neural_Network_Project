#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:07:24 2023

@author: mugiwara
"""


import glob
import os
import pandas as pd
import numpy as np
import random
import time
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input
from keras.optimizers import SGD
from scikeras.wrappers import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import math



data_base_path = "Data"
# make raw data path
raw_data_path = os.path.join(data_base_path, "RAW")
joined_data_path = os.path.join(data_base_path, "Joined")

# Control Variables
run_join_files = False
get_mlp = True
get_hybrid = True
get_assigned = False

#globals
normalization = False
standardization = False
get_balance = False





def main():
    # Make a list of all the control variables for 
    control_variables = [get_mlp, get_hybrid, get_assigned]
    csv_names = [
        "MLP",
        "CNN",
        "CNN_VGG"
    ]

    # join files if needed
    if run_join_files:
        make_joined_files()

    for i, bool_var in enumerate(control_variables):
        # If bool_var is false, skip to the next iteration
        if bool_var == False:
            continue
        
        # If this is running, then the bool_var is true
        # Depending on the index, run the appropriate model
        
        if i == 0:
            model = KerasClassifier(build_fn=MlpModel)
        elif i ==1:
            model = KerasClassifier(build_fn=CNNModel)
        elif i ==2:
            model= cnn_vgg(input_shape=(3000, 1), nb_class=5)
        #   model = KerasClassifier(build_fn=modelX)
        else:
            print("No model selected")
            return
        
    # gridSearch(model)
        # group files in groups of 4 patients
        groups=group_patients()
        
        train_test_list= get_train_test(groups)
        
        
        results_df = run_model(train_test_list, model, folds=3)
        # Save results to csv
        results_df.to_csv(os.path.join("Outputs", f"{csv_names[i]}.csv"), index=False)


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

def group_patients(make_combinations = False, make_random = True):
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

def run_preprocessing(X, y, normalize=False, balance=False, standardize=False):
    weights = None
    if normalize:
        # Normalize data
        scaler = MinMaxScaler()
        scaler.fit(X, y)
        X = scaler.transform(X)
        
    elif standardize:
        # Standardize data
        scaler = StandardScaler()
        scaler.fit(X, y)
        X = scaler.transform(X)


    elif balance:
        class_labels = np.unique(y)  # classes contained in y_train
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y)
        weights = dict(zip(class_labels, class_weights))

        
   

    return X, y, weights


def run_model(train_test_list, model, folds=4):
    global normalization
    global standardization
    global get_balance

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

        #preprocess data
        X_train, y_train, train_weights = run_preprocessing(X_train,y_train,normalize=normalization, balance=get_balance, standardize=standardization)
        X_val, y_val, weights = run_preprocessing(X_val,y_val,normalize=False, balance=False, standardize=False)

         #convert labels to integers
        y_train = LabelEncoder().fit_transform(y_train)
        y_val = LabelEncoder().fit_transform(y_val)

        #convert to categorical
        y_train = keras.utils.to_categorical(y_train, num_classes=5)
        

        #run model
        start_time = time.time()
           # For data balancing, use the balancing option available in the "fit" function. Example:
        # model.fit(X_train, Y_train, batch_size=batch_size, class_weight=weights)
        model.fit(X_train, y_train, class_weight=train_weights)
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


    print("Predicting Test")
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
    y_test = LabelEncoder().fit_transform(y_test)

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


def MlpModel():
    model = Sequential()
   
    # Add the input layer
    model.add(Dense(units=128, activation='relu', input_dim=3000))
    
    # Add one or more hidden layers
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    
    # Add the output layer
    model.add(Dense(units=5))

    #define optimizer
    sgd = SGD(learning_rate=0.01, momentum=0.2)

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics= None)

    #model = KerasClassifier( model=model, loss="categorical_crossentropy", optimizer=sgd, epochs=100, batch_size=10, verbose=0)
    
    return model



def cnn_vgg(input_shape, nb_class):
    # Author: K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.

    # Create the input layer with the specified input_shape
    ip = Input(shape=input_shape)

    # Start with the input layer as the current layer
    conv = ip

    # Calculate the number of CNN blocks based on the logarithm of the number of columns in the input shape
    nb_cnn = int(round(math.log(input_shape[0], 2)) - 3)
    print("pooling layers: %d" % nb_cnn)

    for i in range(nb_cnn):
        # Calculate the number of filters for the current CNN block
        num_filters = min(64 * 2 ** i, 512)

        # Add two convolutional layers with ReLU activation and He uniform kernel initialization
        conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)

        # Add an extra convolutional layer for iterations greater than 1
        if i > 1:
            conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)

        # Add a MaxPooling1D layer with a pool size of 2
        conv = MaxPooling1D(pool_size=2)(conv)

    # Flatten the last layer of the convolutional block
    flat = Flatten()(conv)

    # Add a fully connected layer with 256 units and ReLU activation
    fc = Dense(units=256, activation="relu", kernel_initializer='he_uniform')(flat)

    # Add a dropout layer with a rate of 0.5 to reduce overfitting
    dropout = Dropout(rate=0.5)(fc)

    # Add the output layer with softmax activation for multi-class classification
    output = Dense(units=nb_class, activation="softmax")(dropout)

    # Create the model with ip as the input and output as the output layer
    model = Model(ip, output)

    # Compile the model
    sgd = SGD(learning_rate=0.01, momentum=0.2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


def CNNModel():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(3000, 1)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.5))# Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data.
   # model.add(MaxPooling1D(pool_size=2))# Max pooling is a sample-based discretization process. The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to be made about features contained in the sub-regions binned.
    model.add(Flatten())# Flatten layer is for converting the data into a 1-dimensional array for inputting it to the next layer.
    model.add(Dense(100, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def gridSearch(model):
    global normalization
    global standardization
    global get_balance

    # Load in all patient data
    # Get al the files in the joined folder
    joined_files = glob.glob(os.path.join(joined_data_path, "*.csv"))
    # Sort the files by number
    joined_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # Read data into a list of dataframes
    data = [pd.read_csv(file) for file in joined_files]
    # Concatenate the dataframes
    data = pd.concat(data, axis=0)
    # Get X and y
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Change X to numpy array
    X = X.to_numpy()

    # Convert y to integers and numpy array
    y = LabelEncoder().fit_transform(y)

    X, y, weights = run_preprocessing(X,y,normalize=normalization, balance=get_balance, standardize=standardization)


    # Convert y to categorical
    y = keras.utils.to_categorical(y, num_classes=5)

    #modelK = KerasClassifier(build_fn=MlpModel)
        # Define the parameters for grid search
    parameters = {
    #'loss': ['categorical_crossentropy', 'binary_crossentropy'],
    #'optimizer': ['SGD', 'Adam', 'Adamax'],
    'epochs': [5],
    #'shuffle': [True, False]
        }

    

    
        # Perform grid search on the training data
    grid_search_model = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy')

    grid_search_model.fit(X, y, class_weight=weights)
    best_params = grid_search_model.best_params_
    # Set the best parameters for the model
    model.set_params(**best_params)
    print()
    print()
    print()
    print()    
    print()
    print("Finished Looking for best parameters")
    print("Best parameters:", best_params)
    print()
    print()
    print()
    print()    
    print()
    
    return model



main()

