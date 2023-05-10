#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:07:24 2023

@author: mugiwara
"""
import glob
import os
import pandas as pd

data_base_path = "Data"
# make raw data path
raw_data_path = os.path.join(data_base_path, "Raw")
joined_data_path = os.path.join(data_base_path, "Joined")

# Control Variables
run_join_files = False


def main():
    # join files if needed
    if run_join_files:
        make_joined_files()
    
    # group files in groups of 4 patients
    groups = group_patients()

    # Iterate through groups using for loop
    # Each value in for loop is a list of 4 files, one should be the test patient

def make_joined_files():

    print("Making joined files")

    # use glob to get *{num}.csv files, until there is no result
    i = 0
    path_pattern = os.path.join(raw_data_path, f"*{i}.csv")
    results = glob.glob(path_pattern)
    # While results dont return an empty array, keep looping
    while results != []:
        print(results)
        
        # Read in data to df
        df_data = pd.read_csv(results[0], header=None)
        # Read in label to df
        df_label = pd.read_csv(results[1], header=None)
        # Change column name to label 
        df_label.columns = ['label']
        # Concatenate both
        df_joined = pd.concat([df_data, df_label], axis=1)



        #or 
        # df_joined = pd.concat([pd.read_csv(results[0], header=None), pd.read_csv(results[1], header=None)], axis=1)
        # Save to joined folder as joined_{i}
        df_joined.to_csv(os.path.join(joined_data_path, f"joined_{i}.csv"), index=False)


        # Update results
        i+=1
        path_pattern = os.path.join(raw_data_path, f"*{i}.csv")
        results = glob.glob(path_pattern)
    
    print("Finished making joined files")

def group_patients():
    # get all the files in the joined folder
    joined_files = glob.glob(os.path.join(joined_data_path, "*.csv"))
    # order files by patient_id, found in the file name by splitting by _
    # Explanation:
    #  sort can sort elements in a list using values returned by a custom function
    #  the custom function is defined by lambda x, where x is the element in the list
    #  Since each element is a filepath, we can split the filepath by _ and get the last element, 
    #   then split by . and get the first element to get the patient_id
    #   Example: joined_0.csv, split by _ and get las element to get 0.csv, split by . and get first element to get 0
    # Instead of sorting out the normal list, it will sort the elements 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ... as ints
    joined_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # group files in groups of 4 in a list of lists
    # Example: [[file1, file2, file3, file4], [file5, file6, file7, file8]]

main()