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
    if run_join_files:
        make_joined_files()
    


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
    
    
    
    
    # patient_files_list = []
    # max_num = 0
    
    # # Make data structure where the files will be grouped by ID
    # # Structure stored in patient_files_list
    # for file in files_list:
    #     # get the max number of patient ID
    #     # get file name
    #     filename = os.path.basename(file)

    #     # take away extension
    #     filename_no_ext_list = filename.split('.')
    #     filename_no_ext = filename_no_ext_list[0]

    #     # Get patient ID
    #     id_list = filename_no_ext.split('_')
    #     patient_id_str = id_list[1]
    #     patient_id = int(patient_id_str)

    #     if patient_id > max_num:
    #         max_num = patient_id

    # patients_num = max_num + 1
    # # create a list of lists using the max number
    # for i in range(patients_num):
    #     patient_files_list.append([])

    # # patient_files_list should have [[], [], [], ... , []]

    # # here we should have a list from 0 to max_patient_id

    # # get all the files per patient
    # for file in files_list:
    #     # get file name
    #     filename = os.path.basename(file)

    #     # take away extension
    #     filename_no_ext_list = filename.split('.')
    #     filename_no_ext = filename_no_ext_list[0]

    #     # Get patient ID
    #     id_list = filename_no_ext.split('_')
    #     patient_id_str = id_list[1]
    #     patient_id = int(patient_id_str)

    #     # add the file to the list
    #     patient_files_list[patient_id].append(file)
    # i = 0
    # for grouped_files in patient_files_list:
        
    #     sc_files = []
    #     y_files = []
    #     for file in grouped_files:
    #         if 'SC' in file:
    #             sc_files.append(file)
    #         elif 'y' in file:
    #             y_files.append(file)
        
    #     # sort the files
    #     sc_files.sort()
    #     y_files.sort()                
    #     sc_dfs = []
    #     y_dfs = []

    #     for file in sc_files:
    #         sc_dfs.append(pd.read_csv(file, header=None))
    #     for file in y_files:
    #         y_dfs.append(pd.read_csv(file, header=None))
        
    #     # concatenate the dataframes
    #     sc_df = pd.concat(sc_dfs, axis=0)
    #     y_df = pd.concat(y_dfs, axis=0)

    #     # change y column name to "label"
    #     y_df.columns = ['label']

    #     # concatenate the dataframes
    #     joined_df = pd.concat([sc_df, y_df], axis=1)

    #     # save the dataframe to a file
    #     filename = 'joined_' + str(i) + '.csv'
    #     # make output file path
    #     output_file_path = os.path.join(joined_folder_path, filename)
    #     print("Saving file: " + output_file_path)
    #     # save the dataframe to a file
    #     joined_df.to_csv(output_file_path, index=False)

    #     i+=1


main()