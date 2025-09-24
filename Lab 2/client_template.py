#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 02:19:00 2022

@author: alexanderpavlyuk
"""

import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
from FMI_data import daily_avg
from sklearn.metrics import mean_squared_error


# Regularization by data augmentation.
# Adding d synthetic data points (x^{[d]]}, y^{[d]})
# to the local data set (d is the number of features).
# The feature vectors are the unit vectors:
# x^{(1)} = (1,0,0,…,0)^{T}
# x^{(2)} = (0,1,0,….0)^{T}
# …
# x^{(d)} = (0,0,…,0,1)^{T}
# and the labels are y^{(r)} = w’_{r}
# (the r-th entry of the global weight vector w’).
def data_augmentation(x, y, w):
    x_synth = np.array([])
    y_synth = np.array([])
    if len(np.shape(x)) == 1:
        n_x = 1
    else:
        n_x = np.shape(x)[1]
    if len(np.shape(y)) == 1:
        n_y = 1
    else:
        n_y = np.shape(y)[1]
    for i in range(n_x):
        x_synth = np.append(x_synth, np.zeros(n_x))
        x_synth[i * n_x + i] = 1
        y_synth = np.append(y_synth, w[i])
    x_synth = np.array(np.split(x_synth, n_x))
    y_synth = np.array(np.split(y_synth, n_y))
    x_aug = np.append(x, x_synth).reshape(-1, n_x)
    y_aug = np.append(y, y_synth)

    return x_aug, y_aug


# Construct feature matrix and labels for the FMI data
# Output: feature matrix and labels array
def FMI_features_labels(df):
    # Features & lables creation
    X = np.split(np.zeros(5 * (len(df) - 5)), (len(df) - 5))
    y = np.zeros(len(df) - 5)
    for i in range(len(df) - 5):
        k = 0
        for j in range(i, i + 5):
            X[i][k] = df.iloc[j]['Average air temperature']
            k += 1
        y[i] = df.iloc[i + 5]['Average air temperature']

    return X, y


# Linear regression with data augmentation
# Output: local weight vector and intercept
def linear_with_augmentation(X_tr, y_tr, global_w):
    # Linear regression creation
    # TODO: define linear regression
    # reg =

    # Regularization by data augmentation
    # TODO: implement data augmentation with one of the helper functions
    # X_augmented, y_augmented =

    # Fitting (RERM solution)
    # TODO: fit the linear model

    return list(reg.coef_), reg.intercept_


# Data
# TODO:
#  1. Read the training data set from csv file.
#  2. Implement data pre-processing with the imported helper function.
#  3. Extract features and labels using one of the functions above.
# df_train =
# X_train, y_train =

# TODO:
#  1. Read the testing data set from csv file.
#  2. Implement data pre-processing with the imported helper function.
#  3. Extract features and labels using one of the functions above.
# df_test =
# X_test, y_test =

# Obtaining the initial local weight vector.
# Initially, the global weight vector consists of zeros
zero_global_w = np.zeros(np.shape(X_train)[1])
# TODO: obtain the initial local weight vector using one of the functions above
# local_weight, intercept =

# Dictionary to be sent to the server
# weight - contains the local weight vector
# clients_cnt - contains the number of clients in the network
# email - contains the client's email address to be logged in to the server
# X_test - contains the test set feature matrix
# testing - boolean variable:
#       False - the server sends back the global weight vector.
#       True - the server sends back the predicted labels
data = {'weight': local_weight, 'clients_cnt': 10, 'email': '',
        'X_test': list(map(list, X_test)), "testing": False}

local_storage = {'intercept': intercept}

# Email specification
print("Specify your email address:")
email = str(input())
data['email'] = email

# cmd instructions
print("Enter GET command to receive global parameter vector\n"
      "Enter POST command to send data to the server\n"
      "Enter QUIT command to stop the program")
command = str(input())

while command != "QUIT":

    # POST command
    if command == "POST":

        # Initial value for command_post to satisfy the while-loop condition
        command_post = ""
        print("Enter FEDAVG command to send the local weight vector and implement federated averaging algorithm\n"
              "Enter TEST command to send the test feature vectors and receive the predictions")

        while command_post != "FEDAVG" and command_post != "TEST":
            command_post = str(input())

            if command_post == "FEDAVG":
                data["testing"] = False
            elif command_post == "TEST":
                data['testing'] = True
            else:
                print("Unknown command")

        # Data sending
        # "response" variable stores the server's output
        # "tmp" variable stores the decoded server's output
        response = requests.post("http://fljung.cs.aalto.fi:5000/test", json=data)
        tmp = response.json()

        # The printed output depends on the server's output type
        if type(tmp['response']) == str:
            print(tmp['response'])
        else:
            y_pred_DT = tmp['response']
            y_pred_FedAvg = np.dot(X_test, data['weight']) + local_storage['intercept']
            mse_DT = mean_squared_error(y_test, y_pred_DT)
            mse_FedAvg = mean_squared_error(y_test, y_pred_FedAvg)
            print("Decision tree MSE: {}\nFederated Averaging MSE: {}".format(mse_DT, mse_FedAvg))


    # GET command
    elif command == "GET":

        # Data sending
        # "response" variable stores the server's output
        # "tmp" variable stores the decoded server's output
        response = requests.get("http://fljung.cs.aalto.fi:5000/create", json=data)
        tmp = response.json()

        # If the server's output is a string value, then just print it
        if type(tmp['response']) == str:
            print(tmp['response'])

        # If the server's output is not a string value, then perform local training using the new global weight vector
        else:

            # Receive the global weight
            global_weight = np.array(tmp['response'])

            # Update local weight vector
            # TODO: update the local weight vector with one of the functions above
            # local_weight, intercept =
            local_storage['intercept'] = intercept
            data['weight'] = local_weight

            print("The global weight vector is ", global_weight)
            print("The local weight vector is ", data['weight'])

    else:
        print("Unknown command")

    command = str(input())
print("You have quited the program")
