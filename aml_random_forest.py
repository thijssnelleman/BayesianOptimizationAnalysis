# -*- coding: utf-8 -*-
"""AML Random Forest

File on optimizing Random Forest with hyperparameters. To put to personal use,
change save_path variables, insert your own data set (last column is variable to predict)
"""

!pip install scikit-optimize
#Basic Bayesion optimizer init
#https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

import sys
import numpy as np
import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
from sklearn.ensemble import RandomForestRegressor
from skopt.space import Real, Integer, Categorical
import xgboost as xgb

save_path = "PATH"
numberOfCalls = 50 #Calls to objective function
numberOfRuns = 10 #Number of total runs to do (Only >1 for statistics)

def generateTrainTestSets(dataset):
    np.random.shuffle(dataset)
    num_rows, num_cols = dataset.shape
    split = int((num_rows/100) * 80)
    X_train, X_test = dataset[:split,0:num_cols-1], dataset[split:,0:num_cols-1]
    Y_train, Y_test = dataset[:split,num_cols-1], dataset[split:,num_cols-1]
    return X_train, X_test, Y_train, Y_test



#Select which data set is to be used
dataset = 2
if dataset == 1:
    #Load the KIN8NM dataset https://www.openml.org/d/189
    KIN8NM_dataset = np.load('KIN8NM.npy')
    xtrain, xtest, ytrain, ytest = generateTrainTestSets(KIN8NM_dataset)
if dataset == 2:
    #Load ASP-POTASSCO dataset https://www.openml.org/d/41704
    CCF_dataset = np.load('ASP_POTASSCO.npy')
    xtrain, xtest, ytrain, ytest = generateTrainTestSets(CCF_dataset)
if dataset == 3:
    #LOAD ??
    print("Invalid Selection")

#n_estimators specifies the size of the forest to be trained; it is converted to num_parallel_tree, instead of the number of boosting rounds
#learning_rate is set to 1 by default
#colsample_bynode and subsample are set to 0.8 by default
def xgb_rf_error(params):
    #xtrain, xtest, ytrain, ytest = generateTrainTestSets(ISOLET_dataset)
    xgb_rf = xgb.XGBRFRegressor(max_depth=1000000, learning_rate=params[0],
                                n_estimators=params[1],
                                colsample_bynode=params[2],
                                subsample=params[3], random_state=42).fit(xtrain, ytrain)

    error = 0.0
    for i,e in enumerate(xtest):
        res = xgb_rf.predict(e.reshape(1, -1))[0]
        error += abs(ytest[i] - res)
    return error

#Do XGB Random Forest Bayesian Optimization
xgb_rf_space = [Real(0, 1, name = "learning_rate"),
                Integer(1, 1024, name='est'),
                Real(0, 1, name = "csbn"),
                Real(0, 1, name = "ss")
]

for run in range(numberOfRuns):
    xgb_res = gp_minimize(xgb_rf_error,                  # the function to minimize
                    xgb_rf_space,      # the bounds on each dimension of x
                    acq_func="EI",      # the acquisition function
                    n_calls=numberOfCalls,         # the number of evaluations of f
                    n_initial_points=1,  # the number of random initialization points
                    verbose=True)   # the random seed
    fn = "XGB_RF-Result-RUN-" + str(run + 1)
    np.save(save_path + fn, np.array(xgb_res.func_vals))


#------------------------------------------------------------------------------
# RANDOM SEARCH BELOW
# Only use for statistics
# -----------------------------------------------------------------------------

#Do Random search
from sklearn.model_selection import RandomizedSearchCV
import time
import random
from random import randrange

prev = time.time()
for run in range(numberOfRuns):
    random_res = []
    for iterations in range(numberOfCalls):
        #rSearch = RandomizedSearchCV(RandomForestRegressor(), rParams, n_iter=1)
        lr = random.uniform(0, 1)
        trees = random.randrange(0, 1024) + 1
        cs = random.uniform(0,1)
        ss = random.uniform(0,1)
        rSearch = xgb.XGBRFRegressor(max_depth=1000000, learning_rate=lr, n_estimators=trees, colsample_bynode=cs, subsample=ss, random_state=42).fit(xtrain, ytrain)
        error = 0.0
        for i, e in enumerate(xtest):
            pred = rSearch.predict(e.reshape(1, -1))[0]
            if int(round(pred)) != int(ytest[i]):
                error += abs(ytest[i] - pred)
        random_res.append(error)
        now = time.time()
        print("Done with iteration: " + str(iterations) + " in " + str(now - prev))
        print("Error: " + str(error))
        prev = now
    fn = "Random_RF-Result-RUN-" + str(run+1)
    np.save(save_path + fn, np.array(random_res))
