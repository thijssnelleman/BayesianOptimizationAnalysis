import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import random
from skopt import gp_minimize
from skopt.plots import plot_convergence
from functools import partial
save_path = "/content/drive/MyDrive/LIACS/AML/mlp/"

(fmnist_X_train_full, fmnist_Y_train_full), (fmnist_X_test, fmnist_Y_test) = keras.datasets.fashion_mnist.load_data()
fmnist_X_valid, fmnist_X_train = fmnist_X_train_full[:5000] / 255.0, fmnist_X_train_full[5000:] / 255.0
fmnist_Y_valid, fmnist_Y_train = fmnist_Y_train_full[:5000], fmnist_Y_train_full[5000:]

def train_mlp(x):
    learning_rate, momentum, decay_rate, batchsize = x[0],x[1],x[2],x[3]
    print(learning_rate, momentum, decay_rate, batchsize)
    mlp = keras.models.Sequential()
    mlp.add(keras.layers.Flatten(input_shape=[28,28]))
    mlp.add(keras.layers.Dense(512, activation="relu"))
    mlp.add(keras.layers.Dense(256, activation="relu"))
    mlp.add(keras.layers.Dense(10,activation="softmax"))
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)
    mlp.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    
    history = mlp.fit(fmnist_X_train, fmnist_Y_train, verbose=0, batch_size=batchsize, epochs = 30, validation_data=(fmnist_X_valid,fmnist_Y_valid)) 
    return history.history['loss'][-1]

def train_mlp_randomsearch(learning_rate, momentum, decay_rate, batchsize):
    mlp = keras.models.Sequential()
    mlp.add(keras.layers.Flatten(input_shape=[28,28]))
    mlp.add(keras.layers.Dense(512, activation="relu"))
    mlp.add(keras.layers.Dense(256, activation="relu"))
    mlp.add(keras.layers.Dense(10,activation="softmax"))
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)
    mlp.compile(loss="sparse_categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    history = mlp.fit(fmnist_X_train, fmnist_Y_train, verbose=0, batch_size=batchsize, epochs = 30, validation_data=(fmnist_X_valid,fmnist_Y_valid))
    return history.history['loss'][-1]

def train_cnn(x):
    learning_rate, momentum, decay_rate, batchsize = x[0],x[1],x[2],x[3]
    DefaultConv2D = partial(keras.layers.Conv2D,kernel_size=3, activation='relu', padding="SAME")

    cnn = keras.models.Sequential([
                                    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28,28,1]),
                                    keras.layers.MaxPooling2D(pool_size=2),
                                    DefaultConv2D(filters=128),
                                    DefaultConv2D(filters=128),
                                    keras.layers.MaxPooling2D(pool_size=2),
                                    DefaultConv2D(filters=256),
                                    DefaultConv2D(filters=256),
                                    keras.layers.MaxPooling2D(pool_size=2),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(units=128, activation='relu'),
                                    keras.layers.Dropout(0.5),
                                    keras.layers.Dense(units=64, activation='relu'),
                                    keras.layers.Dropout(0.5),
                                    keras.layers.Dense(units=10, activation='softmax'),                                
    ])
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)
    cnn.compile(loss="sparse_categorical_crossentropy", optimizer = sgd, metrics=["accuracy"])
    
    history = cnn.fit(fmnist_X_train, fmnist_Y_train, verbose=0, batch_size=batchsize, epochs = 30, validation_data=(fmnist_X_valid,fmnist_Y_valid))
    return history.history['loss'][-1]

def train_cnn_randomsearch(learning_rate, momentum, decay_rate, batchsize):
    DefaultConv2D = partial(keras.layers.Conv2D,kernel_size=3, activation='relu', padding="SAME")

    cnn = keras.models.Sequential([
                                    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28,28,1]),
                                    keras.layers.MaxPooling2D(pool_size=2),
                                    DefaultConv2D(filters=128),
                                    DefaultConv2D(filters=128),
                                    keras.layers.MaxPooling2D(pool_size=2),
                                    DefaultConv2D(filters=256),
                                    DefaultConv2D(filters=256),
                                    keras.layers.MaxPooling2D(pool_size=2),
                                    keras.layers.Flatten(),
                                    keras.layers.Dense(units=128, activation='relu'),
                                    keras.layers.Dropout(0.5),
                                    keras.layers.Dense(units=64, activation='relu'),
                                    keras.layers.Dropout(0.5),
                                    keras.layers.Dense(units=10, activation='softmax'),                                
    ])
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate)
    cnn.compile(loss="sparse_categorical_crossentropy", optimizer = sgd, metrics=["accuracy"])
    
    history = cnn.fit(fmnist_X_train, fmnist_Y_train, verbose=1, batch_size=batchsize, epochs = 30, validation_data=(fmnist_X_valid,fmnist_Y_valid))
    return history.history['loss'][-1]

#Train MLP with BO
for run in range(10):    
    res = gp_minimize(train_mlp,        # the function to minimize
                    [(1e-3,1e-2),       # learningrate
                    (0.0,1.0),          # momentum
                    (1e-6,1e-5),        # decay rate
                    (16,200)],          # batchsize
                    acq_func="EI",      # the acquisition function
                    n_calls=50,         # the number of evaluations of f
                    n_initial_points=5, # the number of random initialization points
                    )
    fn = "MLP-Result-RUN-" + str(run + 1)
    np.save(save_path + fn, np.array(res.func_vals))

#Train MLP with Random Search
for i in range(10):
    randomsearch_result = []
    minFx = 100
    for j in range (50):    
        learning_rate = np.random.uniform(1e-3,1e-2)
        momentum = np.random.uniform(0.0,1.0)
        decay_rate = np.random.uniform(1e-6,1e-5)
        batchsize = np.random.randint(16,200)
        loss = train_mlp_randomsearch(learning_rate, momentum, decay_rate, batchsize)
        if loss < minFx:
            minFx = loss
        randomsearch_result.append(minFx)
    fn = "RandomMLP-Result-RUN-" + str(i+1)
    np.save(save_path + fn, np.array(randomsearch_result))

#reshape data for CNN
fmnist_X_train = fmnist_X_train.reshape(fmnist_X_train.shape[0], 28, 28, 1)
fmnist_X_valid = fmnist_X_valid.reshape(fmnist_X_valid.shape[0], 28, 28, 1)

#Train CNN with BO
for run in range(10):    
    res = gp_minimize(train_cnn,        # the function to minimize
                    [(1e-3,1e-2),       # learningrate
                    (0.0,1.0),          # momentum
                    (1e-6,1e-5),        # decay rate
                    (16,200)],          # batchsize
                    acq_func="EI",      # the acquisition function
                    n_calls=50,         # the number of evaluations of f
                    n_initial_points=5, # the number of random initialization points
                    )
    fn = "CNN-Result-RUN-" + str(run + 1)
    np.save(save_path + fn, np.array(res.func_vals))

#Train CNN with Random Search
for i in range(10):
    randomsearch_result = []
    minFx = 100
    for j in range (50):    
        learning_rate = np.random.uniform(1e-3,1e-2)
        momentum = np.random.uniform(0.0,1.0)
        decay_rate = np.random.uniform(1e-6,1e-5)
        batchsize = np.random.randint(16,200)
        loss = train_cnn_randomsearch(learning_rate, momentum, decay_rate, batchsize)
        if loss < minFx:
            minFx = loss
        randomsearch_result.append(minFx)
    fn = "Random_CNN-Result-RUN-" + str(i+1)
    np.save(save_path + fn, np.array(randomsearch_result))