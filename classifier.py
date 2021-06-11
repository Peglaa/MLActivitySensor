import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
import pathlib
import csv

import tensorflow as tf
import tensorflow.keras as keras

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.backend import sparse_categorical_crossentropy
from tensorflow.python.keras.layers.core import Dropout
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def get_feature_data():
    x = pd.read_csv("X_train.csv")
    y = pd.read_csv("y_train.csv")
    x_test = pd.read_csv("X_test.csv", skiprows=1)
    y_test = pd.read_csv("y_test.csv")

    print(x_test)

    return x,y, x_test, y_test

def select_data(x, y, x_test, y_test):

    selector = SelectKBest(f_classif, k=10)
    selected_features = selector.fit_transform(x, y)
    test_features = selector.fit_transform(x_test, y_test)

    return selected_features, test_features, y, y_test

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(X.shape[1],)),


        keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu"),

        
        keras.layers.Dense(8, kernel_regularizer=keras.regularizers.l2(0.001), activation="relu"),

        
        keras.layers.Dense(13, activation="softmax")
    ])

    return model

def create_overfitting_graph():
    fig, axs = plt.subplots(2)

    axs[0].plot(history.history["sparse_categorical_accuracy"], label="train_accuracy")
    axs[0].plot(history.history["val_sparse_categorical_accuracy"], label="test_accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    axs[1].plot(history.history["loss"], label="train_error")
    axs[1].plot(history.history["val_loss"], label="test_error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def eval_metric(model, history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric. 
    Training and validation metric are plotted in a
    line chart for each epoch.
    
    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]    
    e = range(1, NB_START_EPOCHS + 1)    
    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.xlabel('Epoch number')
    plt.ylabel(metric_name)
    plt.title('Comparing training and validation ' + metric_name + ' for ' + model.name)
    plt.legend()
    plt.show()

def create_tensorflowlite_file(model):
    TF_LITE_MODEL_NAME = "tf_lite_model.tflite"
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    open(TF_LITE_MODEL_NAME, "wb").write(tflite_model)


if __name__ == "__main__":
    NB_START_EPOCHS = 50
    X, Y , x_test, y_test= get_feature_data()
    #X_train, X_test, Y_train, Y_test = select_data(X,Y, x_test, y_test)

    model = create_model()

    optimizer = keras.optimizers.Adam(learning_rate = 0.001)
    model.compile(  optimizer=optimizer,
                    loss="sparse_categorical_crossentropy",
                    metrics=["sparse_categorical_accuracy"])

    model.summary()

    history = model.fit(X, Y, validation_data=(x_test, y_test), epochs=50, shuffle=True, batch_size=512)

    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, batch_size=512)
    print("test loss, test acc:", results)

    eval_metric(model, history, 'loss')
    create_overfitting_graph()

    #create_tensorflowlite_file(model)