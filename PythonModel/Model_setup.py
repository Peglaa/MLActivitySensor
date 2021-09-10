import numpy as np
#import pandas as pd
import os
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from keras import backend as k
from tensorflow.python.tools import freeze_graph, optimize_for_inference_lib
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

#prepare dataset
path='DataSet/'
files = os.listdir(path)
files.pop()

df = pd.DataFrame()
for file in files:
    df_temp = pd.read_csv(path+file, header=1)
    df=pd.concat([df,df_temp], sort=False)

left_pocket = df[df.columns[1:10]]
right_pocket = df[df.columns[15:24]]
right_pocket.columns=left_pocket.columns

train_data = pd.concat([left_pocket, right_pocket], sort=False)

labels = pd.concat([df["Unnamed: 69"], df["Unnamed: 69"]], axis=0, sort=False)
labels.columns=["Activity"]

train_data["Activity"] = labels

train_data.loc[(train_data.Activity == "upsatirs")] = "upstairs"

#split the dataset
X_train = pd.read_csv("XTrain.csv")
Y_train = pd.read_csv("YTrain.csv")

encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)

#split to train and test data
def train_test_split(X, Y, split_size=0.8):
    split = int(len(X) * split_size)
    train_x = X[:split]
    train_y = Y[:split]
    test_x = X[split:]
    test_y = Y[split:]
    return train_x, test_x, train_y, test_y

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", y_train.shape)
print("X_test shape: ", X_test.shape)
print("Y_test shape: ", y_test.shape)

#convert the dataset into timeseries sequence(we have to feed a continuous input to our model, in batches of 100 samples)
n_time_steps = 100
n_features = 9

X_train = np.asarray(X_train).astype(np.float32)
y_train = np.asarray(y_train).astype(np.float32)
X_test = np.asarray(X_test).astype(np.float32)
y_test = np.asarray(y_test).astype(np.float32)

train_gen = TimeseriesGenerator(X_train, y_train, length= n_time_steps, batch_size=1024)
test_gen = TimeseriesGenerator(X_test, y_test, length= n_time_steps, batch_size=1024)

#create simple LSTM model
model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape = (n_time_steps, n_features), kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), name="lstm_1"))
model.add(Flatten(name="flatten"))
model.add(Dense(64, activation="relu", kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), name="dense_1"))
model.add(Dense(len(np.unique(y_train)), activation="softmax", kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001), name="output"))
model.summary()

#compile the model
#model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(), metrics=["accuracy"])

#prepare callback
#callback = [ModelCheckpoint("model.h5", save_weights_only=False, save_best_only=True, verbose=1)]

#train_model
#history = model.fit_generator(train_gen, epochs=2, validation_data=test_gen, callbacks=callback)

#save model for android usage
model = load_model('model.h5')

input_node_name = ["lstm1_input"]
output_node_name = "output/Softmax"
model_name = "model"

tf.io.write_graph(k.get_session().graph_def, 'models', model_name + "_graph.pbtxt")
saver = tf.compat.v1.train.Saver()
saver.save(k.get_session(), 'models/'+model_name + ".chkp")

freeze_graph.freeze_graph('models/'+model_name + "_graph.pbtxt", None, False, 'models/'+model_name+".chkp", output_node_name, "save/restore_all", 'save/Const:0', 'models/frozen' + model_name + ".pb", True, "")
