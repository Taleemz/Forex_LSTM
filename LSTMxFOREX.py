# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 22:06:27 2024

@author: Khronik
"""

# For data manipulation
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl

import pandas as pd
from collections import deque
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint
import time
from sklearn import preprocessing

import seaborn as sns
import yfinance as yf

# Import date class from datetime module
from datetime import date
from dateutil.relativedelta import relativedelta
 
# For visualisation
import matplotlib.pyplot as plt
plt.style.use('seaborn-darkgrid')
%matplotlib inline

#Set the constants that would be used
SEQ_LEN = 3  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 1  # how far into the future are we trying to predict?
RATIO_TO_PREDICT = "Adj Close_0"
EPOCHS = 10  # how many passes through our data
BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


# Set the classification of the future price based on current price
def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current, that's a buy, or a 1
        return 1
    else:  # otherwise... it's a 0!
        return 0

# Create the sliding window series version of the dataset
def PrePro (df,labels):
    seq_df = pd.DataFrame()
    step=0
    for i in labels:
        for j in range(SEQ_LEN,-1,-1):
            col_name = i + '_' + str(j)
            seq_df[col_name] = df[i].shift(j,axis=0)
            #seq_df = pd.concat([new_col,seq_df], axis=1)
            step+=1
        if step == (len(labels)*SEQ_LEN):
            break
    return seq_df


# Resampling dataset to have equal number of ups and downs
def Shuffle_Split(df):
    df = df.drop(columns=['future'])
    X = pd.DataFrame()
    y = pd.DataFrame()
    df = df.sample(frac = 1)
    buys = df[df['target'] == 1]
    sells = df[df['target'] == 0]
    buys = buys.sample(frac = 1)
    sells = sells.sample(frac = 1)
    lower = min(len(buys), len(sells))
    buys = buys.head(lower)
    sells = sells.head(lower)
    result = pd.concat([buys, sells], axis=0)
    result = result.sample(frac = 1)
    X = result.drop(columns=['target'])
    # reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    y = result[['target']]
    return X,y
    
main_df = pd.DataFrame() # begin empty

# Set the ticker as 'EURUSD=X'
main_df = yf.download('EURUSD=X', period='10y', interval='1d')

#print(main_df.columns)

# Set the index to a datetime object
date_time = main_df.index
main_df.fillna(method="ffill", inplace=True)  # if there are gaps in data, use previously known values
main_df.dropna(inplace=True)
#print(main_df.head())  # how did we do??

# Transform index type from datetime to string
date_time = pd.to_datetime(date_time, format='%d.%m.%Y %H:%M:%S')

plot_cols = ['Close', 'High', 'Low']
plot_features = main_df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)

plot_features = main_df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)

main_df.describe().transpose()

timestamp_s = np.arange(0, len(main_df), dtype=int)
print(timestamp_s)

day = 365.2425
year = 10

main_df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
main_df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
main_df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
main_df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

plt.plot(np.array(main_df['Day sin'])[:1000])
plt.plot(np.array(main_df['Day cos'])[:1000])
plt.xlabel('Time [h]')
plt.title('Time of day signal')

plt.plot(np.array(main_df['Year sin'])[:24])
plt.plot(np.array(main_df['Year cos'])[:24])
plt.xlabel('Time [h]')
plt.title('Time of day signal')

Labels = ['Day sin','Day cos','Year sin','Year cos','Adj Close']
main_df = main_df[Labels]  # don't need this anymore.
column_indices = {name: i for i, name in enumerate(main_df.columns)}

new_window = PrePro(main_df,Labels)

new_window['future'] = new_window[f'{RATIO_TO_PREDICT}'].shift(-FUTURE_PERIOD_PREDICT)
new_window['target'] = list(map(classify, new_window[f'{RATIO_TO_PREDICT}'], new_window['future']))

new_window = new_window.dropna()

n = len(new_window)
train_df = new_window[0:int(n*0.7)]
test_df = new_window[int(n*0.7):int(n*0.9)]
validation_main_df = new_window[int(n*0.9):]

train_x, train_y = Shuffle_Split(train_df)
test_x, test_y = Shuffle_Split(test_df)
validation_x, validation_y = Shuffle_Split(validation_main_df)

model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

tensorboard = TensorBoard(log_dir="logs\{}".format(NAME))

filepath = "RNN_Final-{epoch:02d}-{val_loss:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
checkpoint = ModelCheckpoint("models\{}.model".format(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')) # saves only the best ones

# Train model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_x, test_y),
    callbacks=[tensorboard, checkpoint],
)

# Score model
score = model.evaluate(test_x, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models\{}".format(NAME))


