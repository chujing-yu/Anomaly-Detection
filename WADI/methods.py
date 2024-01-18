import pandas as pd
import numpy as np
from tensorflow.python import keras as kr
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.metrics import accuracy
from tensorflow.python.keras.models import load_model
import json


# Helper function for scaling continous values
def minmax_scale_values(training_df, testing_df, col_name):
    scaler = MinMaxScaler()
    scaler = scaler.fit(training_df[col_name].values.reshape(-1, 1))
    train_values_standardized = scaler.transform(training_df[col_name].values.reshape(-1, 1))
    training_df[col_name] = train_values_standardized
    test_values_standardized = scaler.transform(testing_df[col_name].values.reshape(-1, 1))
    testing_df[col_name] = test_values_standardized


# Helper function for one hot encoding
def encode_text(training_df, testing_df, name):
    training_set_dummies = pd.get_dummies(training_df[name])  # get_dummies 是利用pandas实现one hot 编码的方式
    testing_set_dummies = pd.get_dummies(testing_df[name])
    for x in training_set_dummies.columns:
        dummy_name = "{}_{}".format(name, x)
        training_df[dummy_name] = training_set_dummies[x]
        if x in testing_set_dummies.columns:
            testing_df[dummy_name] = testing_set_dummies[x]
        else:
            testing_df[dummy_name] = np.zeros(len(testing_df))
    training_df.drop(name, axis=1, inplace=True)
    testing_df.drop(name, axis=1, inplace=True)


# calculate_losses是一个辅助函数，计算每个数据样本的重建损失
def calculate_losses(x, preds):
    losses = np.zeros(len(x))
    losses = np.mean(np.power(preds - x, 2), axis=1)
    return losses
