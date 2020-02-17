import sklearn_xarray as skx
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import xarray as xr
import pandas as pd

dataset = pd.read_csv('/home/joe/Code/SNICAR_NeuralNet/NNtraining_data.csv')

# SPLIT
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# LABELS
train_labels = train_dataset.pop('BBA')
test_labels = test_dataset.pop('BBA')
column_labels = train_dataset.columns.values

# NORMALIZE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_dataset)
train_dataset = (scaler.transform(train_dataset))
train_dataset = pd.DataFrame(train_dataset,columns=column_labels)

scaler.fit(test_dataset)
test_dataset = (scaler.transform(test_dataset))


NN = skx.wrap(MLPRegressor(hidden_layer_sizes=(5000, 5000),
                                 tol=1e-5, max_iter=1500, random_state=0))

NN.fit(train_dataset,train_labels)
predictions = NN.predict(test_dataset)
plt.scatter(predictions,test_labels)