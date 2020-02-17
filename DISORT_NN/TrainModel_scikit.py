import sklearn_xarray as skx
import sklearn as skl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
import xarray as xr
import pandas as pd

dataset = pd.read_csv('/home/joe/Code/Albedo_NeuralNet/DISORT_NN/trainingData.csv')

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


NN = skx.wrap(MLPRegressor(hidden_layer_sizes=(3000, 3000),
                                 tol=1e-5, max_iter=3000, random_state=0))

NN.fit(train_dataset,train_labels)
predictions = NN.predict(test_dataset)
plt.scatter(predictions,test_labels)

# pickle the classifier model for archiving or for reusing in another code
joblibfile = str('/home/joe/Code/Albedo_NeuralNet/DISORT_NN/Model_SKL.pkl')
joblib.dump(clf, joblibfile)

# to load this classifier into another code use the following syntax:
# clf = joblib.load(joblib_file)