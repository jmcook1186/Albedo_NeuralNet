import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


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


# BUILD MODEL
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


model = build_model()



EPOCHS = 500

history = model.fit(
  train_dataset, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)


loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f}".format(mae))

start = time.time()

for i in range(10290):

    test_predictions = model.predict(test_dataset).flatten()

elapsed_time = time.time() - start
print(elapsed_time)


a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [BBA]')
plt.ylabel('Predictions [BBA]')
lims = [0, 1.5]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


