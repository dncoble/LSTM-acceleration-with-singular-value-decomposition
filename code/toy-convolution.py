import joblib
import pickle
import tensorflow.keras as keras
import matplotlib.pyplot as plt
# from math import isnan
import numpy as np
from numpy.random import randint
"""
As an aside, I want to try to create a convolutional model and see how well
it performs on DROPBEAR, and how small I can make it. Since pin location is
heavily tied to frequency, I expect that even a small convolutional model
can do well.
"""
#%% load data
from load_preprocess import preprocess
# preprocess()
print("loading data...")
load_X_train = open("./pickles/X_train", 'rb')
load_y_train = open("./pickles/y_train", 'rb')
load_t_train = open("./pickles/t_train", 'rb')
load_X_test = open("./pickles/X_test", 'rb')
load_y_test = open("./pickles/y_test", 'rb')
load_t_test = open("./pickles/t_test", 'rb')


X_train = pickle.load(load_X_train)
y_train = pickle.load(load_y_train)
t_train = pickle.load(load_t_train)
X_test = pickle.load(load_X_test)
y_test = pickle.load(load_y_test)
t_test = pickle.load(load_t_test)

pin_scaler = joblib.load('./pickles/pin_scaler')
acc_scaler = joblib.load('./pickles/pin_scaler')

load_X_train.close()
load_y_train.close()
load_t_train.close()
load_X_test.close()
load_y_test.close()
load_t_test.close()
#%% model is a splice of the period and amplitude for the lower two layers
def split_train_random(batch_size, train_len):
    runs = X_train.shape[0]
    run_size = X_train.shape[1]
    indices = [(randint(1, runs), randint(0, run_size - train_len)) for i in range(batch_size)]
    X_mini = np.copy(np.array([X_train[index[0],index[1]:index[1]+train_len] for index in indices]))
    y_mini = np.copy(np.array([y_train[index[0],index[1]+train_len][0] for index in indices]))
    return X_mini, y_mini
print("making model...")
units = [30, 30, 30, 30]

# model = keras.Sequential(
#     [keras.layers.LSTM(units[0],return_sequences=True,input_shape=[None, 1])] + 
#     [keras.layers.LSTM(i, return_sequences = True) for i in units[1:]] +
#     [keras.layers.TimeDistributed(keras.layers.Dense(1))]
# )

# WaveNet - doesn't work

# model = keras.models.Sequential()
# model.add(keras.layers.InputLayer(input_shape=[None,1]))
# for rate in (1,2,4) * 2:
#     model.add(keras.layers.Conv1D(filters=10,kernel_size=2,padding="causal",
#                                   activation="relu",dilation_rate=rate))
# model.add(keras.layers.Conv1D(filters=5,kernel_size=1))

# combined convolutional and small LSTMs

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=15,kernel_size=10,padding="causal",
                        activation="relu"),
    keras.layers.LSTM(15, return_sequences = True),
    keras.layers.LSTM(15, return_sequences = True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])

model.compile(loss="mse",
    optimizer="adam",
    metrics = ['accuracy']
)
print("training model...")

X_mini, y_mini = split_train_random(10000, 100)
model.fit(X_mini, y_mini, epochs=20)
 
#%% evaluate model
from sklearn.metrics import mean_squared_error
# from keras.utils.layer_utils import count_params
from math import sqrt
y_pred_scaled = pin_scaler.inverse_transform(model.predict(X_test).squeeze())
y_test_scaled = pin_scaler.inverse_transform(y_test.squeeze())

mse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_test_scaled, y_pred_scaled)])/y_test.shape[0]
rmse = sqrt(sum([mean_squared_error(y_t, y_p, squared=False)**2 for y_t, y_p in zip(y_test_scaled, y_pred_scaled)])/y_test.shape[0])

print("mean squared error: " + str(mse))
print("root mean squared error: " + str(rmse))

weight_sum = 0
for weights in model.trainable_weights:
    weight_sum += np.size(weights)

print("total weights: " + str(weight_sum)) 

plt.figure(figsize=(7,3.3))
plt.title("LSTM prediction of pin location")
plt.plot(t_test[0], y_pred_scaled[0], label = "predicted pin location")
plt.plot(t_test[0], y_test_scaled[0], label = "actual pin location",alpha=.8)
plt.xlabel("time (s)")
plt.ylabel("pin location (m)")
plt.ylim((0.045, .23))
plt.legend(loc=1)
plt.tight_layout()
plt.savefig("./plots/convolutional_prediction.png", dpi=800)

model.save("./model_saves/conv1d")