import joblib
import pickle
import tensorflow.keras as keras
import matplotlib.pyplot as plt
# from math import isnan
import numpy as np
from numpy.random import randint
""" v2 and we are moving to tensorflow 2.8.0 and trying to use the the adam
optimizer rather than the pure stochastic gradient descent I programmed
initially. I am also getting rid of the pretrained lower layers (if I can)
i.e., training as it should be.
 - add statefullness
 - work on plots
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
model = keras.Sequential(
    [keras.layers.LSTM(units[0],return_sequences=True,input_shape=[None, 1])] + 
    [keras.layers.LSTM(i, return_sequences = True) for i in units[1:]] +
    [keras.layers.TimeDistributed(keras.layers.Dense(1))]
)


model.compile(loss="mse",
    optimizer="adam",
    metrics = ['accuracy']
)
print("training model...")

X_mini, y_mini = split_train_random(10000, 100)
model.fit(X_mini, y_mini, validation_data=(X_test,y_test), epochs=20)
#%%
# model = keras.models.load_model("./model_saves/pretrained_sequential")
#%% plots and evaluate accuracy
from sklearn.metrics import mean_squared_error
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
plt.xlabel("time [s]")
plt.ylabel("pin location [m]")
plt.ylim((0.045, .23))
plt.legend(loc=1)
plt.tight_layout()
plt.savefig("./plots/full_model_prediction.png", dpi=800)

model.save("./model_saves/pretrained_sequential")