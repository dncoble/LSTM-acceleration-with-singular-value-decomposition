import json
import pickle
import joblib
import math
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import sklearn as sk
import time

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

"""
SVD acceleration using svd_classes_v3 classes.

TensorFlow 2.5.0
TensorFlow Model Optimization 0.6.0 (for compatibility with TF 2.5.0)
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
#%%
def split_train_random(batch_size, train_len):
    runs = X_train.shape[0]
    run_size = X_train.shape[1]
    indices = [(randint(1, runs), randint(0, run_size - train_len)) for i in range(batch_size)]
    X_mini = np.copy(np.array([X_train[index[0],index[1]:index[1]+train_len] for index in indices]))
    y_mini = np.copy(np.array([y_train[index[0],index[1]+train_len][0] for index in indices]))
    return X_mini, y_mini
def signaltonoise(signal, noisy_signal, dB=True):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)

from svd_classes_v3 import make_LSTM_singular_model, PrunableTimeDistributed, make_LSTM_reduced_model
model = keras.models.load_model("./model_saves/pretrained_sequential")
#%% regularization
smodel = make_LSTM_singular_model(model, hoyer=.01, merged_kernel=False)

X_mini, y_mini = split_train_random(3200, 100)
smodel.compile(
    loss="mse",
    optimizer="adam"
)
smodel.fit(X_mini, y_mini, batch_size=32, validation_data=(X_test,y_test), epochs=10)
s = []
for layer in smodel.layers[:-1]: 
    s.append(layer.cell.kernel.numpy())
    s.append(layer.cell.recurrent_kernel.numpy())
#%% pruning
smodel = make_LSTM_singular_model(model, merged_kernel=False)

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

def apply_pruning_to_LSTM(layer):
    # pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(
    #     target_sparsity=.7, begin_step=0)
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0, final_sparsity=.5, begin_step=0, end_step=5000, power=1)
    if not isinstance(layer, keras.layers.TimeDistributed):
        return tfmot.sparsity.keras.prune_low_magnitude(layer,pruning_schedule)
    return layer

smodel = keras.models.clone_model(smodel, clone_function=apply_pruning_to_LSTM)
smodel.compile(
    loss="mse",
    optimizer="adam"
)

X_mini, y_mini = split_train_random(6400, 100)

smodel.fit(X_mini, y_mini, batch_size=32, validation_data=(X_test,y_test), epochs=25, 
           callbacks=[pruning_callbacks.UpdatePruningStep()])
#%% created reduced model
rmodel = make_LSTM_reduced_model(smodel, merged_kernel=False, cutoff=.05)

start_time = time.perf_counter()
fy = model.predict(X_test)
print("full model timing: " + str(time.perf_counter() - start_time) + " sec")
start_time = time.perf_counter()
ry = rmodel.predict(X_test)
print("reduced model timing: " + str(time.perf_counter() - start_time) + " sec")
plt.figure(figsize=(7,3.3))
plt.title("LSTM prediction of pin location")
plt.plot(t_test[0], ry[0], label = "predicted pin location")
plt.plot(t_test[0], y_test[0], label = "actual pin location",alpha=.8)
plt.xlabel("time [s]")
plt.ylabel("pin location [m]")
# plt.ylim((0.045, .23))
plt.legend(loc=1)
plt.tight_layout()

#%% analysis\
from sklearn.metrics import mean_squared_error
from math import sqrt

# count weights in rmodel
reduced_weights = 0
for weight_matrix in rmodel.get_weights():
    reduced_weights += weight_matrix.size
full_weights = 0
for weight_matrix in model.get_weights():
    full_weights += weight_matrix.size

print("%d weights in full matrix"%full_weights)
print("%d weights in reduced matrix"%reduced_weights)
print("%f percent reduction in weights"%(100 - reduced_weights/full_weights*100))

ry_scaled = pin_scaler.inverse_transform(ry.squeeze())
my_scaled = pin_scaler.inverse_transform(fy.squeeze())
y_test_scaled = pin_scaler.inverse_transform(y_test.squeeze())

rmse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_test_scaled, ry_scaled)])/y_test.shape[0]
rrmse = sqrt(sum([mean_squared_error(y_t, y_p, squared=False)**2 for y_t, y_p in zip(y_test_scaled, ry_scaled)])/y_test.shape[0])
mmse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_test_scaled, my_scaled)])/y_test.shape[0]
mrmse = sqrt(sum([mean_squared_error(y_t, y_p, squared=False)**2 for y_t, y_p in zip(y_test_scaled, my_scaled)])/y_test.shape[0])

print("%f m RMSE of reduced model"%(rrmse))
print("%f m RMSE of full model"%(mrmse))
print("%f percent increase in RMSE"%(rrmse/mrmse*100-100))


rsnr = signaltonoise(y_test_scaled, ry_scaled)
msnr = signaltonoise(y_test_scaled, my_scaled)

print("%f dB SNR of reduced model"%rsnr)
print("%f dB SNR of full model"%msnr)
print("%f dB reduction of model SNR"%(msnr-rsnr))

# start_time = time.perf_counter()
# sy = smodel.predict(X_test)
# print("singular model timing: " + str(time.perf_counter() - start_time) + " sec")
# plt.figure(figsize=(7,3.3))
# plt.title("LSTM prediction of pin location")
# plt.plot(t_test[0], sy[0], label = "predicted pin location")
# plt.plot(t_test[0], y_test[0], label = "actual pin location",alpha=.8)
# plt.xlabel("time [s]")
# plt.ylabel("pin location [m]")
# # plt.ylim((0.045, .23))
# plt.legend(loc=1)
# plt.tight_layout()